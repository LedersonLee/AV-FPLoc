"""VAL-locked colleague mode gate versus frozen E+C+N fixed equal fusion.

No retraining, no new RIR, no test-selected scalars, and no proposed top-K policy
for our method. Exact imported reference code defines the colleague decision.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import csv
import fcntl
from pathlib import Path
import shutil
import time
import traceback

import numpy as np

from .contracts import atomic_json,immutable_json,sha
from .data import read
from .colleague_projection import REL as PROJECTION,CORRECTED,membership,schedule_at,load_raw,describe
from .colleague_mode_fusion import PROTOCOL,GRID,reference,evidence,choose,select_val
from .pilot_evaluate import angular_rays,bank_windows,heat_values

REL='AcousticContrastive/colleague_mode_fusion_v2'
ORIGINAL='AcousticContrastive/pilots/scene_subset_v1/input_ablation_v1'
LABELS=dict(vision_only='Vision-only · UnLoc epoch20',colleague_acoustic_xy='Colleague acoustic · XY only',
            colleague_mode_gate='Colleague acoustic + spatial modes + gate',
            ours_acoustic='Ours · E+C+N acoustic',ours_equal_fusion='Ours · E+C+N + fixed 1:1 fusion')
METHODS=tuple(LABELS)
KEYS=('recall_0.1m','recall_0.5m','recall_1m','recall_1m_30deg',
      'yaw_within_10deg','yaw_within_30deg','yaw_within_50deg',
      'median_xy_m','median_yaw_deg','repair','harm')


def metrics(records,method):
    values=[r['methods'][method] for r in records]
    xy=np.array([r['xy_error_m'] for r in values]);yaw=[r['yaw_error_deg'] for r in values]
    v_ok=np.array([r['methods']['vision_only']['xy_error_m']<=1 for r in records])
    result={f'recall_{t:g}m':float((xy<=t).mean()) for t in (.1,.5,1)}
    if all(y is None for y in yaw):
        result.update({k:None for k in ('recall_1m_30deg','yaw_within_10deg','yaw_within_30deg','yaw_within_50deg','median_yaw_deg')})
    else:
        if any(y is None for y in yaw):raise ValueError('Mixed known/unknown heading')
        yaw=np.array(yaw);result['recall_1m_30deg']=float(((xy<=1)&(yaw<=30)).mean())
        result.update({f'yaw_within_{t}deg':float((yaw<=t).mean()) for t in (10,30,50)})
        result['median_yaw_deg']=float(np.median(yaw))
    result.update(queries=len(records),median_xy_m=float(np.median(xy)),
                  repair=int((~v_ok&(xy<=1)).sum()),harm=int((v_ok&(xy>1)).sum()))
    return result


def pose_record(cell,yaw,xy,truth):
    pose=[*map(float,xy[cell]),None if yaw is None else int(yaw)*10]
    return dict(cell=int(cell),pose_xzyaw=pose,xy_error_m=float(np.linalg.norm(xy[cell]-truth[:2])),
                yaw_error_deg=None if yaw is None else float(abs((pose[2]-truth[2]+180)%360-180)))


def raw_visual(windows,location,scale):
    rays,scales=angular_rays(location,scale)
    return -(abs(windows-rays)/scales).sum(-1)/40


def load_w(root,device):
    import torch
    folder=root/PROJECTION/'runs/linear_yaw0';done=read(folder/'complete.json')
    if done['status']!='complete' or done['best_sha256']!=sha(folder/'best.pt'):
        raise ValueError('Colleague VAL-selected checkpoint changed')
    return torch.load(folder/'best.pt',map_location='cpu',weights_only=False)['W'].to(device)


def project(features,cells,W,device):
    import torch
    with torch.inference_mode():
        bank=torch.tensor(features[:cells],device=device)@W
        query=torch.tensor(features[cells:],device=device)@W
        scores=np.concatenate([-torch.cdist(query[i:i+32],bank,p=1).cpu().numpy()
                               for i in range(0,len(query),32)])
    return scores


def cache_val(root,entry,refdir,config_sha,workers,status):
    scene=entry['scene_id'];folder=root/REL/'val_features'/scene;folder.mkdir(parents=True,exist_ok=True)
    fp=schedule_at(root,entry['schedule_path'],'val',scene)
    observed=schedule_at(root,entry['observation_schedule_path'],'val',scene)
    expected=dict(config_sha256=config_sha,fp_schedule_sha256=fp['_sha256'],
                  observation_schedule_sha256=observed['_sha256'])
    receipt_path=folder/'receipt.json'
    if receipt_path.exists():
        receipt=read(receipt_path)
        if any(receipt[k]!=v for k,v in expected.items()) or sha(folder/'features.npy')!=receipt['features_sha256']:
            raise ValueError('Val projection cache changed')
        return np.load(folder/'features.npy'),observed['queries']
    if any(0 not in p['base_headings'] for p in fp['positions']):raise ValueError('Missing yaw-zero FP')
    work=[(f'RIR/FP/val/{scene}/records/p{p["pose_id"]:05d}/base_acoustic_yaw_000.npy',fp,'FP')
          for p in fp['positions']]
    paths=list(dict.fromkeys(q['observed_rir_path'] for q in observed['queries']))
    work += [(p,observed,'Observed') for p in paths]
    rolls={p:sorted({q['channel_roll'] for q in observed['queries'] if q['observed_rir_path']==p}) for p in paths}
    def one(item):
        path,schedule,role=item
        array,digest=load_raw(root,path,schedule,role,single_read=True)
        features=(describe(array,refdir/'grid_score.py') if role=='FP' else
                  {roll:describe(np.roll(array,roll,axis=0),refdir/'grid_score.py') for roll in rolls[path]})
        return features,dict(path=path,sha256=digest)
    values=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i,value in enumerate(pool.map(one,work)):
            values.append(value)
            if (i+1)%500==0:status('val_feature_extraction',scene=scene,physical_complete=i+1,physical_total=len(work))
    cells=len(fp['positions']);lut={p:values[cells+i][0] for i,p in enumerate(paths)}
    # Apply the physical recipe before feature extraction. Although row rolling
    # commutes algebraically, float32 normalization changes its summation order.
    # Reuse the raw read, not an approximately equivalent quantized feature roll.
    queries=observed['queries']
    qfeatures=[lut[q['observed_rir_path']][q['channel_roll']] for q in queries]
    array=np.concatenate([np.stack([v[0] for v in values[:cells]]),np.stack(qfeatures)])
    temporary=folder/'features.pending.npy';np.save(temporary,array);temporary.replace(folder/'features.npy')
    immutable_json(receipt_path,dict(**expected,features_sha256=sha(folder/'features.npy'),cells=cells,
        queries=len(queries),raw_records=[v[1] for v in values],physical_reads=len(work),test_accessed=False))
    return array,queries


def build_report(report,summary,records,banks):
    report.mkdir(parents=True,exist_ok=True)
    atomic_json(report/'summary.json',summary);atomic_json(report/'records.json',records);atomic_json(report/'banks.json',banks)
    with (report/'metrics.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['method',*KEYS])
        for method,m in summary['micro'].items():writer.writerow([method,*[m[k] for k in KEYS]])
    shutil.copy2(Path(__file__).with_name('colleague_mode_report.html'),report/'index.html')


def run(args):
    import torch
    root,vision,report,refdir=args.root,args.vision_root,args.report,args.reference
    out=root/REL;out.mkdir(parents=True,exist_ok=True);report.mkdir(parents=True,exist_ok=True)
    lock=(out/'run.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def status(stage,**extra):
        atomic_json(out/'status.json',dict(status='running',stage=stage,updated_unix=time.time(),**extra))
    selected=membership(root)
    old_config=read(root/PROJECTION/'config.json')
    if sha(refdir/'grid_score.py')!=old_config['reference_sha256']:raise ValueError('Reference feature differs from trained W')
    ours=root/ORIGINAL/'runs/all/best.pt'
    contract=dict(version='colleague_mode_fusion_hssd_v2',protocol=PROTOCOL,scalar_grid=GRID,
        observed_feature_recipe='Exact physical channel roll before reference phi; single raw read per physical record',
        subset_sha256=sha(root/'AcousticContrastive/pilots/scene_subset_v1/scene_splits.json'),
        reference={p.name:sha(p) for p in refdir.iterdir() if p.is_file()},
        colleague_checkpoint_sha256=sha(root/PROJECTION/'runs/linear_yaw0/best.pt'),
        ours_checkpoint_sha256=sha(ours),ours_policy='fixed 1:1 independently z-scored JOINT XY/yaw scores',
        visual_checkpoint_sha256=sha(vision/'UnLoc/evaluation_checkpoints/epoch020/model.ckpt'),
        grid_spacing_m=.2,fp_clearance_m=.06,source_grid_m=.1,source_clearance_m=.25,
        adaptation='Same HSSD comparison support; native Y-up extrusion; actual observed channel order; no new RIR',
        test_fitting=False,gate_selection='VAL only; no retraining or test reselection')
    immutable_json(out/'contract.json',contract)
    torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu';W=load_w(root,device)
    try:
        selection_path=out/'selection_frozen.json'
        if not selection_path.exists():
            rows={};truth={};positions={};proofs={}
            index=read(root/'ReliabilityFusion/pilots/v1/bank_index.json')
            entries=[e for e in index['scenes'] if e['split']=='val']
            if {e['scene_id'] for e in entries}!=set(selected['val']):raise ValueError('Wrong val scenes')
            for entry in entries:
                scene=entry['scene_id'];status('val_features_and_modes',scene=scene,test_accessed=False)
                array,queries=cache_val(root,entry,refdir,sha(out/'contract.json'),args.workers,status)
                xy=np.asarray(entry['candidate_xz']);positions[scene]=xy
                scores=project(array,len(xy),W,device)
                path=root/f'ReliabilityFusion/pilots/v1/vision/val/{scene}.npz';receipt=read(path.with_suffix('.json'))
                if sha(path)!=receipt['data_sha256'] or receipt['checkpoint_sha256']!=contract['visual_checkpoint_sha256']:
                    raise ValueError('Frozen VAL visual predictions changed')
                if receipt['observation_schedule_sha256']!=entry['observation_schedule_sha256']:
                    raise ValueError('Val visual/query schedule differs')
                with np.load(path) as blob:
                    if list(blob['sample_indices'])!=[q['sample_index'] for q in queries]:raise ValueError('Val query order differs')
                    location,scale=blob['location'],blob['scale']
                payload=vision/f'UnLoc/structural_targets/{scene}/desdf.npy'
                windows=bank_windows(np.load(payload,allow_pickle=True).item(),xy)
                truth[scene]=np.array([[q['centre_xyz_m'][0],q['centre_xyz_m'][2],q['world_yaw_deg']] for q in queries])
                rows[scene]=[evidence(raw_visual(windows,location[i],scale[i]),scores[i],xy,.2,refdir)
                             for i in range(len(queries))]
                proofs[scene]=dict(feature_receipt_sha256=sha(root/REL/'val_features'/scene/'receipt.json'),
                    visual_prediction_sha256=sha(path),desdf_sha256=sha(payload),queries=len(queries),cells=len(xy))
            status('val_scalar_selection',test_accessed=False)
            selection=select_val(rows,truth,positions,refdir)
            immutable_json(selection_path,dict(**selection,contract_sha256=sha(out/'contract.json'),val_sources=proofs,
                selected_before_test_phase=True))
        selection=read(selection_path)
        if selection['contract_sha256']!=sha(out/'contract.json') or selection['test_accessed']:
            raise ValueError('Val selection contract changed')
        scalars=selection['selected']['scalars']
        # Test-derived inputs are opened only after immutable VAL selection.
        status('test_evaluation',test_accessed=True,selected_scalars=scalars)
        sources=vision/'AcousticContrastive/pilots/scene_subset_v1/evaluation/report/input-ablation-dense/all'
        source_summary=read(sources/'summary.json');source_records=read(sources/'records.json')
        if source_summary['acoustic_checkpoint_sha256']!=contract['ours_checkpoint_sha256']:
            raise ValueError('Our frozen three-input result/checkpoint differs')
        old_records=read(root/PROJECTION/'report/records.json')
        old_lut={(r['scene_id'],r['sample_index']):r for r in old_records}
        lookup={(r['scene_id'],r['sample_index']):r for r in source_records}
        if len(lookup)!=1472 or len(old_lut)!=1472:raise ValueError('Wrong test query scope')
        bank_root=root/CORRECTED/'evaluation';index=read(bank_root/'bank_index.json');ready=read(bank_root/'ready.json')
        if ready['status']!='complete' or ready['bank_sha256']!=sha(bank_root/'bank_index.json'):
            raise ValueError('Corrected test bank not verified')
        if {e['scene_id'] for e in index['scenes']}!=set(selected['test']):raise ValueError('Wrong test scenes')
        source_banks=read(sources/'banks.json');records=[];banks={};test_proofs={}
        for entry in index['scenes']:
            scene=entry['scene_id'];xy=np.asarray(entry['candidate_xz']);status('test_scene',scene=scene,test_accessed=True)
            np.testing.assert_allclose(xy,source_banks[scene]['xy'],atol=1e-10)
            schedule=schedule_at(root,entry['observation_schedule_path'],'test',scene);queries=schedule['queries']
            fp=schedule_at(root,entry['schedule_path'],'test',scene)
            cache=root/PROJECTION/'test_features'/scene;receipt=read(cache/'receipt.json')
            if (receipt['reference_sha256']!=sha(refdir/'grid_score.py') or receipt['features_sha256']!=sha(cache/'features.npy')
                or receipt['fp_schedule_sha256']!=fp['_sha256'] or receipt['observation_schedule_sha256']!=schedule['_sha256']
                or receipt['fp_count']!=len(xy) or receipt['observed_count']!=len(queries)):
                raise ValueError('Frozen colleague test features changed')
            audio=project(np.load(cache/'features.npy'),len(xy),W,device)
            visual_eval=args.vision_eval;depthpath=visual_eval/'depth'/(scene+'.npz');rowpath=visual_eval/'rows'/(scene+'.json')
            frozen=read(vision/'AcousticContrastive/pilots/scene_subset_v1/evaluation/report/corrected-test/summary.json')['frozen_inputs'][scene]
            if sha(depthpath)!=frozen['depth_sha256'] or sha(rowpath)!=frozen['rows_sha256']:raise ValueError('Test visual source changed')
            qlut={r['sample_index']:i for i,r in enumerate(read(rowpath))}
            with np.load(depthpath) as blob:location,scale=blob['location'],blob['scale']
            desdf=vision/f'UnLoc/structural_targets/{scene}/desdf.npy'
            if sha(desdf)!=frozen['desdf_sha256']:raise ValueError('Test DESDF changed')
            windows=bank_windows(np.load(desdf,allow_pickle=True).item(),xy)
            banks[scene]=copy.deepcopy(source_banks[scene]);banks[scene]['map']='maps/'+scene+'.png'
            (report/'maps').mkdir(exist_ok=True);shutil.copy2(sources/'maps'/(scene+'.png'),report/'maps'/(scene+'.png'))
            (report/'heatmaps').mkdir(exist_ok=True)
            for i,q in enumerate(queries):
                source=lookup[(scene,q['sample_index'])];old=old_lut[(scene,q['sample_index'])]
                gt=np.array([q['centre_xyz_m'][0],q['centre_xyz_m'][2],q['world_yaw_deg']])
                np.testing.assert_allclose(gt,source['gt_pose_xzyaw'],atol=1e-7)
                j=qlut[q['sample_index']];v=raw_visual(windows,location[j],scale[j])
                cell,yaw=np.unravel_index(v.argmax(),v.shape)
                np.testing.assert_allclose([*xy[cell],yaw*10],source['methods']['vision_only']['pose_xzyaw'],atol=1e-7)
                ac=int(audio[i].argmax())
                np.testing.assert_allclose(xy[ac],old['methods']['projection_acoustic_xy']['pose_xzyaw'][:2],atol=1e-7)
                ev=evidence(v,audio[i],xy,.2,refdir);decision=choose(ev,scalars,refdir)
                row={k:copy.deepcopy(source[k]) for k in ('scene_id','sample_index','gt_pose_xzyaw',
                    'predicted_visual_rays','gt_nearest_fp_rays','existing_QA_flags')}
                row['methods']=dict(vision_only=copy.deepcopy(source['methods']['vision_only']),
                    colleague_acoustic_xy=pose_record(ac,None,xy,gt),
                    colleague_mode_gate=pose_record(decision['cell'],decision['yaw_bin'],xy,gt),
                    ours_acoustic=copy.deepcopy(source['methods']['acoustic_only']),
                    ours_equal_fusion=copy.deepcopy(source['methods']['equal_weight_fusion']))
                distance=np.linalg.norm(xy[ev['cells']]-gt[:2],axis=1)
                inside=bool((distance<=1).any());good=row['methods']['colleague_mode_gate']['xy_error_m']<=1
                _,fusion=reference(str(refdir))
                row['gate']=dict(gamma=decision['gamma'],active=decision['audio_active'],selected_mode=decision['index'],
                    visual_gap=fusion.visual_ambiguity(ev['visual']) if len(ev['cells'])>1 else None,
                    proposal_oracle_1m=inside,failure='success' if good else 'verifier_miss' if inside else 'proposal_miss',
                    candidates=[dict(cell=int(c),pose_xzyaw=[*xy[c].tolist(),int(ev['yaw'][k])*10],
                        visual=float(ev['visual'][k]),audio=float(ev['audio'][k]),score=float(decision['mode_scores'][k]),
                        disc_cells=int(ev['disc_sizes'][k])) for k,c in enumerate(ev['cells'])])
                original=np.fromfile(sources/source['heatmap'],dtype=np.uint8).reshape(4,len(xy))
                name=f'{scene}_{q["sample_index"]}.bin'
                np.stack([original[0],heat_values(audio[i]),original[1],original[2]]).tofile(report/'heatmaps'/name)
                row['heatmap']='heatmaps/'+name;row['image']='rgb/'+f'{scene}_{q["sample_index"]}.png'
                row['tags']=[];vok=row['methods']['vision_only']['xy_error_m']<=1
                for method in ('colleague_mode_gate','ours_equal_fusion'):
                    ok=row['methods'][method]['xy_error_m']<=1
                    if not vok and ok:row['tags'].append(method+'_repair')
                    if vok and not ok:row['tags'].append(method+'_harm')
                row['tags'].append('colleague_'+row['gate']['failure'])
                records.append(row)
            test_proofs[scene]=dict(features_sha256=sha(cache/'features.npy'),query_schedule_sha256=schedule['_sha256'],
                                   desdf_sha256=sha(desdf),depth_sha256=sha(depthpath),queries=len(queries))
        if len(records)!=1472 or set((r['scene_id'],r['sample_index']) for r in records)!=set(lookup):
            raise ValueError('Incomplete test query coverage')
        # Reused scores and outputs must reproduce every official baseline metric.
        for method,old_method in [('vision_only','vision_only'),('ours_acoustic','acoustic_only'),('ours_equal_fusion','equal_weight_fusion')]:
            result=metrics(records,method)
            for key in ('recall_0.1m','recall_0.5m','recall_1m','recall_1m_30deg'):
                np.testing.assert_allclose(result[key],source_summary['micro'][old_method][key],atol=1e-12)
        nms_good=sum(r['gate']['proposal_oracle_1m'] for r in records)
        misses=[r for r in records if r['methods']['colleague_mode_gate']['xy_error_m']>1]
        summary=dict(status='complete',queries=1472,scenes=3,labels=LABELS,contract=contract,
            selection=selection,selection_sha256=sha(selection_path),test_sources=test_proofs,
            ours_source_records_sha256=sha(sources/'records.json'),
            micro={m:metrics(records,m) for m in METHODS},
            per_scene={s:{m:metrics([r for r in records if r['scene_id']==s],m) for m in METHODS} for s in banks},
            proposal_diagnostics=dict(coverage_1m=nms_good/1472,covered_queries=nms_good,
                proposal_misses=sum(r['gate']['failure']=='proposal_miss' for r in records),
                verifier_misses=sum(r['gate']['failure']=='verifier_miss' for r in records),
                mean_gamma=float(np.mean([r['gate']['gamma'] for r in records])),
                active_queries=sum(r['gate']['active'] for r in records)),
            limitations=['Original 12/3/3 scene subset, existing exploratory test reuse, not a blind final benchmark',
                'Original colleague NMS/evidence/continuous rule imported; HSSD support .2m/.06m rather than original .1m/.25m',
                'Colleague trained on 818 yaw-zero physical pairs; ours on 6387 multi-yaw queries: method comparison, not controlled architecture ablation',
                'Gate scalars selected by strict<1m VAL micro recall; final common metrics use <= thresholds',
                'XY-only acoustic has no yaw; all joint/yaw metrics are N/A, never zero-filled',
                'Mode fusion has scores only over proposed places, not a dense calibrated likelihood',
                'Our policy is frozen equal fusion, not a trained reliability-aware gate; no top-K policy for ours'])
        (report/'rgb').mkdir(exist_ok=True)
        for row in records:
            name=Path(row['image']).name;target=report/row['image']
            if not target.exists():
                import os
                try:os.link(sources/'rgb'/name,target)
                except OSError:shutil.copy2(sources/'rgb'/name,target)
        status('write_report',test_accessed=True)
        build_report(report,summary,records,banks)
        atomic_json(out/'summary.json',summary)
        atomic_json(out/'verification.json',dict(status='verified',test_queries=1472,
            val_queries=selection['selected']['queries'],selection_frozen_sha256=sha(selection_path),
            report_records_sha256=sha(report/'records.json'),report_summary_sha256=sha(report/'summary.json'),
            original_vision_poses_reproduced=1472,colleague_acoustic_xy_reproduced=1472,
            ours_frozen_metrics_reproduced=True,reference_rule_exact=True,test_fitting=False,new_RIRs=False))
        atomic_json(out/'status.json',dict(status='complete',report=str(report),updated_unix=time.time(),test_fitting=False))
        print(summary['micro'],flush=True)
    except Exception:
        atomic_json(out/'status.json',dict(status='failed',error=traceback.format_exc(),updated_unix=time.time()))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','vision-root','vision-eval','reference','report'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--workers',type=int,default=8);run(p.parse_args())

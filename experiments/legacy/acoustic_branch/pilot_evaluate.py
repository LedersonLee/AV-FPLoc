"""Four-way same-grid evaluation and auditable pilot report, after training.

Uses frozen epoch20 vision depth predictions, never fits knobs on test.
"""
import argparse
from collections import Counter,defaultdict
import csv
import json
from pathlib import Path
import time

import numpy as np

from .contracts import sha,atomic_json
from .data import read,splits,safe_path
from .evaluation import select,diagnose,summarize,standardized,METHODS
from .features import roll_features
from .test_rirs import REL


def angular_rays(depth,scale):
    """Matches UnLoc get_ray_from_depth_uncertainty(V11,F_W3/8)."""
    angles=np.deg2rad(np.arange(-5,6)*10)
    coordinates=np.tan(angles)*len(depth)*3/8+(len(depth)-1)/2
    if coordinates.min()<0 or coordinates.max()>len(depth)-1:raise ValueError('Ray fan exceeds image')
    return np.interp(coordinates,np.arange(len(depth)),depth)/np.cos(angles),np.interp(coordinates,np.arange(len(depth)),scale)


def bank_windows(payload,xy):
    step=payload['desdf_spacing_m'];origin=np.array([payload['origin_x_m'],payload['origin_z_min_m']])
    indices=np.rint((np.asarray(xy)-origin)/step).astype(int)
    reconstructed=origin+indices*step
    if not np.allclose(reconstructed,xy,atol=1e-6,rtol=0):raise ValueError('FP/vision grids misaligned')
    if (indices<0).any() or (indices[:,0]>=payload['desdf'].shape[1]).any() or (indices[:,1]>=payload['desdf'].shape[0]).any():
        raise ValueError('FP bank exceeds vision DESDF')
    field=payload['desdf'][indices[:,1],indices[:,0]]
    if not np.all(np.any(field>0,axis=1)):raise ValueError('FP and vision disagree on valid grid cells')
    bearings=(np.arange(36)[:,None]+np.arange(-5,6))%36
    return field[:,bearings].astype(np.float32)


def feature_arrays(root,ready,scene,role):
    item=ready['features'][scene][role];path=safe_path(root,item['path'])
    if sha(path)!=item['sha256']:raise ValueError('Test feature receipt changed')
    receipt=read(path)
    for name,digest in receipt['files'].items():
        if sha(path.parent/name)!=digest:raise ValueError('Test feature checksum changed')
    return (np.load(path.parent/'energy.npy',mmap_mode='r'),np.load(path.parent/'coherence.npy',mmap_mode='r'),receipt)


def encode(model,energies,coherences,device):
    import torch
    with torch.inference_mode():
        result=model(torch.from_numpy(np.ascontiguousarray(energies,dtype=np.float32)).to(device),
                     torch.from_numpy(np.ascontiguousarray(coherences,dtype=np.float32)).to(device))
    return result.cpu().numpy()


def embeddings(root,entry,ready,model,device,batch):
    scene=entry['scene_id'];e,c,receipt=feature_arrays(root,ready,scene,'FP');cells=entry['cells']
    if len(e)!=cells*6:raise ValueError('Incomplete candidate feature bank')
    result=np.empty((cells,36,128),np.float32)
    from .contracts import recipe
    for yaw in range(36):
        base,roll=recipe(yaw*10)
        for first in range(0,cells,batch):
            ids=np.arange(first,min(first+batch,cells))*6+base//10
            energies=np.roll(np.array(e[ids],np.float32),roll,axis=1)
            coherences=np.roll(np.array(c[ids],np.float32),roll,axis=2)
            result[first:first+len(ids),yaw]=encode(model,energies,coherences,device)
    oe,oc,oreceipt=feature_arrays(root,ready,scene,'Observed')
    schedule=read(root/entry['observation_schedule_path']);queries=schedule['queries']
    lut={path:i for i,path in enumerate(oreceipt['paths'])};observed=[]
    for first in range(0,len(queries),batch):
        values=[]
        for q in queries[first:first+batch]:
            i=lut[q['observed_rir_path']];values.append(roll_features(oe[i],oc[i],q['channel_roll']))
        observed.append(encode(model,np.stack([v[0] for v in values]),np.stack([v[1] for v in values]),device))
    return result,np.concatenate(observed),queries


def heat_values(score):
    score=np.asarray(score,float);out=np.zeros(score.shape,np.uint8);valid=np.isfinite(score)
    if valid.any():
        a=score[valid];out[valid]=np.clip(np.rint((a-a.min())/max(float(np.ptp(a)),1e-12)*254)+1,1,255).astype(np.uint8)
    return out


def paired_intervals(records,draws=1000):
    groups=defaultdict(list)
    for i,r in enumerate(records):groups[(r['scene_id'],*np.round(r['gt_pose_xzyaw'][:2],6))].append(i)
    clusters=list(groups.values());rng=np.random.default_rng(42);output={}
    baseline=np.array([r['methods']['vision_only']['xy_error_m']<=1 for r in records],float)
    for method in METHODS[1:]:
        delta=np.array([r['methods'][method]['xy_error_m']<=1 for r in records],float)-baseline
        values=[]
        for _ in range(draws):
            ids=np.concatenate([clusters[i] for i in rng.integers(len(clusters),size=len(clusters))])
            values.append(float(delta[ids].mean()))
        output[method]=dict(delta_recall_1m=float(delta.mean()),ci95=np.quantile(values,[.025,.975]).tolist(),
            unit='paired physical positions, all retained yaws together',clusters=len(clusters),
            caveat='Within these3 exploratory scenes; not an unseen-scene generalization CI')
    return output


def run(args):
    import torch
    from .train import AcousticEncoder
    root=args.root;pilot=root/'AcousticContrastive/pilots/scene_subset_v1'
    out=pilot/getattr(args,'evaluation_relative','evaluation')
    config_path=getattr(args,'config',None) or pilot/'config.json'
    config=read(config_path)
    run=getattr(args,'run_relative',None) or (read(pilot/'active_acoustic_run.json')['path'] if (pilot/'active_acoustic_run.json').exists() else 'runs/acoustic_v1')
    ready=read(out/'ready.json');index=read(out/'bank_index.json');train=read(pilot/run/'status.json')
    if ready['status']!='complete' or train['status']!='complete':raise ValueError('Training/test bank not complete')
    if ready['bank_sha256']!=sha(out/'bank_index.json') or ready['config_sha256']!=sha(config_path):
        raise ValueError('Test bank provenance mismatch')
    if config.get('corrected_extrusion'):
        verification=read(out/'raw_verification.json')
        if (ready.get('frame_version')!='corrected_yup_v2'
            or ready.get('raw_verification_sha256')!=sha(out/'raw_verification.json')
            or verification.get('status')!='complete'
            or verification.get('bank_sha256')!=sha(out/'bank_index.json')):
            raise ValueError('Corrected test raw verification missing/stale')
    if {e['scene_id'] for e in index['scenes']}!=set(splits(root,config)['test']):raise ValueError('Test scene scope changed')
    checkpoint=pilot/run/'best.pt';checkpoint_sha=sha(checkpoint)
    source=read(args.vision_eval/'protocol.json');vision_checkpoint=args.vision_hssd/'UnLoc/evaluation_checkpoints/epoch020/model.ckpt'
    if sha(vision_checkpoint)!=source['checkpoint_sha256']:raise ValueError('Frozen vision checkpoint changed')
    value=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if value['config']!=config:raise ValueError('Acoustic checkpoint config mismatch')
    device=torch.device(args.device);torch.set_num_threads(4)
    model=AcousticEncoder(**config['model']).to(device);model.load_state_dict(value['model']);model.eval()
    report=args.report;report.mkdir(parents=True,exist_ok=True);(report/'heatmaps').mkdir(exist_ok=True)
    records=[];scene_summaries={};scene_banks={};frozen_inputs={};k_summaries={};exclusions={}
    started=time.monotonic()
    for entry in index['scenes']:
        scene=entry['scene_id'];xy=np.asarray(entry['candidate_xz']);scene_records=[];ks={k:[] for k in [5,10,20,50]}
        candidate,observed,queries=embeddings(root,entry,ready,model,device,args.batch_size)
        np.save(out/f'{scene}_candidate_embeddings_{checkpoint_sha[:12]}.npy',candidate)
        np.save(out/f'{scene}_observed_embeddings_{checkpoint_sha[:12]}.npy',observed)
        payload_path=args.vision_hssd/'UnLoc/structural_targets'/scene/'desdf.npy'
        payload=np.load(payload_path,allow_pickle=True).item();windows=bank_windows(payload,xy)
        rowpath=args.vision_eval/'rows'/f'{scene}.json';depthpath=args.vision_eval/'depth'/f'{scene}.npz'
        rows=read(rowpath);rowlut={r['sample_index']:i for i,r in enumerate(rows)}
        with np.load(depthpath) as data:depth=data['location'].copy();scale=data['scale'].copy()
        if depth.shape!=(len(rows),40) or scale.shape!=depth.shape or not (scale>0).all():raise ValueError('Frozen visual predictions invalid')
        frozen_inputs[scene]=dict(rows_sha256=sha(rowpath),depth_sha256=sha(depthpath),desdf_sha256=sha(payload_path),
            candidate_schedule_sha256=entry['schedule_sha256'],observation_schedule_sha256=entry['observation_schedule_sha256'])
        for i,q in enumerate(queries):
            j=rowlut[q['sample_index']];truth=[q['centre_xyz_m'][0],q['centre_xyz_m'][2],q['world_yaw_deg']]
            if not np.allclose(rows[j]['pose_xzyaw'],truth,rtol=0,atol=1e-6):raise ValueError('RGB/Observed/vision query pose mismatch')
            rays,scales=angular_rays(depth[j],scale[j])
            v=-(abs(windows-rays)/scales).sum(axis=-1)/40
            a=np.einsum('cyd,d->cy',candidate,observed[i],optimize=True)
            result=diagnose(v,a,xy,truth,50)
            for k in ks:
                kresult=diagnose(v,a,xy,truth,k);ks[k].append(kresult)
            flags=rows[j].get('furnished_qa',{}).get('warning_reasons',[]);tags=[]
            if not result['diagnostics']['topk_oracle_1m']:tags.append('topk_exclusion')
            elif result['methods']['visual_topk_acoustic_rerank']['xy_error_m']>1:tags.append('acoustic_selection_failure')
            for method in METHODS:
                m=result['methods'][method]
                if m['xy_error_m']<=1 and m['yaw_error_deg']>30:tags.append(method+'_yaw_only')
                if m['vs_vision_1m'] in ('repair','harm'):tags.append(method+'_'+m['vs_vision_1m'])
            if flags:tags.append('existing_geometry_QA_flag')
            if float(np.ptp(a))<1e-6:tags.append('degenerate_acoustic_scores')
            top=select(v,a,50)[1];topheat=np.full(len(xy),np.nan);topheat[top]=a.max(axis=1)[top]
            heat=np.stack([heat_values(v.max(axis=1)),heat_values(a.max(axis=1)),
                heat_values((.5*standardized(v)+.5*standardized(a)).max(axis=1)),heat_values(topheat)])
            heatname=f'{scene}_{q["sample_index"]}.bin';heat.tofile(report/'heatmaps'/heatname)
            result.update(scene_id=scene,sample_index=q['sample_index'],gt_pose_xzyaw=truth,image_path=q['image_path'],
                heatmap='heatmaps/'+heatname,tags=tags,existing_QA_flags=flags,
                geometry_caveat='Existing QA flags indicate review risk, not confirmed geometry error or model failure cause',
                score_statistics=dict(acoustic_std=float(a.std()),acoustic_range=float(np.ptp(a)),visual_std=float(v.std())),
                predicted_visual_rays=rays.tolist(),predicted_visual_scales=scales.tolist(),
                gt_nearest_fp_rays=windows[int(np.linalg.norm(xy-np.array(truth[:2]),axis=1).argmin()),int(round(truth[2]/10))%36].tolist())
            records.append(result);scene_records.append(result)
        scene_summaries[scene]=summarize(scene_records)
        scene_banks[scene]=dict(xy=xy.tolist(),cells=len(xy),query_count=len(queries),grid_spacing_m=.2,
            floorplan_metadata=read(root/'2DFP'/scene/'floorplan_metadata.json'))
        original=read(root/entry['observation_schedule_path']);exclusions[scene]=original['excluded']
        k_summaries[scene]={str(k):dict(metrics=summarize(vals),oracle_recall_1m=float(np.mean([r['diagnostics']['topk_oracle_1m'] for r in vals]))) for k,vals in ks.items()}
        atomic_json(out/'evaluation_progress.json',dict(status='running',completed_scenes=len(scene_summaries),updated_unix=time.time()))
        print(f'[evaluated] {scene}: {len(queries)} queries / {len(xy)} cells',flush=True)
    summary=dict(status='complete',micro=summarize(records),per_scene=scene_summaries,k_diagnostics=k_summaries,
        queries=len(records),scenes=len(scene_summaries),acoustic_checkpoint_sha256=checkpoint_sha,
        acoustic_epoch=value['epoch'],vision_checkpoint_sha256=source['checkpoint_sha256'],
        bank_sha256=sha(out/'bank_index.json'),frozen_inputs=frozen_inputs,excluded_query_records=exclusions,
        corrected_frame=bool(config.get('corrected_extrusion')),
        tags=dict(Counter(tag for r in records for tag in r['tags'])),paired_pose_bootstrap=paired_intervals(records),
        elapsed_seconds=time.monotonic()-started,
        limitations=['Exploratory3scenes, including historicalscene5','Vision trained116scenes vs acoustic12',
            '0.2m grid imposes a quantization ceiling at0.1m','Geometry flags are not causal attribution',
            'Equal-weight z scores are not calibrated probabilities'])
    history_path=pilot/run/'history.json'
    summary['training_diagnostics']=dict(best_epoch=value['epoch'],last_epoch=train['epoch'],best_val_loss=train['best_val_loss'],
        history=read(history_path) if history_path.exists() else None,
        interpretation='Train loss improvement is not evidence of held-out pose retrieval; inspect val generalization separately.')
    oracle=np.array([r['diagnostics']['topk_oracle_1m'] for r in records],bool)
    rerank=np.array([r['methods']['visual_topk_acoustic_rerank']['xy_error_m']<=1 for r in records],bool)
    quantization=np.array([r['diagnostics']['gt_nearest_xy_quantization_error_m'] for r in records])
    summary['failure_analysis']=dict(topk_oracle_recall_1m=float(oracle.mean()),
        topk_exclusion_count=int((~oracle).sum()),acoustic_selection_failure_count=int((oracle&~rerank).sum()),
        topk_success_given_oracle=float(rerank[oracle].mean()) if oracle.any() else None,
        full_bank_oracle={f'recall_{t:g}m':float((quantization<=t).mean()) for t in (.1,.5,1)},
        existing_QA_flag_count=sum(bool(r['existing_QA_flags']) for r in records))
    base=summary['micro']['vision_only']['recall_1m'];topk=summary['micro']['visual_topk_acoustic_rerank']['recall_1m']
    fusion=summary['micro']['equal_weight_fusion']['recall_1m'];interval=summary['paired_pose_bootstrap']['visual_topk_acoustic_rerank']['ci95']
    summary['interpretations']=[
        f'Vision R@1m={base*100:.2f}%; Top-K={topk*100:.2f}%; delta={(topk-base)*100:+.2f}pp; paired-position CI=[{interval[0]*100:+.2f},{interval[1]*100:+.2f}]pp.',
        '이 pilot에서 Top-K 개선의 관측이 있습니다. 3개 scene 내 탐색적 결과이며 일반화된 필요성의 증명은 아닙니다.' if topk>base else
        '이 pilot에서는 Top-K가 vision-only를 개선하지 못했습니다. Top-K를 채택해야 한다는 근거가 확보됐다고 볼 수 없습니다.',
        f'Top-K oracle R@1m={oracle.mean()*100:.2f}%. 후보 누락={int((~oracle).sum())}; 후보가 있지만 음향 선택 실패={int((oracle&~rerank).sum())}.',
        f'전역 1:1 fusion R@1m={fusion*100:.2f}%. 이 값이 Top-K와 같거나 높다면 후보 제한이 필수라는 근거는 더 약합니다.',
        '기존 geometry QA 플래그는 검토 단서입니다. 자연스러운 clutter·ambiguity와 실제 데이터 불일치는 RGB/도면을 함께 확인해야 구분할 수 있습니다.',
        f'Acoustic best epoch={value["epoch"]}, last epoch={train["epoch"]}. Train loss 감소만으로 unseen-scene 위치 추정이 학습됐다고 결론내리지 않습니다.']
    summary['macro']={m:{k:float(np.mean([s[m][k] for s in scene_summaries.values()])) for k in scene_summaries[next(iter(scene_summaries))][m]
                       if k.startswith('recall_')} for m in METHODS}
    atomic_json(report/'summary.json',summary);atomic_json(report/'records.json',records);atomic_json(report/'banks.json',scene_banks)
    with (report/'metrics.csv').open('w',newline='') as stream:
        names=['scope','method',*summary['micro'][METHODS[0]].keys()];writer=csv.DictWriter(stream,fieldnames=names);writer.writeheader()
        for scope,table in [('micro',summary['micro']),*scene_summaries.items()]:
            for method,metrics in table.items():writer.writerow(dict(scope=scope,method=method,**metrics))
    from .pilot_report import build_report
    build_report(root,report,records,scene_banks,summary)
    atomic_json(out/'evaluation_complete.json',dict(status='complete',report=str(report),
        summary_sha256=sha(report/'summary.json'),updated_unix=time.time()))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--vision-hssd',type=Path,required=True);p.add_argument('--vision-eval',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True);p.add_argument('--device',default='cuda');p.add_argument('--batch-size',type=int,default=256)
    run(p.parse_args())


if __name__=='__main__':main()

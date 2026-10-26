"""Same HSSD test queries/grid: yaw-zero colleague projection versus CNN branch.

No test fitting. Acoustic-only projection has NO yaw estimate; joint metrics
are N/A. Fusion broadcasts the XY acoustic score and keeps visual yaw.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import time

import numpy as np

from .colleague_projection import (REL,CORRECTED,describe,load_raw,membership,schedule_at)
from .contracts import atomic_json,immutable_json,sha
from .data import read
from .evaluation import standardized
from .pilot_evaluate import angular_rays,bank_windows,heat_values


def metrics(records,method):
    values=[r['methods'][method] for r in records]
    xy=np.array([v['xy_error_m'] for v in values])
    headings=[v['yaw_error_deg'] for v in values]
    result={f'recall_{t:g}m':float((xy<=t).mean()) for t in (.1,.5,1)}
    result.update(queries=len(values),median_xy_m=float(np.median(xy)),
        recall_1m_30deg=None if headings[0] is None else float(((xy<=1)&(np.array(headings)<=30)).mean()))
    base=np.array([r['methods']['vision_only']['xy_error_m']<=1 for r in records])
    result.update(repair=int((~base&(xy<=1)).sum()),harm=int((base&(xy>1)).sum()))
    return result


def choose(v,a,xy,truth):
    """a is XY-only, with yaw intentionally absent from acoustic output."""
    vc,vy=np.unravel_index(v.argmax(),v.shape)
    fused=standardized(v)+standardized(a)[:,None]
    fc,fy=np.unravel_index(fused.argmax(),fused.shape)
    top=np.argsort(-v.max(axis=1),kind='stable')[:50]
    tc=int(top[a[top].argmax()]);ac=int(a.argmax())
    choices={'vision_only':(int(vc),int(vy)), 'projection_acoustic_xy':(ac,None),
        'projection_equal_fusion':(int(fc),int(fy)),
        'projection_top50_rerank':(tc,int(v[tc].argmax()))}
    result={}
    for name,(cell,yaw) in choices.items():
        result[name]=dict(cell=cell,pose_xzyaw=[*xy[cell].tolist(),None if yaw is None else yaw*10],
            xy_error_m=float(np.linalg.norm(xy[cell]-truth[:2])),
            yaw_error_deg=None if yaw is None else float(abs((yaw*10-truth[2]+180)%360-180)))
    return result,fused


def evaluate(args):
    import torch
    root=args.root;out=root/REL;report=args.report;report.mkdir(parents=True,exist_ok=True)
    model_dir=out/'runs/linear_yaw0';selected=read(model_dir/'complete.json')
    if selected['best_sha256']!=sha(model_dir/'best.pt'):raise ValueError('Selected W changed')
    config=read(out/'config.json')
    if config['reference_sha256']!=sha(args.reference):raise ValueError('Reference feature changed')
    checkpoint=torch.load(model_dir/'best.pt',map_location='cpu',weights_only=False)
    W=checkpoint['W'].to(args.device);torch.set_num_threads(4)
    bank_root=root/CORRECTED/'evaluation';index=read(bank_root/'bank_index.json')
    ready=read(bank_root/'ready.json')
    if ready['status']!='complete' or ready['bank_sha256']!=sha(bank_root/'bank_index.json'):
        raise ValueError('Corrected test bank not ready')
    if {e['scene_id'] for e in index['scenes']}!=set(membership(root)['test']):raise ValueError('Wrong test scene scope')
    old=args.vision_root/'AcousticContrastive/pilots/scene_subset_v1/evaluation/report/corrected-test'
    old_summary=read(old/'summary.json');old_records=read(old/'records.json')
    old_lut={(r['scene_id'],r['sample_index']):r for r in old_records}
    if old_summary.get('corrected_frame') is not True:raise ValueError('Legacy baseline forbidden')
    immutable_json(out/'test_selection_lock.json',dict(selected_best_sha256=selected['best_sha256'],
        selected_before_test=True,training_complete_sha256=sha(model_dir/'complete.json'),
        old_test_summary_sha256=sha(old/'summary.json'),old_test_records_sha256=sha(old/'records.json'),
        corrected_bank_sha256=sha(bank_root/'bank_index.json'),reference_sha256=sha(args.reference)))
    records=[];banks={};start=time.monotonic()
    for entry in index['scenes']:
        scene=entry['scene_id'];xy=np.asarray(entry['candidate_xz'])
        fp=schedule_at(root,entry['schedule_path'],'test',scene)
        observed=schedule_at(root,entry['observation_schedule_path'],'test',scene)
        cache=out/'test_features'/scene;cache.mkdir(parents=True,exist_ok=True)
        if not (cache/'receipt.json').exists():
            def feature(item):
                path,schedule,role,roll=item
                a,digest=load_raw(root,path,schedule,role,roll)
                return describe(a,args.reference),dict(path=path,sha256=digest,channel_roll=roll)
            work=[]
            for p in fp['positions']:
                raw=f'RIR/FP/test/{scene}/records/p{p["pose_id"]:05d}/base_acoustic_yaw_000.npy'
                work.append((raw,fp,'FP',0))
            for q in observed['queries']:
                work.append((q['observed_rir_path'],observed,'Observed',q['channel_roll']))
            with ThreadPoolExecutor(max_workers=args.workers) as pool:values=list(pool.map(feature,work))
            array=np.stack([v[0] for v in values]);tmp=cache/'features.tmp.npy'
            np.save(tmp,array);tmp.replace(cache/'features.npy')
            immutable_json(cache/'receipt.json',dict(features_sha256=sha(cache/'features.npy'),
                reference_sha256=sha(args.reference),fp_schedule_sha256=fp['_sha256'],
                observation_schedule_sha256=observed['_sha256'],fp_count=len(xy),
                observed_count=len(observed['queries']),raw_records=[v[1] for v in values]))
        receipt=read(cache/'receipt.json')
        if (receipt['reference_sha256']!=sha(args.reference) or receipt['fp_schedule_sha256']!=fp['_sha256']
            or receipt['observation_schedule_sha256']!=observed['_sha256'] or
            receipt['features_sha256']!=sha(cache/'features.npy')):
            raise ValueError('Stale test feature cache')
        array=np.load(cache/'features.npy');c=torch.tensor(array[:len(xy)],device=args.device)
        q=torch.tensor(array[len(xy):],device=args.device)
        with torch.inference_mode():
            cp=c@W;qp=q@W
            projected=np.concatenate([-torch.cdist(qp[i:i+32],cp,p=1).cpu().numpy() for i in range(0,len(q),32)])
            identity=np.concatenate([-torch.cdist(q[i:i+16],c,p=1).cpu().numpy() for i in range(0,len(q),16)])
        np.savez(cache/'xy_scores.npz',projected=projected,identity=identity)
        payload_path=args.vision_root/'UnLoc/structural_targets'/scene/'desdf.npy'
        if sha(payload_path)!=old_summary['frozen_inputs'][scene]['desdf_sha256']:
            raise ValueError('DESDF differs from frozen baseline')
        windows=bank_windows(np.load(payload_path,allow_pickle=True).item(),xy)
        rowpath=args.vision_eval/'rows'/(scene+'.json');depthpath=args.vision_eval/'depth'/(scene+'.npz')
        if sha(rowpath)!=old_summary['frozen_inputs'][scene]['rows_sha256'] or sha(depthpath)!=old_summary['frozen_inputs'][scene]['depth_sha256']:
            raise ValueError('Vision input changed')
        rows=read(rowpath);lut={r['sample_index']:i for i,r in enumerate(rows)}
        with np.load(depthpath) as z:depth=z['location'].copy();scale=z['scale'].copy()
        (report/'heatmaps').mkdir(exist_ok=True);(report/'assets'/scene).mkdir(parents=True,exist_ok=True)
        shutil.copy2(root/'2DFP'/scene/'map.png',report/'assets'/scene/'map.png')
        banks[scene]=dict(xy=xy.tolist(),meta=read(root/'2DFP'/scene/'floorplan_metadata.json'),
                          map=f'assets/{scene}/map.png')
        for i,query in enumerate(observed['queries']):
            old_record=old_lut[(scene,query['sample_index'])];j=lut[query['sample_index']]
            truth=np.array([query['centre_xyz_m'][0],query['centre_xyz_m'][2],query['world_yaw_deg']])
            np.testing.assert_allclose(rows[j]['pose_xzyaw'],truth,atol=1e-6,rtol=0)
            rays,scales=angular_rays(depth[j],scale[j])
            v=-(abs(windows-rays)/scales).sum(axis=-1)/40
            methods,fused=choose(v,projected[i],xy,truth)
            np.testing.assert_allclose(methods['vision_only']['pose_xzyaw'],
                old_record['methods']['vision_only']['pose_xzyaw'],atol=1e-7,rtol=0)
            methods.update({k:old_record['methods'][k] for k in
                ('acoustic_only','equal_weight_fusion','visual_topk_acoustic_rerank')})
            rawcell=int(identity[i].argmax())
            methods['untrained_reference_xy']=dict(cell=rawcell,pose_xzyaw=[*xy[rawcell].tolist(),None],
                xy_error_m=float(np.linalg.norm(xy[rawcell]-truth[:2])),yaw_error_deg=None)
            heatname=f'{scene}_{query["sample_index"]}.bin'
            np.stack([heat_values(v.max(axis=1)),heat_values(projected[i]),
                heat_values(fused.max(axis=1)),heat_values(identity[i])]).tofile(report/'heatmaps'/heatname)
            image=report/'assets'/scene/(str(query['sample_index'])+Path(query['image_path']).suffix)
            shutil.copy2(root/query['image_path'],image)
            records.append(dict(scene_id=scene,sample_index=query['sample_index'],gt_pose_xzyaw=truth.tolist(),
                methods=methods,image=str(image.relative_to(report)),heatmap='heatmaps/'+heatname))
        atomic_json(out/'evaluation_progress.json',dict(status='running',completed_scenes=len(banks),updated_unix=time.time()))
        print(f'[evaluated] {scene}: {len(observed["queries"])} queries, {len(xy)} yaw0 FP cells',flush=True)
    if len(records)!=len(old_records):raise ValueError('Unequal query coverage')
    methods=list(records[0]['methods'])
    summary=dict(status='complete',queries=len(records),scenes=len(banks),training=selected,
        micro={m:metrics(records,m) for m in methods},
        per_scene={s:{m:metrics([r for r in records if r['scene_id']==s],m) for m in methods} for s in banks},
        per_query_yaw={str(y):{m:metrics([r for r in records if r['gt_pose_xzyaw'][2]==y],m) for m in methods}
            for y in sorted({r['gt_pose_xzyaw'][2] for r in records})},
        elapsed_seconds=time.monotonic()-start,
        limitations=['Same12/3/3 exploratory scenes; not full HSSD or blind test',
            'HSSD uses the same0.2m grid/0.06m clearance as CNN control, not colleague original0.1m/0.25m',
            'Yaw0 paired training, exact reference feature/linear/L1 objective; room-shuffle RNG additionally seeded',
            'Projection acoustic-only returns XY, no yaw; joint metric N/A',
            'Query physical channel reconstruction only, not GT-yaw rotation compensation',
            'Equal fusion/top50 are controlled comparisons, NOT the colleague documented NMS10/quantile/gate fusion',
            'Identity validation and projected validation use strict<1m, final common test metrics use<='])
    atomic_json(report/'summary.json',summary);atomic_json(report/'records.json',records);atomic_json(report/'banks.json',banks)
    build_report(report,summary)
    atomic_json(out/'evaluation_complete.json',dict(status='complete',summary_sha256=sha(report/'summary.json'),
        test_selection_lock_sha256=sha(out/'test_selection_lock.json'),updated_unix=time.time()))


def build_report(report,summary):
    import html
    names={'vision_only':'Vision epoch20','acoustic_only':'Ours: CNN acoustic joint XY/yaw',
        'equal_weight_fusion':'Ours: CNN + vision 1:1','visual_topk_acoustic_rerank':'Ours: CNN top50',
        'untrained_reference_xy':'Colleague feature: untrained L1 XY',
        'projection_acoustic_xy':'Colleague W: acoustic XY only',
        'projection_equal_fusion':'Colleague W + vision 1:1',
        'projection_top50_rerank':'Colleague W: top50'}
    rows=[]
    for method,m in summary['micro'].items():
        values=[names[method],*(f'{m[f"recall_{t:g}m"]*100:.2f}%' for t in (.1,.5,1)),
            'N/A' if m['recall_1m_30deg'] is None else f'{m["recall_1m_30deg"]*100:.2f}%',str(m['repair']),str(m['harm'])]
        rows.append('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in values)+'</tr>')
    template=Path(__file__).with_name('colleague_report.html').read_text()
    template=template.replace('__ROWS__',''.join(rows)).replace('__LIMITATIONS__',
        ''.join('<li>'+html.escape(v)+'</li>' for v in summary['limitations']))
    temporary=report/'index.tmp.html';temporary.write_text(template);temporary.replace(report/'index.html')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('root','reference','vision-root','vision-eval','report'):p.add_argument('--'+arg,type=Path,required=True)
    p.add_argument('--workers',type=int,default=8);p.add_argument('--device',default='cuda')
    evaluate(p.parse_args())

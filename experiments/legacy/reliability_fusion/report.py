"""Six-way matched test report after val-only checkpoint selection."""
import argparse
import os
from pathlib import Path
import shutil

import numpy as np
import torch

from acoustic_branch.contracts import atomic_json,sha
from acoustic_branch.data import read
from acoustic_branch.evaluation import METHODS
from acoustic_branch.pilot_evaluate import heat_values
from .prepare import REL,ACOUSTIC
from .core import FEATURE_NAMES,metrics,prediction
from .model import Gate

ALL_METHODS=[*METHODS,'constant_weight_fusion','adaptive_weight_fusion']


def publish(root,vision_root,report):
    root,vision_root,report=map(Path,(root,vision_root,report));out=root/REL
    training=read(out/'runs/training_complete.json');assert training['status']=='complete'
    assert sha(out/'runs/adaptive/best.pt')==training['best_sha256']
    models={}
    for mode in ('constant','adaptive'):
        state=torch.load(out/'runs'/mode/'best.pt',map_location='cpu',weights_only=False)
        assert not state['test_accessed'] and state['config_sha256']==sha(out/'config.json')
        model=Gate(len(FEATURE_NAMES),mode=='constant');model.load_state_dict(state['model']);model.eval()
        models[mode]=(model,state)
    ready=read(out/'test_scores_ready.json');assert ready['test_accessed']
    old_report=vision_root/'AcousticContrastive/pilots/scene_subset_v1/evaluation/report/corrected-test'
    original={(r['scene_id'],r['sample_index']):r for r in read(old_report/'records.json')}
    for name in ('heatmaps','maps','rgb'): (report/name).mkdir(parents=True,exist_ok=True)
    records=[];banks={};baseline_reproduced=0
    for entry in ready['entries']:
        scene=entry['scene_id'];folder=root/entry['directory'];receipt=read(folder/'receipt.json')
        assert sha(folder/'receipt.json')==entry['receipt_sha256']
        for name,digest in receipt['files'].items():assert sha(folder/name)==digest
        scores=np.load(folder/'scores.npy',mmap_mode='r');features=np.load(folder/'features.npy')
        truth=np.load(folder/'truth.npy');xy=np.load(folder/'xy.npy');queries=read(folder/'queries.json')
        baseline=read(folder/'baselines.json');weights={}
        with torch.no_grad():
            for mode,(model,state) in models.items():
                x=(features-state['mean'])/state['std'];weights[mode]=model(torch.tensor(x,dtype=torch.float32)).numpy()
        meta=read(root/'2DFP'/scene/'floorplan_metadata.json')
        shutil.copy2(root/'2DFP'/scene/'map.png',report/'maps'/f'{scene}.png')
        banks[scene]=dict(xy=xy.tolist(),cells=len(xy),metadata=meta)
        for i,q in enumerate(queries):
            v,a=scores[i];poses=baseline[i].copy()
            for method in METHODS:
                np.testing.assert_allclose(poses[method],original[(scene,q['sample_index'])]['methods'][method]['pose_xzyaw'],atol=1e-7,rtol=0)
            baseline_reproduced+=1
            for mode in ('constant','adaptive'):
                poses[mode+'_weight_fusion']=prediction(v,a,xy,weights[mode][i])
            values={}
            for method,pose in poses.items():
                distance=float(np.linalg.norm(np.asarray(pose[:2])-truth[i,:2]));angle=float(abs((pose[2]-truth[i,2]+180)%360-180))
                values[method]=dict(pose_xzyaw=pose,xy_error_m=distance,yaw_error_deg=angle)
            v_ok=values['vision_only']['xy_error_m']<=1;new_ok=values['adaptive_weight_fusion']['xy_error_m']<=1
            equal_ok=values['equal_weight_fusion']['xy_error_m']<=1
            record=dict(scene_id=scene,sample_index=q['sample_index'],gt_pose_xzyaw=truth[i].tolist(),methods=values,
                acoustic_weight=float(weights['adaptive'][i]),constant_acoustic_weight=float(weights['constant'][i]),
                features=dict(zip(FEATURE_NAMES,features[i].tolist())),adaptive_repair=not v_ok and new_ok,
                adaptive_harm=v_ok and not new_ok,repair_vs_equal=not equal_ok and new_ok,harm_vs_equal=equal_ok and not new_ok)
            top=np.argsort(-v.max(1),kind='stable')[:50];topheat=np.full(len(xy),np.nan);topheat[top]=a.max(1)[top]
            heat=[heat_values(v.max(1)),heat_values(a.max(1)),heat_values((.5*v+.5*a).max(1)),heat_values(topheat)]
            for mode in ('constant','adaptive'):
                w=float(weights[mode][i]);heat.append(heat_values(((1-w)*v+w*a).max(1)))
            name=f'{scene}_{q["sample_index"]}';np.stack(heat).tofile(report/'heatmaps'/(name+'.bin'))
            source=old_report/'rgb'/(name+'.png');destination=report/'rgb'/(name+'.png')
            if not destination.exists():
                try:os.link(source,destination)
                except OSError:shutil.copy2(source,destination)
            records.append(record)
    summary={};per_scene={}
    for method in ALL_METHODS:
        def summarize(rr):
            result=metrics([r['methods'][method]['pose_xzyaw'] for r in rr],[r['gt_pose_xzyaw'] for r in rr])
            result['repair']=sum(r['methods']['vision_only']['xy_error_m']>1 and r['methods'][method]['xy_error_m']<=1 for r in rr)
            result['harm']=sum(r['methods']['vision_only']['xy_error_m']<=1 and r['methods'][method]['xy_error_m']>1 for r in rr)
            return result
        per_scene[method]={s:summarize([r for r in records if r['scene_id']==s]) for s in banks}
        summary[method]=summarize(records)
        summary[method]['scene_macro_recall_1m']=float(np.mean([x['recall_1m'] for x in per_scene[method].values()]))
    atomic_json(report/'records.json',records);atomic_json(report/'banks.json',banks)
    atomic_json(report/'summary.json',dict(methods=summary,per_scene=per_scene,queries=len(records),
        feature_names=FEATURE_NAMES,training=training,baseline_reproduced_queries=baseline_reproduced,
        test_notice='Existing3 exploratory scenes, not a blind new-scene benchmark. No test fitting.',
        candidate_grid_m=.2,yaw_step_deg=10,range_cap_m=10,config=read(out/'config.json')))
    template=Path(__file__).resolve().parent/'report.html';shutil.copy2(template,report/'index.html')
    for mode in ('constant','adaptive'):
        shutil.copy2(out/'runs'/mode/'history.json',report/(mode+'_history.json'))
    atomic_json(report/'status.json',dict(status='complete',queries=len(records),baseline_reproduced_queries=baseline_reproduced,
        best_checkpoint_sha256=training['best_sha256'],test_fitting=False))
    atomic_json(out/'evaluation_complete.json',dict(status='complete',report=str(report),summary_sha256=sha(report/'summary.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','vision-root','report'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();publish(a.root,a.vision_root,a.report)

"""Independent saved-output checks for frozen cohort coverage and pose metrics."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
from .prepare_targets import read,sha


def main():
    p=argparse.ArgumentParser()
    for key in ['study','cache','run','freeze','out']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--code-sha',required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    assert (a.run/'COMPLETE').exists() and (a.cache/'COMPLETE').exists()
    freeze=read(a.freeze);rp=read(a.run/'protocol.json');cp=read(a.cache/'protocol.json')
    assert rp['freeze_sha256']==cp['freeze_sha256']==sha(a.freeze)
    assert rp['cache_protocol_sha256']==sha(a.cache/'protocol.json')
    for r in freeze['checkpoints']:assert sha(a.study/r['run']/r['model']/'model.pt')==r['sha256']
    corpus={}
    for r in read(a.cache/'manifest.json'):
        path=a.cache/f"{r['scene']}.pt";assert sha(path)==r['sha256'];corpus[r['scene']]=torch.load(path,map_location='cpu',weights_only=False)
    assert list(corpus)==freeze[cp['part']]
    summary=read(a.run/'summary.json');rep_checks=0;pose_checks=0;visual={};keys_ref=None
    for condition in next(iter(corpus.values()))['conditions']:
        for family in ['representation_v1','phase_zero_v1']:
            for r in read(a.study/family/'registry.json'):
                directory=a.run/condition/family/r['run'];rows=read(directory/'metrics.json')
                assert len(rows)==len(corpus)*2
                assert {(x['scene'],x['condition']) for x in rows}=={(s,c) for s in corpus for c in ['normal','same_scene_different_position']}
                for x in rows:
                    s=corpus[x['scene']];z=np.load(directory/f"{x['scene']}__{x['condition']}.npz")
                    assert np.array_equal(z['target'],s['depth'].numpy()) and np.array_equal(z['mask'],s['mask'].numpy())
                    assert z['mean'].shape==z['target'].shape==(64,36)
                    donor=z['donor'];xy=s['poses'][:,:2].numpy();assert np.all(np.linalg.norm(xy-xy[donor],axis=1)>.5)
                    e=z['mean']-z['target'];mask=z['mask'];mae=float(np.abs(e)[mask].mean());rmse=float(np.sqrt(np.square(e)[mask].mean()))
                    assert abs(mae-x['mae'])<2e-5 and abs(rmse-x['rmse'])<2e-5
                    if 'probability' in z:
                        pr=z['probability'];assert pr.shape==(64,36,80) and np.all(pr>=0)
                        assert np.allclose(pr.sum(-1),1,atol=2e-6)
                        label=np.clip((z['target']*4).astype(int),0,79)
                        nll=-np.log(np.take_along_axis(pr,label[...,None],axis=-1)[...,0])[mask].mean()
                        assert abs(float(nll)-x['categorical_nll'])<2e-5
                    matching=[v for v in summary['representation_per_scene'] if v['training']==family and v['architecture']==r['architecture'] and v['head']==r['head'] and v['seed']==r['seed'] and v['input_condition']==condition and v['scene']==x['scene'] and v['condition']==x['condition']]
                    assert len(matching)==1 and matching[0]['mae']==x['mae'];rep_checks+=1
        tc=read(a.study/'transfer_v1/protocol.json')
        for seed in tc['seeds']:
            for arm in tc['arms']:
                rows=read(a.run/condition/'transfer_v1'/f'{arm}_{seed}'/'metrics.json')
                keys={(r['scene'],r['query'],r['condition']) for r in rows}
                if keys_ref is None:keys_ref=keys
                assert keys==keys_ref and len(rows)==len(corpus)*64*5
                for r in rows:
                    s=corpus[r['scene']];q=s['query_ids'].index(r['query']);gt=s['gt'][q].numpy();poses=s['candidates'][:,:3].numpy();i=r['predicted_index']
                    assert 0<=i<len(poses) and r['candidate_count']==len(poses)
                    xy=float(np.linalg.norm(poses[i,:2]-gt[:2]));yaw=float(abs(np.arctan2(np.sin(poses[i,2]-gt[2]),np.cos(poses[i,2]-gt[2]))))
                    assert abs(xy-r['position_error_m'])<2e-5 and abs(yaw*180/math.pi-r['yaw_error_deg'])<3e-5
                    assert r['r1']==float(xy<=1+1e-6) and r['fine']==float(xy<=.1+1e-6)
                    assert r['joint']==float(xy<=1+1e-6 and yaw<=math.pi/6+1e-6)
                    assert r['nll']>=0 and 0<=r['entropy']<=math.log(len(poses))+1e-5
                    if r['condition']=='visual':
                        key=(seed,r['scene'],r['query'])
                        if key in visual:assert r==visual[key]
                        else:visual[key]=r
                    pose_checks+=1
                for scene in corpus:
                    for view in ['visual','acoustic','fixed_fusion','shuffle_acoustic','shuffle_fusion']:
                        group=[r for r in rows if r['scene']==scene and r['condition']==view]
                        matching=[r for r in summary['transfer_per_scene'] if r['arm']==arm and r['seed']==seed and r['scene']==scene and r['input_condition']==condition and r['condition']==view]
                        assert len(group)==64 and len(matching)==1
                        for metric in ['r1','joint','fine','nll','entropy']:assert abs(np.mean([r[metric] for r in group])-matching[0][metric])<1e-10
    result=dict(code_sha=a.code_sha,part=cp['part'],scenes=len(corpus),frozen_checkpoints=len(freeze['checkpoints']),representation_scene_condition_checks=rep_checks,query_pose_metric_checks=pose_checks,identical_visual_predictions_across_arms_and_input_conditions=True,summary_rows_match_saved_metrics=True,cache_hashes_match=True,freeze_sha256=sha(a.freeze),evaluation_summary_sha256=sha(a.run/'summary.json'),limits='Does not independently regenerate every score field/NLL or prove acoustic identifiability; separate raw inference replay covers two development probes')
    (a.out/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()

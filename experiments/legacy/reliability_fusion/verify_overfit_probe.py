"""Recompute every Train probe MAP/metric and inspect frozen MLP activations."""
from pathlib import Path
import numpy as np
import torch
from acoustic_branch.contracts import atomic_json,sha
from acoustic_branch.data import read
from .model import Gate
from .overfit_probe import REL
from .revision_data import REL as SOURCE


def calculate(poses,truth,fixed):
    d=np.linalg.norm(poses[:,:2]-truth[:,:2],axis=1);a=abs((poses[:,2]-truth[:,2]+180)%360-180)
    fd=np.linalg.norm(fixed[:,:2]-truth[:,:2],axis=1);fa=abs((fixed[:,2]-truth[:,2]+180)%360-180)
    result={f'recall_{t:g}m':float(np.mean(d<=t)) for t in (.1,.5,1)}
    result.update({f'yaw_within_{t}deg':float(np.mean(a<=t)) for t in (10,30,50)})
    correct=(d<=1)&(a<=30);before=(fd<=1)&(fa<=30)
    result.update(recall_1m_30deg=float(correct.mean()),median_xy_m=float(np.median(d)),median_yaw_deg=float(np.median(a)),
        repair_vs_fixed_xy=int(((fd>1)&(d<=1)).sum()),harm_vs_fixed_xy=int(((fd<=1)&(d>1)).sum()),
        repair_vs_fixed_joint=int((~before&correct).sum()),harm_vs_fixed_joint=int((before&~correct).sum()))
    return result,~correct


def run(root,cache,report):
    root,cache,report=map(Path,(root,cache,report));out=root/REL;s=read(out/'summary_with_logit.json');selection=read(out/'selection.json')
    source=read(root/SOURCE/'scores_ready.json');entries={e['scene_id']:e for e in source['entries'] if e['split']=='train'}
    scenes={}
    for sid,e in entries.items():
        p=cache/e['cache_directory'];scenes[sid]={key:np.load(p/(key+'.npy'),mmap_mode='r') for key in ('scores','features','truth','xy')}
    items=selection['queries'];x=np.array([r['features'] for r in items],np.float32);truth=np.array([r['truth'] for r in items]);fixed=np.array([r['initial_pose'] for r in items])
    mean=np.array(selection['normalization_mean'],np.float32);std=np.array(selection['normalization_std'],np.float32)
    normalized=(x-mean)/std
    for i,q in enumerate(items):
        arr=scenes[q['scene_id']];np.testing.assert_array_equal(x[i],arr['features'][q['row']]);np.testing.assert_array_equal(truth[i],arr['truth'][q['row']])
    maps_checked=0;metric_checks=0;run_audits=[]
    for r in s['results']:
        n=r['n'];where=out/'controls/logit_v1/runs'/f'n{n}_seed{r["seed"]}' if r['arm']=='interior_logit_mse' else out/'runs'/f'n{n}_{r["arm"]}_seed{r["seed"]}'
        for phase in ('initial','best','last'):
            row=r[phase];poses=[]
            for q,w in zip(items[:n],row['weights']):
                arr=scenes[q['scene_id']];v,a=arr['scores'][q['row']];idx=int(np.argmax((1-w)*v+w*a));poses.append(np.r_[arr['xy'][idx//36],idx%36*10])
            poses=np.array(poses);np.testing.assert_allclose(poses,row['poses'],atol=1e-7,rtol=0);maps_checked+=n
            values,failed=calculate(poses,truth[:n],fixed[:n])
            for key,value in values.items():np.testing.assert_allclose(value,row[key],atol=1e-12,rtol=0);metric_checks+=1
            if phase=='initial':continue
            checkpoint=where/('best_train.pt' if phase=='best' else 'last.pt');state=torch.load(checkpoint,map_location='cpu',weights_only=False)
            assert state['train_only'] and state['step']==row['step']
            model=Gate(14).cuda().eval();model.load_state_dict(state['model'])
            with torch.no_grad():
                inputs=torch.tensor(normalized[:n],device='cuda');weights=model(inputs).cpu().numpy();hidden=model.net[:4](inputs).cpu().numpy()
            np.testing.assert_allclose(weights,row['weights'],atol=1e-7,rtol=0)
            zero=(hidden==0).all(1);failures=[]
            for i in np.flatnonzero(failed):
                q=items[i];w=row['weights'][i];lo,hi=q['target_interval']
                failures.append(dict(i=int(i),scene_id=q['scene_id'],sample_index=q['sample_index'],predicted_weight=w,good_grid=q['good_grid'],
                    inspected_success_interval=[lo,hi],interval_width=hi-lo,nearest_success_grid_distance=float(min(abs(w-v) for v in q['good_grid'])),
                    hidden_all_zero=bool(zero[i])))
            run_audits.append(dict(n=n,arm=r['arm'],seed=r['seed'],phase=phase,zero_hidden_queries=int(zero.sum()),
                zero_hidden_failed_queries=int((zero&failed).sum()),failures=failures,checkpoint_sha256=sha(checkpoint)))
    widths=np.array([q['interval_width'] for q in items]);audit=dict(status='passed',metric_checks=metric_checks,dense_MAP_recomputations=maps_checked,
        phases='initial/best/last for all24 runs',checkpoints_verified=48,source_features_and_GT_reproduced=128,val_test_arrays_accessed=False,
        interval_width_quantiles={str(t):float(np.quantile(widths,t)) for t in (0,.1,.5,.9,1)},normalized_feature_rank=int(np.linalg.matrix_rank(normalized-normalized.mean(0))),
        run_audits=run_audits,report_summary_sha256=sha(report/'summary.json'),base_summary_sha256=sha(out/'summary.json'),
        note='Post-hoc diagnostics only. Exact winner intervals can be subsets of total successful weights. Hidden collapse is a trained-representation observation, not proof of raw-feature information loss.')
    assert len(s['results'])==24 and s['val_test_arrays_accessed'] is False
    atomic_json(out/'verification.json',audit);atomic_json(report/'verification.json',audit)
    print({k:v for k,v in audit.items() if k!='run_audits'})
    for r in run_audits:
        if r['phase']=='last' and r['n']==128:print({k:v for k,v in r.items() if k not in ('checkpoint_sha256','failures')},'failures',[(f['i'],f['hidden_all_zero']) for f in r['failures']])


if __name__=='__main__':
    import sys
    run(*sys.argv[1:])

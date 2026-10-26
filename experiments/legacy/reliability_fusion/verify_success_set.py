"""Independent reported-metric audit and frozen train fit diagnostic."""
import json
from pathlib import Path
import numpy as np
import torch
from acoustic_branch.contracts import atomic_json,sha
from .model import Gate
from .success_set import REL,load_scenes,predict,metrics,set_loss


def run(root,cache,report):
    root,cache,report=map(Path,(root,cache,report));out=root/REL
    read=lambda p:json.loads(p.read_text())
    config=read(out/'config.json');summary=read(report/'summary.json');rows=read(report/'records.json')
    # Recompute every public metric without invoking the reporting metrics code.
    truth=np.array([r['gt'] for r in rows]);fixed=np.array([r['methods']['fixed_01']['pose'] for r in rows])
    err=lambda p:(np.linalg.norm(p[:,:2]-truth[:,:2],axis=1),abs((p[:,2]-truth[:,2]+180)%360-180))
    fd,fa=err(fixed);checks=0
    for name,m in summary['methods'].items():
        d,a=err(np.array([r['methods'][name]['pose'] for r in rows]))
        actual={f'recall_{t:g}m':float(np.mean(d<=t)) for t in (.1,.5,1)}
        actual.update({f'yaw_within_{t}deg':float(np.mean(a<=t)) for t in (10,30,50)})
        actual.update(recall_1m_30deg=float(np.mean((d<=1)&(a<=30))),median_xy_m=float(np.median(d)),median_yaw_deg=float(np.median(a)),
            repair_vs_fixed_xy=int(np.sum((fd>1)&(d<=1))),harm_vs_fixed_xy=int(np.sum((fd<=1)&(d>1))),
            repair_vs_fixed_joint=int(np.sum(~((fd<=1)&(fa<=30))&((d<=1)&(a<=30)))),
            harm_vs_fixed_joint=int(np.sum(((fd<=1)&(fa<=30))&~((d<=1)&(a<=30)))))
        for key,value in actual.items():
            np.testing.assert_allclose(value,m[key],atol=1e-12,rtol=0);checks+=1
    # Old Last is independently persisted by the previous oracle audit.
    old=read(root/'ReliabilityFusion/pilots/ecn14_v1/review_v2/last_records.json')
    old={(r['scene_id'],r['sample_index']):r for r in old['adaptive_margin_joint']}
    for r in rows:
        np.testing.assert_allclose(r['methods']['margin_last']['pose'],old[(r['scene_id'],r['sample_index'])]['pose'],atol=1e-7,rtol=0)
    scenes=load_scenes(root,cache,config);diagnostics={}
    for split in ('train','val'):
        selected=[s for s in scenes if s['split']==split];truths=[];fixed_poses=[];last_poses=[];losses=[];eligible=0;weights=[]
        state=torch.load(out/'run/last.pt',map_location='cpu',weights_only=False);model=Gate(14).cuda().eval();model.load_state_dict(state['model'])
        for s in selected:
            with torch.no_grad():
                w=model(torch.tensor((s['arrays']['features']-state['mean'])/state['std'],device='cuda',dtype=torch.float32))
                loss,valid=set_loss(w,torch.tensor(s['good'],device='cuda'),torch.tensor(config['weight_grid'],device='cuda'))
                losses.extend(loss[valid].cpu().tolist());eligible+=int(valid.sum())
            last_poses.extend(predict(s,w.cpu().numpy()));fixed_poses.extend(predict(s,np.full(len(w),.1)));truths.extend(s['arrays']['truth']);weights.extend(w.cpu().tolist())
        diagnostics[split]=dict(queries=len(truths),success_set_eligible=eligible,empty_success_set=len(truths)-eligible,
            fixed=metrics(fixed_poses,truths),last=metrics(last_poses,truths,fixed_poses),mean_set_loss=float(np.mean(losses)),
            weight_mean=float(np.mean(weights)),weight_min=float(np.min(weights)),weight_max=float(np.max(weights)))
    result=dict(status='passed',metric_checks=checks,margin_last_poses_verified=len(rows),split_disjoint=True,
        test_records_sha256=sha(report/'records.json'),diagnostics=diagnostics,
        note='Post-hoc frozen-model diagnostic; no optimization or checkpoint selection')
    splitsets=[set(config['selection'][s]) for s in ('train','val','test')]
    assert all(not a&b for i,a in enumerate(splitsets) for b in splitsets[i+1:])
    atomic_json(out/'verification.json',result);atomic_json(report/'verification.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    import sys
    run(*sys.argv[1:])

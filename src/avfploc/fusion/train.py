"""Portable adaptive-gate training on explicit disjoint Train/Val bank manifests.

This is a reusable trainer, not a claim that new runs reproduce every historical
schedule. Historical controlled experiment scripts live under experiments/legacy.
"""
import json
from pathlib import Path
import random
import numpy as np
from avfploc.data.banks import load, aligned
from avfploc.evaluation.baselines import standardized
from .features import extract
from .fixed import decode
from .targets import target_statistics


def run(manifest, output, epochs=17, hidden=(128,64), schema="processed58", lr=1e-3, decay=.1, seed=42, device="cpu", temperature=1., objective="nll"):
    import torch
    from .model import Gate
    from .losses import initialize, objective as pose_objective
    manifest=Path(manifest).resolve();config=json.loads(manifest.read_text())
    if set(config)!={"train","val"} or not config['train'] or not config['val']:
        raise ValueError("Training manifest must contain only nonempty train/val lists")
    ids={s:[e['scene_id'] for e in config[s]] for s in config}
    if set(ids['train'])&set(ids['val']) or any(len(set(v))!=len(v) for v in ids.values()):
        raise ValueError("Scene overlap/duplicates")
    if epochs<1 or temperature<=0 or objective not in ('nll','margin_xy','margin_joint'):
        raise ValueError("Invalid training configuration")
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    rows={s:[] for s in config};names=None
    files={s:set() for s in config}
    for split,entries in config.items():
        for entry in entries:
            paths=[(manifest.parent/entry[k]).resolve() for k in ('vision','acoustic','truth')]
            files[split].update(paths)
            v,a=map(load,paths[:2]);aligned(v,a)
            with np.load(paths[2],allow_pickle=False) as t:
                truth=t['poses'];qids=t['query_ids']
            if not np.array_equal(qids,v['query_ids']) or truth.shape!=(len(qids),3) or not np.isfinite(truth).all():
                raise ValueError("Truth/query alignment mismatch")
            for i,gt in enumerate(truth):
                vs,ac=standardized(v['scores'][i]),standardized(a['scores'][i])
                x,nn=extract(vs,ac,v['xy'],v['depth'][i],schema,v.get('ray_bank'))
                if names is not None and names!=nn:raise ValueError("Feature schema changed")
                names=nn
                expectation,_=target_statistics(vs,ac,v['xy'],gt)
                distance=np.linalg.norm(v['xy']-gt[:2],axis=1)
                yaw=abs((np.arange(36)*10-gt[2]+180)%360-180)
                inside=np.broadcast_to(distance[:,None]<=1,vs.shape).copy()
                if objective=='margin_joint':inside &= yaw[None,:]<=30
                rows[split].append((x,vs,ac,expectation,v['xy'],gt,inside,entry['scene_id']))
    if files['train']&files['val']:raise ValueError("Train/Val files overlap")
    x=np.stack([r[0] for r in rows['train']]);mean=x.mean(0);std=x.std(0);std=np.where(std>1e-6,std,1.)
    model=Gate(len(names),hidden=tuple(hidden)).to(device);initialize(model,.1)
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=decay)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if (output/'last.pt').exists():raise FileExistsError("Use a new output directory; refusing overwrite")
    def tensors(r):
        x,v,a,e,*_=r
        return [torch.as_tensor(z,dtype=torch.float32,device=device) for z in (((x-mean)/std)[None],v.reshape(1,-1),a.reshape(1,-1),e[None])]
    def loss_for(r,w,v,a,e):
        inside=None if objective=='nll' else torch.as_tensor(r[6].reshape(1,-1),device=device)
        return pose_objective(w,v,a,e,temperature,inside=inside).mean()
    def evaluate(split):
        from avfploc.evaluation.metrics import compare
        pred=[];gt=[];fixed=[];weights=[];losses=[];scene={}
        model.eval()
        with torch.inference_mode():
            for r in rows[split]:
                x,v,a,e=tensors(r);w=model(x);weight=float(w.item())
                p=decode((1-weight)*r[1]+weight*r[2],r[4]);f=decode(.9*r[1]+.1*r[2],r[4])
                pred.append(p);gt.append(r[5]);fixed.append(f);weights.append(weight);losses.append(float(loss_for(r,w,v,a,e)))
                scene.setdefault(r[7],[]).append(len(pred)-1)
        metrics=compare(pred,gt,fixed)
        ps,ts,fs=map(np.asarray,(pred,gt,fixed))
        per_scene={s:compare(ps[ix],ts[ix],fs[ix]) for s,ix in scene.items()}
        metrics.update(loss=float(np.mean(losses)),weight_mean=float(np.mean(weights)),per_scene=per_scene,
                       macro_joint=float(np.mean([v['recall_1m_30deg'] for v in per_scene.values()])))
        return metrics
    def save(name,epoch):
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),mean=mean,std=std,feature_names=names,
            schema=schema,hidden=hidden,epoch=epoch,seed=seed,selection=ids,objective=objective,temperature=temperature,
            training_manifest=config,normalization='train_only',test_accessed=False),output/name)
    history=[];best=-1
    for epoch in range(epochs+1):
        if epoch:
            model.train();order=list(range(len(rows['train'])));random.shuffle(order)
            for i in order:
                r=rows['train'][i];x,v,a,e=tensors(r);optimizer.zero_grad(set_to_none=True)
                loss=loss_for(r,model(x),v,a,e)
                if not torch.isfinite(loss):raise ValueError("Nonfinite training loss")
                loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                if not torch.isfinite(grad):raise ValueError("Nonfinite gradient")
                optimizer.step()
        result={s:evaluate(s) for s in rows};history.append(dict(epoch=epoch,**result))
        if result['val']['macro_joint']>best:
            best=result['val']['macro_joint'];save('best.pt',epoch)
        save('last.pt',epoch)
        (output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
    return history[-1]

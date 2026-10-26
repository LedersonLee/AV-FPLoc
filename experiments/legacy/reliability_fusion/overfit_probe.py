"""Train-only memorization diagnostic: unchanged core14 and scalar MLP.

No Val/Test arrays are opened. GT stratification is deliberately diagnostic,
not a sampled estimate of deployment performance. No production model changes.
"""
import csv
import fcntl
from pathlib import Path
import random
import shutil
import time
import traceback

import numpy as np
import torch

from acoustic_branch.contracts import atomic_json,immutable_json,sha
from acoustic_branch.data import read
from .core import FEATURE_NAMES
from .model import Gate
from .revision_model import initialize
from .success_set import good_weights,metrics,set_loss
from .revision_data import REL as SOURCE

REL='ReliabilityFusion/pilots/core14_overfit_v1'
ARMS={'set_original':dict(loss='set',weight_decay=.001),
      'set_no_decay':dict(loss='set',weight_decay=0.),
      'interior_mse':dict(loss='interior_point',weight_decay=0.)}


def balanced_indices(good,initial_index,sizes,seed):
    """Nested, 50/50 initially joint-correct and repairable-joint-fail sets."""
    if any(n%2 for n in sizes):raise ValueError('Even diagnostic sizes required')
    rng=np.random.default_rng(seed);eligible=good.any(1)
    yes=rng.permutation(np.flatnonzero(eligible&good[:,initial_index]))
    no=rng.permutation(np.flatnonzero(eligible&~good[:,initial_index]))
    if min(len(yes),len(no))<max(sizes)//2:raise ValueError('Insufficient eligible queries')
    order=np.column_stack((yes[:max(sizes)//2],no[:max(sizes)//2])).ravel()
    return {n:order[:n] for n in sizes}


def winner_interval(v,a,weight):
    """Exact interval in [0,1] over which the selected affine line wins.

    Endpoint tie ownership is ignored. Its strict interior has the same MAP.
    This need not be the ENTIRE joint-success interval: adjacent correct poses
    could also win outside this particular candidate's interval.
    """
    v,a=np.asarray(v).ravel(),np.asarray(a).ravel();d=a-v
    winner=int(np.argmax((1-weight)*v+weight*a));slope=d[winner]-d;rhs=v-v[winner]
    pos,neg=slope>0,slope<0
    lo=max(0.,float(np.max(rhs[pos]/slope[pos])) if pos.any() else 0.)
    hi=min(1.,float(np.min(rhs[neg]/slope[neg])) if neg.any() else 1.)
    if hi<lo-1e-9 or not lo-1e-9<=weight<=hi+1e-9:raise ValueError('Invalid affine MAP interval')
    return winner,float(lo),float(hi)


def prepare(root,cache):
    root,cache=map(Path,(root,cache));out=root/REL;source=read(root/SOURCE/'config.json');ready=read(root/SOURCE/'scores_ready.json')
    config=dict(version='core14_overfit_v1',source_config_sha256=sha(root/SOURCE/'config.json'),source_ready_sha256=sha(root/SOURCE/'scores_ready.json'),
        feature_columns=source['feature_columns'],feature_names=[FEATURE_NAMES[i] for i in source['feature_columns']],
        split_sha256=source['split_sha256'],train_scenes=source['selection']['train'],weight_grid=source['weight_grid'],
        sizes=[8,32,128],selection_seed=2026,seeds=[42,43],arms=ARMS,steps=4000,evaluate_every=100,lr=.001,
        normalization='same mean/std fit over all 3105 original train features; no val/test features',
        optimizer='AdamW; full batch; clip norm1; no scheduler or val early stop',
        stop='4000 steps or joint Recall100% at three consecutive checks with arm objective<1e-6',
        initial_weight=.1,architecture='14->32->16->1 sigmoid;1025params',
        sampling='nested 50/50 joint-success at w=.1 and joint-repairable failure; empty grid targets excluded from this feasibility diagnostic only',
        target_control='midpoint of exact MAP-winner interval at nearest joint-success grid weight',
        best='TRAIN joint Recall; ties TRAIN R1 then smaller set loss; NOT a deployment checkpoint',
        notice='Memorization/optimization diagnostic ONLY. Val/Test untouched. No generalization claim. No new RIR/encoder updates.')
    if len(config['feature_names'])!=14 or ready['test_accessed']:raise ValueError('Source contract')
    groups=[set(source['selection'][s]) for s in ('train','val','test')]
    assert all(not a&b for i,a in enumerate(groups) for b in groups[i+1:])
    immutable_json(out/'config.json',config)
    allx=[];good=[];lookup=[];scenes={};verified=[]
    for e in ready['entries']:
        if e['split']!='train':continue  # Never open Val arrays, receipts or feature files.
        if e['scene_id'] not in config['train_scenes']:raise ValueError('Wrong split')
        p=cache/e['cache_directory'];receipt=read(p/'receipt.json')
        if sha(p/'receipt.json')!=e['receipt_sha256']:raise ValueError('Receipt changed')
        for name,digest in receipt['files'].items():
            if sha(p/name)!=digest:raise ValueError('Train source changed: '+name)
        arrays={k:np.load(p/(k+'.npy'),mmap_mode='r') for k in ('scores','features','truth','xy')}
        with np.load(p/'weight_curve.npz') as blob:poses=blob['poses'].copy()
        success=good_weights(poses,arrays['truth']);queries=read(p/'queries.json')
        scenes[e['scene_id']]=arrays;allx.extend(arrays['features']);good.extend(success)
        lookup.extend(dict(scene_id=e['scene_id'],row=i,sample_index=q['sample_index']) for i,q in enumerate(queries))
        verified.append(dict(scene_id=e['scene_id'],receipt_sha256=e['receipt_sha256']))
    if len(allx)!=3105 or len(scenes)!=6:raise ValueError('Unexpected train scope')
    allx=np.asarray(allx);good=np.asarray(good);mean=allx.mean(0);std=allx.std(0);std=np.where(std>1e-6,std,1.)
    grid=np.asarray(config['weight_grid']);start=int(np.argmin(abs(grid-.1)))
    selected=balanced_indices(good,start,config['sizes'],config['selection_seed'])[max(config['sizes'])]
    items=[];scores=[];xs=[];ys=[];xy=[];targets=[];good_sel=[];initial=[]
    for global_id in selected:
        item=dict(lookup[global_id]);s=scenes[item['scene_id']];i=item['row'];v,a=np.asarray(s['scores'][i]);truth=s['truth'][i]
        g=good[global_id];target_grid=float(grid[np.flatnonzero(g)[np.argmin(abs(grid[g]-.1))]])
        winner,lo,hi=winner_interval(v,a,target_grid);target=(lo+hi)/2
        if hi-lo<=1e-10:raise ValueError('Degenerate target interval: diagnose explicitly, do not silently skip')
        idx=int(np.argmax((1-target)*v+target*a));pose=np.r_[s['xy'][idx//36],idx%36*10]
        if np.linalg.norm(pose[:2]-truth[:2])>1 or abs((pose[2]-truth[2]+180)%360-180)>30:raise ValueError('Interior control is not joint-successful')
        fixed_id=int(np.argmax(.9*v+.1*a));fixed=np.r_[s['xy'][fixed_id//36],fixed_id%36*10]
        fixed_joint=np.linalg.norm(fixed[:2]-truth[:2])<=1 and abs((fixed[2]-truth[2]+180)%360-180)<=30
        if bool(fixed_joint)!=bool(g[start]):raise ValueError('Grid/source initial prediction mismatch')
        item.update(global_train_row=int(global_id),truth=truth.tolist(),initial_pose=fixed.tolist(),initial_joint=bool(fixed_joint),
            good_grid=grid[g].tolist(),nearest_good_grid=target_grid,target_interval=[lo,hi],interior_target=target,
            interval_width=hi-lo,features=s['features'][i].tolist())
        items.append(item);scores.append(np.stack((v.ravel(),a.ravel())));xs.append(s['features'][i]);ys.append(truth);xy.append(s['xy']);targets.append(target);good_sel.append(g);initial.append(fixed)
    immutable_json(out/'selection.json',dict(queries=items,source_train_receipts=verified,normalization_mean=mean.tolist(),normalization_std=std.tolist(),val_test_arrays_accessed=False))
    return config,dict(items=items,scores=scores,features=np.asarray(xs),truth=np.asarray(ys),xy=xy,targets=np.array(targets),good=np.asarray(good_sel),fixed=np.asarray(initial),mean=mean,std=std)


def evaluate_weights(data,weights,n,bank):
    weights=np.asarray(weights);w=torch.tensor(weights,device='cuda',dtype=torch.float64)
    with torch.no_grad():ids=((1-w[:,None])*bank[:n,0]+w[:,None]*bank[:n,1]).argmax(1).cpu().numpy()
    poses=np.array([np.r_[data['xy'][i][idx//36],idx%36*10] for i,idx in enumerate(ids)])
    result=metrics(poses,data['truth'][:n],data['fixed'][:n]);result.update(weights=weights.tolist(),poses=poses.tolist())
    return result


def feature_audit(data):
    x=(data['features']-data['mean'])/data['std'];good=data['good'];pairs=[];duplicates=[]
    for i in range(len(x)):
        for j in range(i):
            distance=float(np.linalg.norm(x[i]-x[j]));overlap=bool((good[i]&good[j]).any())
            p=dict(i=i,j=j,normalized_l2=distance,grid_success_overlap=overlap,
                interval_overlap=bool(max(data['items'][i]['target_interval'][0],data['items'][j]['target_interval'][0])<min(data['items'][i]['target_interval'][1],data['items'][j]['target_interval'][1])))
            pairs.append(p)
            if np.array_equal(data['features'][i],data['features'][j]):duplicates.append(p)
    return dict(exact_duplicate_pairs=duplicates,nearest_pairs=sorted(pairs,key=lambda p:p['normalized_l2'])[:30],
        note='No exact collisions is not proof of sufficient reliability information. Disjoint grid targets do not exclude continuous solutions between grid weights.')


def fit(data,config,n,arm,seed,bank,folder,status,tracker):
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'result.json').exists():return read(folder/'result.json')
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=Gate(14).cuda();initialize(model,.1);spec=config['arms'][arm]
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=spec['weight_decay'])
    x=torch.tensor((data['features'][:n]-data['mean'])/data['std'],dtype=torch.float32,device='cuda')
    good=torch.tensor(data['good'][:n],device='cuda');grid=torch.tensor(config['weight_grid'],dtype=torch.float32,device='cuda');target=torch.tensor(data['targets'][:n],dtype=torch.float32,device='cuda')
    initial_parameters=torch.cat([p.detach().flatten().cpu() for p in model.parameters()]);history=[];best=None;stable=0;first_grad=None;started=time.monotonic()
    def check(step):
        nonlocal best,stable
        with torch.no_grad():
            w=model(x);loss,_=set_loss(w,good,grid);point=(w-target).square()
            result=evaluate_weights(data,w.cpu().numpy(),n,bank)
            logit_point=(model.net(x).squeeze(-1)-torch.logit(target)).square()
            objective=loss if spec['loss']=='set' else logit_point if spec['loss']=='interior_logit' else point
            result.update(step=step,set_loss=float(loss.mean()),max_set_loss=float(loss.max()),interior_mse=float(point.mean()),logit_mse=float(logit_point.mean()),
                objective_loss=float(objective.mean()),weight_mean=float(w.mean()),weight_std=float(w.std(unbiased=False)))
        history.append(result)
        key=lambda r:(r['recall_1m_30deg'],r['recall_1m'],-r['set_loss'])
        if best is None or key(result)>key(best):
            best=result
            torch.save(dict(model=model.state_dict(),step=step,train_only=True,feature_names=config['feature_names'],mean=data['mean'],std=data['std']),folder/'best_train.pt')
        stable=stable+1 if result['recall_1m_30deg']==1 and result['objective_loss']<1e-6 else 0
        scalars={k:v for k,v in result.items() if isinstance(v,(int,float))}
        if tracker:tracker.log({f'{n}/{arm}/seed{seed}/{k}':v for k,v in scalars.items()})
        if step%500==0:status('overfit',n=n,arm=arm,seed=seed,step=step,joint=result['recall_1m_30deg'],set_loss=result['set_loss'])
        return result
    last=check(0)
    for step in range(1,config['steps']+1):
        w=model(x);loss,_=set_loss(w,good,grid)
        objective=loss.mean() if spec['loss']=='set' else (model.net(x).squeeze(-1)-torch.logit(target)).square().mean() if spec['loss']=='interior_logit' else (w-target).square().mean()
        optimizer.zero_grad(set_to_none=True);objective.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm):raise ValueError('Nonfinite gradient')
        if first_grad is None:first_grad=float(norm)
        optimizer.step()
        if step%config['evaluate_every']==0 or step==config['steps']:
            last=check(step)
            atomic_json(folder/'progress.json',dict(step=step,last=last,best=best))
            if stable>=3:break
    final_parameters=torch.cat([p.detach().flatten().cpu() for p in model.parameters()])
    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step,mean=data['mean'],std=data['std'],train_only=True),folder/'last.pt')
    result=dict(n=n,arm=arm,seed=seed,steps=step,stop='overfit_target' if stable>=3 else 'step_limit',elapsed_seconds=time.monotonic()-started,
        initial=history[0],best=best,last=last,history=history,first_gradient_norm=first_grad,parameter_l2_change=float(torch.linalg.vector_norm(final_parameters-initial_parameters)),
        checkpoint_notice='Best chosen on these SAME Train queries; not a generalization estimate')
    atomic_json(folder/'result.json',result);return result


def run(root,cache,report):
    root,cache,report=map(Path,(root,cache,report));out=root/REL;out.mkdir(parents=True,exist_ok=True);report.mkdir(parents=True,exist_ok=True)
    lock=(out/'pipeline.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def status(stage,**extra):
        value=dict(status='running',stage=stage,updated_unix=time.time(),val_test_arrays_accessed=False,**extra)
        atomic_json(out/'status.json',value);atomic_json(report/'status.json',value)
    tracker=None
    try:
        shutil.copy2(Path(__file__).with_name('overfit_report.html'),report/'index.html');torch.set_num_threads(2);status('verify_train_only')
        config,data=prepare(root,cache);n=max(config['sizes']);width=max(s.shape[1] for s in data['scores'])
        bank=torch.full((n,2,width),-1e100,dtype=torch.float64,device='cuda')
        for i,s in enumerate(data['scores']):bank[i,:,:s.shape[1]]=torch.tensor(s,device='cuda')
        audit=feature_audit(data);atomic_json(out/'feature_audit.json',audit);atomic_json(report/'selection.json',read(out/'selection.json'))
        # Same supervised scalar control, with a real dense MAP recomputation.
        good=torch.tensor(data['good'],device='cuda');grid=torch.tensor(config['weight_grid'],dtype=torch.float64,device='cuda')
        w=torch.nn.Parameter(torch.full((n,),.1,dtype=torch.float64,device='cuda'));opt=torch.optim.SGD([w],lr=.5)
        for _ in range(8):opt.zero_grad();set_loss(w,good,grid)[0].sum().backward();opt.step()
        scalar=evaluate_weights(data,w.detach().cpu().numpy(),n,bank)
        if scalar['recall_1m_30deg']!=1:raise ValueError('Independent scalar control failed')
        try:
            import wandb
            tracker=wandb.init(project='hssd-reliability-fusion',name='core14-train-overfit-v1',dir=str(out),config=config)
            atomic_json(out/'wandb.json',dict(id=tracker.id,url=tracker.url))
        except ImportError:pass
        results=[]
        for size in config['sizes']:
            for seed in config['seeds']:
                for arm in config['arms']:
                    result=fit(data,config,size,arm,seed,bank,out/'runs'/f'n{size}_{arm}_seed{seed}',status,tracker)
                    results.append(result)
                    status('run_complete',n=size,seed=seed,arm=arm,best_joint=result['best']['recall_1m_30deg'],last_joint=result['last']['recall_1m_30deg'])
                    atomic_json(report/'partial.json',dict(config=config,results=results))
        summary=dict(status='complete',config=config,results=results,independent_scalar=scalar,feature_audit=audit,val_test_arrays_accessed=False)
        atomic_json(out/'summary.json',summary);atomic_json(report/'summary.json',summary)
        fields=['recall_0.1m','recall_0.5m','recall_1m','recall_1m_30deg','yaw_within_10deg','yaw_within_30deg','yaw_within_50deg','median_xy_m','median_yaw_deg',
            'repair_vs_fixed_xy','harm_vs_fixed_xy','repair_vs_fixed_joint','harm_vs_fixed_joint','set_loss','interior_mse']
        with (report/'metrics.csv').open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['n','arm','seed','checkpoint','step',*fields])
            for r in results:
                for stage in ('initial','best','last'):writer.writerow([r['n'],r['arm'],r['seed'],stage,r[stage]['step'],*[r[stage][k] for k in fields]])
        atomic_json(out/'status.json',dict(status='complete',updated_unix=time.time(),val_test_arrays_accessed=False));atomic_json(report/'status.json',read(out/'status.json'))
    except Exception:
        value=dict(status='failed',error=traceback.format_exc(),updated_unix=time.time());atomic_json(out/'status.json',value);atomic_json(report/'status.json',value);raise
    finally:
        if tracker:tracker.finish()


if __name__=='__main__':
    import sys
    run(*sys.argv[1:])

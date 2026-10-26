"""Train/val-only diagnostic of features, scalar-gate capacity and supervision.

No production model is trained/replaced. Oracles are explicitly GT-dependent
diagnostic ceilings, not deployable results. Fixed tiny probes fit TRAIN only;
held-out val and leave-one-train-scene-out checks are never used for tuning.
"""
import argparse
import html
from pathlib import Path
import time

import numpy as np
from scipy.spatial.distance import cdist
import torch

from acoustic_branch.contracts import atomic_json, immutable_json, sha
from acoustic_branch.data import read
from .core import FEATURE_NAMES, features
from .prepare import REL

WEIGHTS = np.linspace(0,1,21)
TEMPERATURES = (.25,.5,1.,2.,4.)
WIDTHS = ((.2,10.),(.5,20.),(1.,30.))


def prefer_half(values, maximum=True):
    """Stable tie rule: prefer an equally good weight nearest 0.5."""
    values=np.asarray(values)
    order=np.argsort(abs(WEIGHTS-.5),kind='stable')
    winner=(np.argmax if maximum else np.argmin)(values[...,order],axis=-1)
    return order[winner]


def metrics(rows, indices):
    n=np.arange(len(rows['scene']))
    return dict(recall_1m=float(rows['ok'][n,indices].mean()),
        recall_1m_30deg=float(rows['joint'][n,indices].mean()),
        same_xy_visual_yaw_joint=float(rows['visual_yaw_joint'][n,indices].mean()),
        mean_xy_m=float(rows['distance'][n,indices].mean()),
        mean_weight=float(WEIGHTS[indices].mean()))


def subset(rows, mask):
    return {k:v[mask] for k,v in rows.items()}


def probe(train, target, family, kind):
    columns=np.asarray(family)
    x=train['features'][:,columns].astype(float);z=target['features'][:,columns].astype(float)
    mean=x.mean(0);std=np.maximum(x.std(0),1e-6)
    x=(x-mean)/std;z=(z-mean)/std
    y=train['ok'].astype(float)
    # Equal total weight per training scene.
    counts={s:int((train['scene']==s).sum()) for s in np.unique(train['scene'])}
    mass=np.array([1/counts[s] for s in train['scene']]);mass/=mass.mean()
    if kind=='ridge':
        x=np.column_stack([np.ones(len(x)),x]);z=np.column_stack([np.ones(len(z)),z])
        regularizer=np.eye(x.shape[1])*.01;regularizer[0,0]=0
        coef=np.linalg.solve(x.T@(mass[:,None]*x)/len(x)+regularizer,x.T@(mass[:,None]*y)/len(x))
        prediction=z@coef
    elif kind=='knn':
        distance=cdist(z,x,metric='sqeuclidean')
        ids=np.argsort(distance,axis=1,kind='stable')[:,:min(64,len(x))]
        proximity=1/(np.take_along_axis(distance,ids,axis=1)+1e-3)
        proximity*=mass[ids];proximity/=proximity.sum(1,keepdims=True)
        prediction=np.einsum('nk,nkw->nw',proximity,y[ids])
    else:raise ValueError(kind)
    return prefer_half(prediction)


def families():
    return dict(all17=list(range(17)),vision=[0,1,2,3,4,10,11,12],
        acoustic=[5,6,7,8,9],relation=[13,14,15,16])


def diagnose(data):
    train=subset(data,data['split']=='train');val=subset(data,data['split']=='val')
    results={}
    for split,rows in [('train',train),('val',val)]:
        n=len(rows['scene']);half=np.full(n,10)
        result=dict(queries=n,scenes=len(np.unique(rows['scene'])),
            equal=metrics(rows,half),vision=metrics(rows,np.zeros(n,int)),acoustic=metrics(rows,np.full(n,20)),
            oracle_recall_grid=metrics(rows,prefer_half(rows['ok'])),
            oracle_joint_grid=metrics(rows,prefer_half(rows['joint'])),
            oracle_ce_grid=metrics(rows,prefer_half(rows['nll'][:,2,0],maximum=False)),
            oracle_xy_marginal_ce=metrics(rows,prefer_half(rows['xy_nll'],maximum=False)),
            oracle_region_mass_ce=metrics(rows,prefer_half(rows['region_nll'],maximum=False)),
            oracle_peak_margin=metrics(rows,prefer_half(rows['margin'],maximum=False)),
            constant_curve=[dict(weight=float(w),nll=float(rows['nll'][:,2,0,i].mean()),**metrics(rows,np.full(n,i))) for i,w in enumerate(WEIGHTS)],
            sensitivity={})
        for t,temperature in enumerate(TEMPERATURES):
            for k,width in enumerate(WIDTHS):
                result['sensitivity'][f'T={temperature:g},sigma={width[0]:g}m/{width[1]:g}deg']=metrics(rows,prefer_half(rows['nll'][:,t,k],maximum=False))
        viable=rows['ok'].sum(1)
        result['oracle_identifiability']=dict(no_successful_weights=int((viable==0).sum()),
            one_successful_grid_weight=int((viable==1).sum()),multiple_successful_weights=int((viable>1).sum()),
            all_weights_successful=int((viable==21).sum()),
            median_successful_weights=float(np.median(viable)),
            ce_minimizer_fails_when_another_weight_succeeds=int(((viable>0)&~rows['ok'][np.arange(n),prefer_half(rows['nll'][:,2,0],False)]).sum()))
        results[split]=result
    # Train-only selection; no val or test tuning.
    train_scene=np.unique(train['scene'])
    rcurve=np.mean([train['ok'][train['scene']==s].mean(0) for s in train_scene],axis=0)
    ncurve=np.mean([train['nll'][train['scene']==s,2,0].mean(0) for s in train_scene],axis=0)
    ri=int(prefer_half(rcurve));ni=int(prefer_half(ncurve,False))
    results['val']['constant_train_recall_selected']=metrics(val,np.full(len(val['scene']),ri))
    results['val']['constant_train_nll_selected']=metrics(val,np.full(len(val['scene']),ni))
    probes={}
    for name,cols in families().items():
        for kind in ('ridge','knn'):
            indices=probe(train,val,cols,kind)
            entry=dict(val=metrics(val,indices),definition='fixed '+kind+' utility-curve probe; not the production gate')
            if name=='all17':
                out=np.zeros(len(train['scene']),int)
                for s in train_scene:
                    held=train['scene']==s
                    out[held]=probe(subset(train,~held),subset(train,held),cols,kind)
                entry['leave_one_train_scene_out']=metrics(train,out)
                rng=np.random.default_rng(42);negative=[]
                for _ in range(100):
                    shuffled=indices.copy()
                    for s in np.unique(val['scene']):
                        ids=np.flatnonzero(val['scene']==s);shuffled[ids]=indices[rng.permutation(ids)]
                    negative.append(metrics(val,shuffled)['recall_1m'])
                entry['within_val_scene_shuffled_weight_control']=dict(mean_recall_1m=float(np.mean(negative)),
                    percentile_2_5_97_5=np.quantile(negative,[.025,.975]).tolist(),draws=100,
                    note='Permutation diagnostic, not a confidence interval for unseen-scene generalization')
            probes[name+'_'+kind]=entry
    return dict(splits=results,feature_probes=probes)


def publish(report, summary):
    report.mkdir(parents=True,exist_ok=True)
    rows=[]
    val=summary['diagnosis']['splits']['val']
    names=['vision','acoustic','equal','constant_train_recall_selected','constant_train_nll_selected',
           'oracle_ce_grid','oracle_recall_grid','oracle_joint_grid','oracle_xy_marginal_ce','oracle_region_mass_ce']
    for name in names:
        m=val[name]
        values=[name,f'{100*m["recall_1m"]:.2f}',f'{100*m["recall_1m_30deg"]:.2f}',
            f'{100*m["same_xy_visual_yaw_joint"]:.2f}',f'{m["mean_weight"]:.3f}']
        rows.append('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in values)+'</tr>')
    prows=[]
    for name,p in summary['diagnosis']['feature_probes'].items():
        m=p['val'];prows.append('<tr>'+''.join('<td>'+str(v)+'</td>' for v in
            (name,f'{100*m["recall_1m"]:.2f}',f'{100*m["recall_1m_30deg"]:.2f}',f'{m["mean_weight"]:.3f}'))+'</tr>')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for split in ('train','val'):
        curve=summary['diagnosis']['splits'][split]['constant_curve']
        axes[0].plot(WEIGHTS,[100*c['recall_1m'] for c in curve],label=split)
        axes[1].plot(WEIGHTS,[c['nll'] for c in curve],label=split)
    for ax,title in zip(axes,('R@1m (micro, %; higher better)','Soft-pose NLL (micro; lower better)')):
        ax.set_title(title);ax.set_xlabel('Acoustic weight');ax.grid(alpha=.2);ax.legend()
    fig.tight_layout();fig.savefig(report/'weight_curves.svg');plt.close(fig)
    page='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Reliability fusion design diagnosis</title><style>body{font:16px system-ui;max-width:1400px;margin:30px;color:#19314a}
table{border-collapse:collapse}td,th{padding:10px;border:1px solid #cdd7e5}th{background:#edf4fa}.note{background:#fff0ce;padding:18px;line-height:1.7}</style>
<h1>Reliability-aware fusion · input/supervision diagnosis</h1><div class="note">TRAIN/VAL ONLY; test is not loaded.
Same original energy+coherence frozen scores and 17 features. No production gate is retrained/replaced.
GT-dependent per-query ORACLES are diagnostic ceilings, NOT deployable validation performance.
Fixed ridge/kNN utility-curve probes fit train scenes only; no hyperparameter/feature tuning on val.</div>
<img src="weight_curves.svg" style="max-width:100%" alt="Localization recall and NLL prefer different weights">
<h2>Held-out val · 3 scenes / 1,704 queries · micro %</h2>
<table><tr><th>Diagnostic policy</th><th>R@1m</th><th>R@1m,30°</th><th>Same fused XY + vision yaw</th><th>Mean audio weight</th></tr>ROWS</table>
<h2>Input feature probes (not final model)</h2><table><tr><th>Features / fixed algorithm</th><th>Val R@1m</th><th>Val R@1m,30°</th><th>Mean weight</th></tr>PROWS</table>
<p>Oracle weights are searched on 0:0.05:1. Common temperature changes no MAP ranking at a fixed weight,
but can change the CE-minimizing weight. Gaussian widths and temperature sensitivity are in JSON.
The same-XY visual-yaw control isolates heading without changing selected XY.</p>
<p>Ridge λ=0.01; kNN k=64, inverse-distance weights. Scene-balanced train contribution. These simple probes
cannot establish that the 17 inputs are sufficient, nor prove that no better learner could use them.
LOSO and within-scene shuffled-weight controls are in JSON.</p><p><a href="summary.json">All curves, sensitivity and feature checks</a></p>'''
    (report/'index.html').write_text(page.replace('PROWS',''.join(prows)).replace('ROWS',''.join(rows)))
    atomic_json(report/'summary.json',summary)


@torch.inference_mode()
def run(root,out,report):
    root,out,report=map(Path,(root,out,report));out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    base=root/REL;ready=read(base/'scores_ready.json')
    audit=read(base/'diagnostics/best_last_v1/summary.json')
    if ready['test_accessed'] or not audit['score_cache_full_sha_verified']:raise ValueError('Need frozen train/val audit')
    contract=dict(ready_sha256=sha(base/'scores_ready.json'),audit_sha256=sha(base/'diagnostics/best_last_v1/summary.json'),
        weights=WEIGHTS.tolist(),temperatures=list(TEMPERATURES),widths=[list(w) for w in WIDTHS],
        test_accessed=False,no_production_retraining=True,feature_names=FEATURE_NAMES,
        fixed_probes=dict(ridge_lambda=.01,knn_k=64),code_sha256=sha(Path(__file__)))
    immutable_json(out/'contract.json',contract)
    collected=[];invariance=[]
    for entry in ready['entries']:
        if entry['split'] not in ('train','val'):raise ValueError('TEST forbidden')
        scene=entry['scene_id'];folder=root/entry['directory']
        if sha(folder/'receipt.json')!=entry['receipt_sha256']:raise ValueError('Cache receipt changed')
        receipt=read(folder/'receipt.json')
        for name in ('features.npy','truth.npy','xy.npy','target_expectations.npy'):
            if sha(folder/name)!=receipt['files'][name]:raise ValueError('Small input file changed')
        dest=out/'scenes'/(scene+'.npz');dest.parent.mkdir(exist_ok=True)
        if dest.exists():
            with np.load(dest) as blob:collected.append({k:blob[k].copy() for k in blob.files})
            continue
        x=np.load(folder/'features.npy');truth=np.load(folder/'truth.npy');xy=np.load(folder/'xy.npy')
        expected=np.load(folder/'target_expectations.npy');scores=np.load(folder/'scores.npy',mmap_mode='r')
        pos=torch.tensor(xy,dtype=torch.float64,device='cuda')
        weights=torch.tensor(WEIGHTS,dtype=torch.float64,device='cuda')[:,None,None]
        output={k:[] for k in ('distance','ok','joint','visual_yaw_joint','nll','xy_nll','region_nll','margin')}
        for i in range(len(x)):
            v,a=scores[i];gt=truth[i]
            if not np.isfinite(v).all() or not np.isfinite(a).all():raise ValueError('Nonfinite frozen scores')
            vt=torch.tensor(v,dtype=torch.float64,device='cuda');at=torch.tensor(a,dtype=torch.float64,device='cuda')
            s=(1-weights)*vt+weights*at
            winner=s.flatten(1).argmax(1).cpu().numpy();cell=winner//36;yaw=winner%36
            distance=np.linalg.norm(xy[cell]-gt[:2],axis=1);angle=abs((yaw*10-gt[2]+180)%360-180)
            vyaw=v.argmax(1)[cell];vang=abs((vyaw*10-gt[2]+180)%360-180)
            output['distance'].append(distance);output['ok'].append(distance<=1)
            output['joint'].append((distance<=1)&(angle<=30));output['visual_yaw_joint'].append((distance<=1)&(vang<=30))
            d=torch.linalg.vector_norm(pos-torch.tensor(gt[:2],device='cuda'),dim=1)
            angle=torch.tensor(abs((np.arange(36)*10-gt[2]+180)%360-180),dtype=torch.float64,device='cuda')
            expectations=[]
            for sigma,syaw in WIDTHS:
                q=torch.softmax((-.5*(d[:,None]/sigma)**2-.5*(angle[None]/syaw)**2).flatten(),dim=0).reshape(v.shape)
                expectations.append(torch.stack([(q*vt).sum(),(q*at).sum()]))
            ex=torch.stack(expectations)
            np.testing.assert_allclose(ex[0].cpu().numpy(),expected[i],atol=1e-8,rtol=1e-8)
            ew=(1-weights[:,0,0,None])*ex[:,0]+weights[:,0,0,None]*ex[:,1]
            nll=torch.stack([torch.logsumexp(s.flatten(1)/temp,dim=1)[:,None]-ew/temp for temp in TEMPERATURES],dim=1)
            output['nll'].append(nll.permute(1,2,0).cpu().numpy())
            z=torch.logsumexp(s.flatten(1),dim=1)
            xylog=torch.logsumexp(s,dim=2);qxy=torch.softmax(-.5*(d/.2)**2,dim=0)
            output['xy_nll'].append((z-(xylog*qxy).sum(1)).cpu().numpy())
            inside=d<=1
            if not inside.any():raise ValueError('No acceptable grid cell')
            output['region_nll'].append((z-torch.logsumexp(s[:,inside].flatten(1),dim=1)).cpu().numpy())
            outside=s[:,~inside].flatten(1).amax(1) if (~inside).any() else torch.full_like(z,-torch.inf)
            output['margin'].append((outside-s[:,inside].flatten(1).amax(1)).cpu().numpy())
            if i%100==0:
                atomic_json(out/'status.json',dict(status='running',scene=scene,split=entry['split'],query=i,total=len(x),updated_unix=time.time()))
        scores._mmap.close()
        row={k:np.asarray(value) for k,value in output.items()}
        row.update(features=x,scene=np.array([scene]*len(x)),split=np.array([entry['split']]*len(x)))
        np.savez_compressed(dest,**row);collected.append(row)
        print('design_probe',entry['split'],scene,len(x),flush=True)
    data={k:np.concatenate([r[k] for r in collected]) for k in collected[0]}
    # Demonstrate precisely which scale information the current summaries lose.
    rng=np.random.default_rng(42);v=rng.normal(size=(300,36));a=rng.normal(size=(300,36));xy=rng.normal(size=(300,2));depth=np.linspace(.5,8,11)
    before=features(v,a,xy,depth);after=features(100*v+37,.01*a-9,xy,depth)
    summary=dict(status='complete',contract=contract,diagnosis=diagnose(data),
        affine_feature_invariance=dict(max_absolute_difference=float(np.max(abs(before-after))),
            transform='vision*100+37; acoustic*0.01-9; same depth and geometry',
            meaning='All current features discard positive affine score scale. This can remove confidence-scale cues, not proof raw scales are calibrated.'),
        test_accessed=False,no_production_retraining=True,updated_unix=time.time())
    atomic_json(out/'summary.json',summary);publish(report,summary)
    atomic_json(out/'status.json',dict(status='complete',test_accessed=False,updated_unix=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','out','report'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.root,a.out,a.report)

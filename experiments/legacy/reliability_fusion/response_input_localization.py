"""Matched input ablation: Train12 fit / Val3 development; no Test access.

Same scalar Gate(32,16), two prior objectives, two seeds and 40 epochs. Only
feature columns change. Exact affine MAP envelopes accelerate evaluation;
boundary ownership uses canonical dense argmax, endpoints use verified grids.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import csv
import hashlib
from pathlib import Path
import random
import time

import numpy as np

from acoustic_branch.contracts import atomic_json,immutable_json,sha
from .input_supervision_audit import read,resolve,SOURCE,REL as AUDIT,pose,yawdiff,CORE_NAMES
from .weight_group_discovery import basin_features

REL='ReliabilityFusion/pilots/response_input_localization_v1'
ARMS={'core14':list(range(14)),'processed58':list(range(58)),
      'core14_response15':list(range(14))+list(range(58,73)),
      'processed58_response15':list(range(73))}


def metrics(p,t,fixed):
    p,t,fixed=map(np.asarray,(p,t,fixed));d=np.linalg.norm(p[:,:2]-t[:,:2],axis=1);a=yawdiff(p[:,2],t[:,2]);fd=np.linalg.norm(fixed[:,:2]-t[:,:2],axis=1);fa=yawdiff(fixed[:,2],t[:,2])
    result={f'recall_{x:g}m':float(np.mean(d<=x))for x in(.1,.5,1)}
    result.update({f'yaw_within_{x}deg':float(np.mean(a<=x))for x in(10,30,50)})
    result.update(queries=len(d),recall_1m_30deg=float(np.mean((d<=1)&(a<=30))),median_xy_m=float(np.median(d)),median_yaw_deg=float(np.median(a)))
    for name,before,after in [('xy',fd<=1,d<=1),('joint',(fd<=1)&(fa<=30),(d<=1)&(a<=30))]:
        result['repair_vs_fixed_'+name]=int(np.sum(~before&after));result['harm_vs_fixed_'+name]=int(np.sum(before&~after))
    return result


def load_scene(out,sid,root,ssd):
    dest=out/'scenes'/sid;receipt=read(dest/'receipt.json')
    assert sha(dest/'data.npz')==receipt['data_sha256']
    with np.load(dest/'data.npz')as b:data={k:b[k].copy()for k in b.files}
    data.update(scene_id=sid,split=receipt['split'],ref=receipt['source_ref'],source=resolve(receipt['source_ref'],ssd),queries=read(dest/'queries.json'))
    return data


def predict(s,w):
    """GT-free decoding of dense argmax; all metrics use original coordinates."""
    w=np.asarray(w,np.float64);assert np.all(np.isfinite(w))and np.all((w>=0)&(w<=1))
    j=np.argmax(w[:,None]<=s['upper'],axis=1);p=s['path_poses'][np.arange(len(w)),j].copy()
    p[w==0]=s['grid_poses'][w==0,0];p[w==1]=s['grid_poses'][w==1,-1]
    near=np.any(abs(w[:,None]-s['upper'])<1e-11,axis=1)&(w>0)&(w<1)
    if near.any():
        bank=np.load(s['source']/'scores.npy',mmap_mode='r')
        for i in np.flatnonzero(near):
            v,a=bank[i];p[i]=pose(int(np.argmax((1-w[i])*v+w[i]*a)),s['xy'])
    return p


def prepare_scene(task):
    root,ssd,out,ref=task;root,ssd,out=map(Path,(root,ssd,out));sid=ref['scene_id'];assert ref['split']in('train','val')
    dest=out/'scenes'/sid;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'receipt.json').exists():return read(dest/'receipt.json')
    source=resolve(ref,ssd);assert sha(source/'receipt.json')==ref['receipt_sha256']
    for name,digest in read(source/'receipt.json')['files'].items():assert sha(source/name)==digest
    audit=root/AUDIT/'scenes'/sid;rec=read(audit/'complete.json')
    assert rec['source_receipt_sha256']==ref['receipt_sha256']
    assert sha(audit/'arrays.npz')==rec['arrays_sha256']and sha(audit/'queries.json')==rec['queries_sha256']
    qs=read(audit/'queries.json');assert all(q['split']==ref['split']for q in qs)
    xy=np.load(source/'xy.npy');truth=np.load(source/'truth.npy')
    with np.load(audit/'arrays.npz')as b:old=np.c_[b['core'],b['extra']];np.testing.assert_array_equal(b['truth'],truth)
    target=ssd/SOURCE/'cache/targets'/ref['split']/(sid+'.npz');tr=read(target.with_suffix('.json'))
    assert sha(target)==tr['sha256']and tr['score_receipt_sha256']==ref['receipt_sha256']
    with np.load(target)as b:labels={k:b[k].copy()for k in b.files}
    new=[];maxparts=max(len(q['parts'])for q in qs);upper=np.full((len(qs),maxparts),2.);path_poses=np.zeros((len(qs),maxparts,3))
    for i,q in enumerate(qs):
        z,nn=basin_features(q['parts'],xy);new.append(z)
        for j,(lo,hi,idx)in enumerate(q['parts']):upper[i,j]=hi;path_poses[i,j]=pose(int(idx),xy)
        path_poses[i,len(q['parts']):]=path_poses[i,len(q['parts'])-1]
    x=np.c_[old,new];assert x.shape[1]==73 and np.isfinite(x).all()
    names=CORE_NAMES+rec['feature_names']+nn
    np.savez_compressed(dest/'data.npz',x=x,xy=xy,truth=truth,upper=upper,path_poses=path_poses,**labels)
    atomic_json(dest/'queries.json',[{k:q[k]for k in('sample_index','pose_id','image_path','parts','success')}for q in qs])
    scene=dict(source=source,xy=xy,upper=upper,path_poses=path_poses,grid_poses=labels['grid_poses'])
    for j,w in enumerate(np.linspace(0,1,21)):
        np.testing.assert_allclose(predict(scene,np.full(len(qs),w)),labels['grid_poses'][:,j],atol=1e-7,rtol=0)
    receipt=dict(scene_id=sid,split=ref['split'],source_ref=ref,queries=len(qs),features=names,eligible=int(labels['valid'].sum()),
        data_sha256=sha(dest/'data.npz'),queries_sha256=sha(dest/'queries.json'),target_sha256=tr['sha256'],grid_pose_reproductions=len(qs)*21,test_accessed=False)
    atomic_json(dest/'receipt.json',receipt);print('PREPARED',sid,ref['split'],len(qs),flush=True);return receipt


def prepare(root,ssd,workers):
    root,ssd=map(Path,(root,ssd));out=root/REL;out.mkdir(parents=True,exist_ok=True)
    base=read(root/SOURCE/'config.json');refs=[r for r in read(root/SOURCE/'scores_ready.json')['entries']if r['split']in('train','val')]
    assert len(refs)==15 and all(r['scene_id']in base['selection'][r['split']]for r in refs)
    assert not set(base['selection']['train'])&set(base['selection']['val'])
    config=dict(version='v1',source_config_sha256=sha(root/SOURCE/'config.json'),selection={s:base['selection'][s]for s in('train','val')},
        arms=ARMS,seeds=[42,43],objectives=['logit_regression','success_set'],epochs=40,lr=.001,weight_decay=0.,batch_size=16,
        initial_weight=.1,hidden=[32,16],clip_grad_norm=1.,scheduler=None,early_stop=False,scene_balanced=True,
        target='EXACT SAME frozen subset12 targets: successful grid, nearest-success interior midpoint in logit space',
        model_selection='Val scene-macro R1, then scene-macro Joint, then own loss; include epoch0',
        normalization='ALL Train queries only, separate per input arm, never Val',
        inputs='Predicted score response and GT-free prior features only; no class labels, success sets, GT errors, oracle w, Laplace scale, absolute coordinates, or scene IDs',
        test_accessed=False,no_encoder_update=True,counts=base['counts'],device_note='Small MLP CPU1 benchmark .305s/1000steps vs GPU .553s; independent runs parallel on local CPU')
    immutable_json(out/'config.json',config)
    with ProcessPoolExecutor(max_workers=workers)as pool:
        receipts=[f.result()for f in as_completed([pool.submit(prepare_scene,(str(root),str(ssd),str(out),ref))for ref in refs])]
    atomic_json(out/'ready.json',dict(scenes=receipts,source_code_sha256=sha(Path(__file__)),test_accessed=False))
    print('READY',out,flush=True)


def fit_task(task):
    import torch
    from .model import Gate
    from .revision_model import initialize
    root,ssd,arm,kind,seed=task[:5];options=task[5]if len(task)>5 else{}
    root,ssd=map(Path,(root,ssd));out=root/options.get('output_rel',REL);dataout=root/options.get('data_rel',REL);config=read(out/'config.json')
    hidden=tuple(options.get('hidden',config.get('hidden',[32,16])))
    folder=out/'runs'/options.get('run_name',f'{arm}_{kind}_seed{seed}');folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():return read(folder/'complete.json')
    torch.set_num_threads(1);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    refs=sorted(read(dataout/'ready.json')['scenes'],key=lambda r:r['scene_id']);scenes=[load_scene(dataout,r['scene_id'],root,ssd)for r in refs]
    train=[s for s in scenes if s['split']=='train'];val=[s for s in scenes if s['split']=='val'];cols=ARMS[arm]
    allx=np.concatenate([s['x'][:,cols]for s in train]);mean=allx.mean(0);std=allx.std(0);std=np.where(std>1e-6,std,1.)
    for s in scenes:
        s['xt']=torch.tensor((s['x'][:,cols]-mean)/std,dtype=torch.float32)
        s['tt']=torch.tensor(np.log(s['interior_target'])-np.log1p(-s['interior_target']),dtype=torch.float32)
        s['gt']=torch.tensor(s['good']);s['validt']=torch.tensor(s['valid'])
    model=Gate(len(cols),hidden=hidden);initialize(model,.1);optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=0.)
    grid=torch.tensor(np.linspace(0,1,21),dtype=torch.float32)
    def loss(raw,s,ids):
        if kind=='logit_regression':return(raw-s['tt'][ids]).square()
        good=s['gt'][ids];d=(torch.sigmoid(raw)[:,None]-grid[None])**2
        return d.masked_fill(~good,torch.inf).min(1).values
    @torch.inference_mode()
    def evaluate(items,details=False):
        ps=[];ts=[];fs=[];ws=[];losses=[];per_scene={};by_scene={}
        model.eval()
        for s in items:
            raw=model.net(s['xt']).squeeze(-1);w=torch.sigmoid(raw).numpy();p=predict(s,w);fixed=s['grid_poses'][:,2]
            m=metrics(p,s['truth'],fixed);per_scene[s['scene_id']]=m;ps.extend(p);ts.extend(s['truth']);fs.extend(fixed);ws.extend(w.tolist())
            ids=np.flatnonzero(s['valid']);losses.extend(loss(raw[ids],s,ids).numpy().tolist())
            if details:by_scene[s['scene_id']]=dict(weights=w.tolist(),poses=p.tolist())
        return dict(micro=metrics(ps,ts,fs),per_scene=per_scene,macro_R1=float(np.mean([m['recall_1m']for m in per_scene.values()])),macro_joint=float(np.mean([m['recall_1m_30deg']for m in per_scene.values()])),loss=float(np.mean(losses)),weight_mean=float(np.mean(ws)),weight_p10=float(np.quantile(ws,.1)),weight_p90=float(np.quantile(ws,.9)),**(dict(predictions=by_scene)if details else{}))
    best=evaluate(val);initial=best;best_epoch=0;history=[];updates=0;batch_digest=hashlib.sha256()
    def checkpoint(path,epoch):
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),mean=mean,std=std,columns=cols,feature_names=[refs[0]['features'][j]for j in cols],epoch=epoch,arm=arm,kind=kind,seed=seed,config_sha256=sha(out/'config.json'),updates=updates,test_accessed=False,
            python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),hidden=hidden),path)
    checkpoint(folder/'best.pt',0)
    total=sum(s['valid'].sum()for s in train);started=time.perf_counter()
    for epoch in range(1,41):
        batches=[];model.train()
        for s in train:
            ids=np.flatnonzero(s['valid']);np.random.shuffle(ids)
            batches.extend((s,ids[i:i+16])for i in range(0,len(ids),16))
        random.shuffle(batches);summed=0.;count=0
        for s,ids in batches:
            batch_digest.update(s['scene_id'].encode()+b'\0'+np.asarray(ids,dtype='<i8').tobytes())
            raw=model.net(s['xt'][ids]).squeeze(-1);values=loss(raw,s,ids)
            objective=values.mean()*total/(len(train)*int(s['valid'].sum()))
            optimizer.zero_grad(set_to_none=True);objective.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(grad):raise ValueError('Nonfinite gradient')
            optimizer.step();updates+=1;summed+=float(values.detach().sum());count+=len(ids)
        last=evaluate(val);key=lambda r:(r['macro_R1'],r['macro_joint'],-r['loss'])
        if key(last)>key(best):best=last;best_epoch=epoch;checkpoint(folder/'best.pt',epoch)
        history.append(dict(epoch=epoch,train_loss=summed/count,val=last));atomic_json(folder/'status.json',dict(epoch=epoch,best_epoch=best_epoch,elapsed_s=time.perf_counter()-started))
    checkpoint(folder/'last.pt',40);atomic_json(folder/'history.json',history)
    evaluated={}
    for phase in('best','last'):
        state=torch.load(folder/(phase+'.pt'),map_location='cpu',weights_only=False);model.load_state_dict(state['model'])
        evaluated[phase]=dict(epoch=state['epoch'],val=evaluate(val,True),train=evaluate(train,True))
    value=dict(arm=arm,objective=kind,seed=seed,input_count=len(cols),initial_val=initial,history=history,evaluated=evaluated,elapsed_s=time.perf_counter()-started,
        best_sha256=sha(folder/'best.pt'),last_sha256=sha(folder/'last.pt'),test_accessed=False,config_sha256=sha(out/'config.json'),hidden=list(hidden),parameters=sum(p.numel()for p in model.parameters()),batch_order_sha256=batch_digest.hexdigest())
    atomic_json(folder/'complete.json',value);print('FIT DONE',arm,kind,seed,best_epoch,evaluated['best']['val']['micro']['recall_1m'],evaluated['last']['val']['micro']['recall_1m'],flush=True);return value


def train(root,ssd,workers):
    root,ssd=map(Path,(root,ssd));out=root/REL;config=read(out/'config.json');assert (out/'ready.json').exists()
    tasks=[(str(root),str(ssd),arm,kind,seed)for arm in ARMS for kind in config['objectives']for seed in config['seeds']]
    with ProcessPoolExecutor(max_workers=workers)as pool:
        for i,f in enumerate(as_completed([pool.submit(fit_task,t)for t in tasks]),1):
            r=f.result();atomic_json(out/'status.json',dict(stage='training',complete=i,total=len(tasks),last=r['arm']+'_'+r['objective'],test_accessed=False))
    atomic_json(out/'training_complete.json',dict(runs=len(tasks),test_accessed=False));print('TRAINING COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','train']);p.add_argument('--root',type=Path,required=True);p.add_argument('--ssd',type=Path,required=True);p.add_argument('--workers',type=int,default=6);args=p.parse_args()
    (prepare if args.mode=='prepare'else train)(args.root,args.ssd,args.workers)

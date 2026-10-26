"""Matched A/B/C supervision; fixed128/64,58 inputs,WD.1,17epochs.

Train mask, normalization, initializations and minibatch order are unchanged.
No hyperparameter search or checkpoint selection; Val first loaded after fits.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
from pathlib import Path
import random
import time

import numpy as np
from acoustic_branch.contracts import atomic_json,immutable_json,sha
from .input_supervision_audit import read,yawdiff,merge_intervals
from .regularization_probe import tensors,evaluate_model,REL as CONTROL
from .response_input_localization import REL as DATA,load_scene

REL='ReliabilityFusion/pilots/supervision_abc_v1'
ARMS={'A':'기존 점 목표','B':'기준값 보존 점 목표','C':'기준값 보존 연속 성공 집합'}


def fixed_good(s):
    p=s['grid_poses'][:,2];t=s['truth']
    return (np.linalg.norm(p[:,:2]-t[:,:2],axis=1)<=1)&(yawdiff(p[:,2],t[:,2])<=30)


def label_tensors(s):
    import torch
    success=[merge_intervals(q['success']['joint'])for q in s['queries']]
    count=max(1,max(map(len,success)));lo=np.zeros((len(success),count));hi=lo.copy();mask=np.zeros_like(lo,dtype=bool)
    for i,intervals in enumerate(success):
        for j,(a,b)in enumerate(intervals):
            # Tiny interior inset only for internal boundaries: avoid a zero
            # distance at an argmax tie whose owner can be a failing pose.
            eps=min(1e-7,(b-a)*1e-3)
            lower=a+eps if a>0 else 0.;upper=b-eps if b<1 else 1.
            lo[i,j]=np.log(lower)-np.log1p(-lower)if lower>0 else -np.inf
            hi[i,j]=np.log(upper)-np.log1p(-upper)if upper<1 else np.inf
            mask[i,j]=True
    good=fixed_good(s)
    assert np.all(~s['valid']|mask.any(1))
    assert np.all(~good|s['valid'])
    s.update(fixed_good=good,fg=torch.tensor(good),lo=torch.tensor(lo,dtype=torch.float32),
        hi=torch.tensor(hi,dtype=torch.float32),component_mask=torch.tensor(mask))


def objective(raw,s,ids,arm):
    import torch
    z0=raw.new_tensor(np.log(.1)-np.log1p(-.1));anchor=(raw-z0).square()
    if arm=='A':return(raw-s['tt'][ids]).square()
    if arm=='B':return torch.where(s['fg'][ids],anchor,(raw-s['tt'][ids]).square())
    if arm!='C':raise ValueError(arm)
    # Padding is masked AFTER squaring finite residuals. Squaring +inf before
    # masking would create inf*zero backward NaNs in unused components.
    delta=torch.maximum(s['lo'][ids]-raw[:,None],raw[:,None]-s['hi'][ids]).clamp_min(0)
    distance=delta.square().masked_fill(~s['component_mask'][ids],torch.inf).min(1).values
    return torch.where(s['fg'][ids],anchor,distance)


def prepare(root):
    out=root/REL;out.mkdir(parents=True,exist_ok=True);cfg=read(root/CONTROL/'config.json')
    selection=read(root/CONTROL/'selection.json')
    assert selection['winners']['early']=='wd01'and selection['early']['wd01']['epoch']==17
    config=dict(version='v1',hidden=[128,64],inputs=58,parameters=15873,seeds=[42,43],arms=ARMS,
        selection=cfg['selection'],epochs=17,decay=.1,lr=.001,batch_size=16,clip=1.,scheduler=None,
        initialization=.1,scene_balanced=True,normalization='ALL Train queries only; identical to frozen A control',
        eligible='EXACT SAME frozen valid mask; do NOT add96 continuous-feasible Train queries missed by old grid',
        A='Squared error to existing frozen logit(interior_target).',
        B='If fixed.1 is Joint correct, squared error to logit(.1); otherwise SAME A target.',
        C='If fixed.1 is Joint correct, SAME B anchor; otherwise squared logit distance to UNION of exact merged Joint-success intervals.',
        C_numerics='Internal interval edges inset min(1e-7,width*1e-3); endpoints0/1 map to +/-infinity. No margin or anchor coefficient search.',
        no_val_selection=True,no_checkpoint_selection=True,test_accessed=False,
        control_selection_sha256=sha(root/CONTROL/'selection.json'),data_ready_sha256=sha(root/DATA/'ready.json'),
        decision='Compare paired Recall and net Repair-Harm, plus recoverable-failure success and weight collapse. No follow-up grid added after Val.')
    immutable_json(out/'config.json',config);print('PREPARED ABC',flush=True)


def evaluate(model,scenes,arm,details=False):
    import torch
    result=evaluate_model(model,scenes,details)
    with torch.inference_mode():
        own=[]
        for s in scenes:
            ids=np.flatnonzero(s['valid']);z=model.net(s['xt'][ids]).squeeze(-1)
            own.extend(objective(z,s,ids,arm).numpy().tolist())
    result['common_A_loss']=result.pop('loss');result['own_loss']=float(np.mean(own))
    return result


def fit_task(task):
    import torch
    from .model import Gate
    from .revision_model import initialize
    root,ssd,arm,seed=task;root,ssd=map(Path,(root,ssd));out=root/REL;cfg=read(out/'config.json')
    folder=out/'runs'/f'{arm}_seed{seed}';folder.mkdir(parents=True,exist_ok=True)
    if(folder/'complete.json').exists():return read(folder/'complete.json')
    torch.set_num_threads(1);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    assert sha(root/DATA/'ready.json')==cfg['data_ready_sha256']
    refs=sorted([r for r in read(root/DATA/'ready.json')['scenes']if r['split']=='train'],key=lambda r:r['scene_id'])
    assert len(refs)==12 and set(r['scene_id']for r in refs)==set(cfg['selection']['train'])
    scenes=[load_scene(root/DATA,r['scene_id'],root,ssd)for r in refs]
    mean,std=tensors(scenes,[s['scene_id']for s in scenes])
    for s in scenes:label_tensors(s)
    model=Gate(58,hidden=(128,64));initialize(model,.1)
    init=hashlib.sha256(b''.join(v.numpy().tobytes()for v in model.state_dict().values())).hexdigest()
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['decay'])
    total=sum(int(s['valid'].sum())for s in scenes);history=[];digest=hashlib.sha256();started=time.perf_counter()
    initial=evaluate(model,scenes,arm)
    for epoch in range(1,18):
        batches=[];model.train();summed=0.;count=0
        for s in scenes:
            ids=np.flatnonzero(s['valid']);np.random.shuffle(ids)
            batches.extend((s,ids[i:i+16])for i in range(0,len(ids),16))
        random.shuffle(batches)
        for s,ids in batches:
            digest.update(s['scene_id'].encode()+b'\0'+np.asarray(ids,dtype='<i8').tobytes())
            z=model.net(s['xt'][ids]).squeeze(-1);values=objective(z,s,ids,arm)
            loss=values.mean()*total/(len(scenes)*int(s['valid'].sum()))
            optimizer.zero_grad(set_to_none=True);loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(grad):raise ValueError('Nonfinite gradient')
            optimizer.step();summed+=float(values.detach().sum());count+=len(ids)
        model.eval();history.append(dict(epoch=epoch,train_online_own_loss=summed/count,train=evaluate(model,scenes,arm)))
        atomic_json(folder/'status.json',dict(epoch=epoch,test_accessed=False))
    checkpoint=folder/'epoch017.pt'
    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),mean=mean,std=std,epoch=17,
        columns=list(range(58)),hidden=[128,64],seed=seed,arm=arm,config_sha256=sha(out/'config.json'),
        python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),test_accessed=False),checkpoint)
    result=dict(arm=arm,seed=seed,initial=initial,history=history,train=evaluate(model,scenes,arm,True),
        batch_order_sha256=digest.hexdigest(),initial_model_sha256=init,checkpoint_sha256=sha(checkpoint),
        supervised_queries=total,config_sha256=sha(out/'config.json'),elapsed_s=time.perf_counter()-started,test_accessed=False)
    if arm=='A':
        old=read(root/CONTROL/'evaluation.json')['methods'][f'wd01_seed{seed}_early']['train']
        assert result['train']['predictions']==old['predictions']and result['train']['micro']==old['micro']
        assert result['train']['common_A_loss']==old['loss']
    atomic_json(folder/'complete.json',result);print('FIT',arm,seed,'Joint',result['train']['micro']['recall_1m_30deg'],flush=True);return result


def train(root,ssd,workers):
    out=root/REL;cfg=read(out/'config.json');tasks=[(str(root),str(ssd),arm,seed)for arm in ARMS for seed in cfg['seeds']]
    with ProcessPoolExecutor(max_workers=workers)as pool:
        for f in as_completed([pool.submit(fit_task,t)for t in tasks]):f.result()
    for seed in cfg['seeds']:
        rr=[read(out/'runs'/f'{arm}_seed{seed}'/'complete.json')for arm in ARMS]
        assert len({r['batch_order_sha256']for r in rr})==1 and len({r['initial_model_sha256']for r in rr})==1
        assert {r['supervised_queries']for r in rr}=={5364}
    atomic_json(out/'training_complete.json',dict(runs=6,matched_batch_orders=True,matched_initializations=True,same_supervised_mask=True,test_accessed=False))


def evaluate_val(root,ssd):
    import torch
    from .model import Gate
    out=root/REL;cfg=read(out/'config.json');assert(out/'training_complete.json').exists()
    # Only now load Val. No early stopping, tuning or arm selection on Val.
    refs=sorted([r for r in read(root/DATA/'ready.json')['scenes']if r['split']=='val'],key=lambda r:r['scene_id'])
    assert len(refs)==3 and set(r['scene_id']for r in refs)==set(cfg['selection']['val'])
    scenes=[load_scene(root/DATA,r['scene_id'],root,ssd)for r in refs];torch.set_num_threads(1);result={}
    for arm in ARMS:
        for seed in cfg['seeds']:
            folder=out/'runs'/f'{arm}_seed{seed}';r=read(folder/'complete.json');assert sha(folder/'epoch017.pt')==r['checkpoint_sha256']
            state=torch.load(folder/'epoch017.pt',map_location='cpu',weights_only=False)
            assert state['config_sha256']==sha(out/'config.json')
            model=Gate(58,hidden=(128,64));model.load_state_dict(state['model']);model.eval()
            for s in scenes:
                s['xt']=torch.tensor((s['x'][:,list(range(58))]-state['mean'])/state['std'],dtype=torch.float32)
                s['tt']=torch.tensor(np.log(s['interior_target'])-np.log1p(-s['interior_target']),dtype=torch.float32);label_tensors(s)
            val=evaluate(model,scenes,arm,True)
            if arm=='A':
                old=read(root/CONTROL/'evaluation.json')['methods'][f'wd01_seed{seed}_early']['val']
                assert val['predictions']==old['predictions']and val['micro']==old['micro']and val['common_A_loss']==old['loss']
            result[f'{arm}_seed{seed}']=dict(arm=arm,seed=seed,train=r['train'],val=val,checkpoint_sha256=r['checkpoint_sha256'])
    atomic_json(out/'evaluation.json',dict(methods=result,A_reproduced_exactly=True,val_used_for_selection=False,test_accessed=False))
    print('EVALUATED: A matches frozen control exactly',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','train','evaluate_val']);p.add_argument('--root',type=Path,required=True);p.add_argument('--ssd',type=Path,required=True);p.add_argument('--workers',type=int,default=6);a=p.parse_args()
    if a.mode=='prepare':prepare(a.root)
    elif a.mode=='train':train(a.root,a.ssd,a.workers)
    else:evaluate_val(a.root,a.ssd)

"""Train-scene-CV-only selection of early epoch / AdamW decay.

Fixed processed58 -> 128 -> 64 -> 1 and frozen logit-MSE targets. Val arrays
are first resolved in evaluate(), after selection and all full-Train fits.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
from pathlib import Path
import random
import time

import numpy as np
from acoustic_branch.contracts import atomic_json,immutable_json,sha
from .input_supervision_audit import read
from .response_input_localization import REL as DATA,load_scene,predict,metrics
from .capacity_probe import REL as CAPACITY

REL='ReliabilityFusion/pilots/regularization_cv_v1'
DECAYS={'wd0':0.,'wd001':.01,'wd01':.1}


def folds_from_counts(refs):
    folds=[[]for _ in range(4)];loads=[0]*4
    for r in sorted(refs,key=lambda r:(-r['queries'],r['scene_id'])):
        k=min((j for j in range(4)if len(folds[j])<3),key=lambda j:(loads[j],j))
        folds[k].append(r['scene_id']);loads[k]+=r['queries']
    return [sorted(f)for f in folds]


def key(r):
    return (r['macro_R1'],r['macro_joint'],-r['loss'])


def early_rule(history,patience=8):
    best=history[0];stale=0;stop=history[-1]['epoch']
    for r in history[1:]:
        if key(r)>key(best):best=r;stale=0
        else:stale+=1
        if stale>=patience:stop=r['epoch'];break
    return dict(epoch=best['epoch'],stop_epoch=stop,metric=best,patience=patience)


def prepare(root):
    out=root/REL;out.mkdir(parents=True,exist_ok=True)
    ready=read(root/DATA/'ready.json');old=read(root/CAPACITY/'config.json')
    assert {r['split']for r in ready['scenes']}=={'train','val'}
    refs=[r for r in ready['scenes']if r['split']=='train']
    assert len(refs)==12 and set(r['scene_id']for r in refs)==set(old['selection']['train'])
    assert not set(old['selection']['train'])&set(old['selection']['val'])
    folds=folds_from_counts(refs)
    assert sorted(sum(folds,[]))==sorted(old['selection']['train'])
    cfg=dict(version='v1',hidden=[128,64],inputs=58,parameters=15873,seeds=[42,43],decays=DECAYS,folds=folds,
        selection={s:old['selection'][s]for s in('train','val')},epochs=40,batch_size=16,lr=.001,clip=1.,
        initialization=.1,supervision='Frozen logit(interior_target), squared error, eligible rows only',
        normalize='Fit scenes only: 9 in each CV fold; all12 in final refit. Never held scene or Val.',
        decay_parameters='ALL AdamW parameters including biases; same groups for all conditions',
        scene_balanced=True,scheduler=None,patience=8,min_delta=0,
        early_monitor='Across all four held-out Train folds AND both seeds: scene-macro R1, then Joint, then eligible-query pooled MSE. Include epoch0. Patience8 lexicographic improvements.',
        refit='One common epoch per decay from Train-CV early rule; both seeds refit all12 to this epoch. Fixed40 control uses same trajectory. No fold models ensembled.',
        decay_selection='Separately for early/fixed40: maximum Train-CV macro_R1, then macro_joint, then lower MSE; exact tie prefers lower decay. Never Val.',
        matched='Identical initialization/minibatch sequence across decays for each seed/fold; full wd0 Last must reproduce prior capacity Last exactly.',
        branch_note='Scene-held-out validation is for fusion MLP only; visual/acoustic encoders and their prepared features remain frozen, not retrained out-of-fold.',
        data_ready_sha256=sha(root/DATA/'ready.json'),old_capacity_config_sha256=sha(root/CAPACITY/'config.json'),test_accessed=False)
    immutable_json(out/'config.json',cfg)
    atomic_json(out/'status.json',dict(stage='prepared',test_accessed=False));print('PREPARED',folds,flush=True)


def tensors(scenes,fit):
    import torch
    cols=list(range(58))
    x=np.concatenate([s['x'][:,cols]for s in scenes if s['scene_id']in fit]);mean=x.mean(0);std=x.std(0);std=np.where(std>1e-6,std,1.)
    for s in scenes:
        s['xt']=torch.tensor((s['x'][:,cols]-mean)/std,dtype=torch.float32)
        s['tt']=torch.tensor(np.log(s['interior_target'])-np.log1p(-s['interior_target']),dtype=torch.float32)
    return mean,std


def evaluate_model(model,scenes,details=False):
    import torch
    with torch.inference_mode():
        per_scene={};losses=[];pp=[];tt=[];ff=[];predictions={}
        for s in scenes:
            raw=model.net(s['xt']).squeeze(-1);w=torch.sigmoid(raw).numpy();p=predict(s,w);fixed=s['grid_poses'][:,2]
            per_scene[s['scene_id']]=metrics(p,s['truth'],fixed)
            ids=np.flatnonzero(s['valid']);losses.extend((raw[ids]-s['tt'][ids]).square().numpy().tolist())
            pp.extend(p);tt.extend(s['truth']);ff.extend(fixed)
            if details:predictions[s['scene_id']]=dict(poses=p.tolist(),weights=w.tolist())
        return dict(micro=metrics(pp,tt,ff),per_scene=per_scene,loss=float(np.mean(losses)),loss_count=len(losses),
            macro_R1=float(np.mean([m['recall_1m']for m in per_scene.values()])),macro_joint=float(np.mean([m['recall_1m_30deg']for m in per_scene.values()])),
            **(dict(predictions=predictions)if details else{}))


def fit_task(task):
    import torch
    from .model import Gate
    from .revision_model import initialize
    root,ssd,decay_name,seed,fold=task;root,ssd=map(Path,(root,ssd));out=root/REL;cfg=read(out/'config.json')
    full=fold=='full';folder=out/('full'if full else'cv')/f'{decay_name}_seed{seed}_fold{fold}';folder.mkdir(parents=True,exist_ok=True)
    if(folder/'complete.json').exists():return read(folder/'complete.json')
    torch.set_num_threads(1);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    refs=sorted([r for r in read(root/DATA/'ready.json')['scenes']if r['split']=='train'],key=lambda r:r['scene_id'])
    assert len(refs)==12 and sha(root/DATA/'ready.json')==cfg['data_ready_sha256']
    scenes=[load_scene(root/DATA,r['scene_id'],root,ssd)for r in refs]
    held=[]if full else cfg['folds'][fold];fit=[s for s in scenes if s['scene_id']not in held];check=[s for s in scenes if s['scene_id']in held]
    fit_ids=[s['scene_id']for s in fit];assert set(fit_ids).isdisjoint(held)
    mean,std=tensors(scenes,fit_ids)
    model=Gate(58,hidden=(128,64));initialize(model,.1)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=DECAYS[decay_name])
    total=sum(int(s['valid'].sum())for s in fit);digest=hashlib.sha256();history=[];started=time.perf_counter();updates=0
    selected_epoch=read(out/'selection.json')['early'][decay_name]['epoch']if full else None
    saved={}
    def save(epoch):
        p=folder/f'epoch{epoch:03}.pt'
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),mean=mean,std=std,epoch=epoch,
            hidden=[128,64],columns=list(range(58)),seed=seed,decay=DECAYS[decay_name],fit_scenes=fit_ids,held_scenes=held,
            config_sha256=sha(out/'config.json'),selection_sha256=sha(out/'selection.json')if full else None,
            updates=updates,python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),test_accessed=False),p)
        saved[str(epoch)]=sha(p)
    if full and selected_epoch==0:save(0)
    if not full:history.append(dict(epoch=0,held=evaluate_model(model,check)))
    for epoch in range(1,41):
        model.train();batches=[]
        for s in fit:
            ids=np.flatnonzero(s['valid']);np.random.shuffle(ids)
            batches.extend((s,ids[i:i+16])for i in range(0,len(ids),16))
        random.shuffle(batches);summed=0.;count=0
        for s,ids in batches:
            digest.update(s['scene_id'].encode()+b'\0'+np.asarray(ids,dtype='<i8').tobytes())
            raw=model.net(s['xt'][ids]).squeeze(-1);values=(raw-s['tt'][ids]).square()
            objective=values.mean()*total/(len(fit)*int(s['valid'].sum()))
            optimizer.zero_grad(set_to_none=True);objective.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(grad):raise ValueError('Nonfinite gradient')
            optimizer.step();summed+=float(values.detach().sum());count+=len(ids);updates+=1
        model.eval()
        if full:
            history.append(dict(epoch=epoch,train_online_loss=summed/count))
            if epoch in(selected_epoch,40):save(epoch)
        else:history.append(dict(epoch=epoch,train_online_loss=summed/count,held=evaluate_model(model,check)))
        atomic_json(folder/'status.json',dict(epoch=epoch,elapsed_s=time.perf_counter()-started,test_accessed=False))
    result=dict(seed=seed,decay_name=decay_name,decay=DECAYS[decay_name],fold=fold,fit_scenes=fit_ids,held_scenes=held,
        history=history,saved=saved,selected_epoch=selected_epoch,batch_order_sha256=digest.hexdigest(),
        config_sha256=sha(out/'config.json'),elapsed_s=time.perf_counter()-started,test_accessed=False)
    atomic_json(folder/'complete.json',result);print('FIT',decay_name,seed,fold,round(result['elapsed_s'],2),flush=True);return result


def cv(root,ssd,workers):
    cfg=read(root/REL/'config.json');tasks=[(str(root),str(ssd),name,seed,fold)for name in DECAYS for seed in cfg['seeds']for fold in range(4)]
    with ProcessPoolExecutor(max_workers=workers)as pool:
        for i,f in enumerate(as_completed([pool.submit(fit_task,t)for t in tasks]),1):
            f.result();atomic_json(root/REL/'status.json',dict(stage='cv',done=i,total=len(tasks),test_accessed=False))
    for seed in cfg['seeds']:
        for fold in range(4):
            values=[read(root/REL/'cv'/f'{name}_seed{seed}_fold{fold}'/'complete.json')['batch_order_sha256']for name in DECAYS]
            assert len(set(values))==1
    atomic_json(root/REL/'cv_complete.json',dict(runs=24,matched_batch_orders=True,test_accessed=False))


def select(root):
    out=root/REL;cfg=read(out/'config.json');curves={};early={};hashes={}
    for name in DECAYS:
        runs=[]
        for seed in cfg['seeds']:
            for fold in range(4):
                p=out/'cv'/f'{name}_seed{seed}_fold{fold}'/'complete.json';hashes[str(p.relative_to(out))]=sha(p);runs.append(read(p))
        curves[name]=[]
        for e in range(41):
            rs=[r['history'][e]['held']for r in runs]
            curves[name].append(dict(epoch=e,macro_R1=float(np.mean([r['macro_R1']for r in rs])),
                macro_joint=float(np.mean([r['macro_joint']for r in rs])),
                loss=float(sum(r['loss']*r['loss_count']for r in rs)/sum(r['loss_count']for r in rs))))
        early[name]=early_rule(curves[name],cfg['patience'])
    # Insertion order (0,.01,.1) implements the exact-tie lower-decay rule.
    winners={p:max(DECAYS,key=lambda name:key(early[name]['metric']if p=='early'else curves[name][-1]))for p in('early','fixed40')}
    result=dict(early=early,curves=curves,winners=winners,cv_complete_sha256=sha(out/'cv_complete.json'),cv_hashes=hashes,
        selection_uses_only='Train scene-held-out metrics; NO Val arrays/predictions resolved',config_sha256=sha(out/'config.json'),test_accessed=False)
    immutable_json(out/'selection.json',result);print('SELECTION',early,'WINNERS',winners,flush=True)


def refit(root,ssd,workers):
    cfg=read(root/REL/'config.json');assert(root/REL/'selection.json').exists()
    tasks=[(str(root),str(ssd),name,seed,'full')for name in DECAYS for seed in cfg['seeds']]
    with ProcessPoolExecutor(max_workers=workers)as pool:
        for i,f in enumerate(as_completed([pool.submit(fit_task,t)for t in tasks]),1):
            f.result();atomic_json(root/REL/'status.json',dict(stage='refit',done=i,total=len(tasks),test_accessed=False))
    for seed in cfg['seeds']:
        values=[read(root/REL/'full'/f'{name}_seed{seed}_foldfull'/'complete.json')['batch_order_sha256']for name in DECAYS]
        assert len(set(values))==1
        old=read(root/CAPACITY/'runs'/f'128_64_seed{seed}'/'complete.json')
        assert values[0]==old['batch_order_sha256']
    atomic_json(root/REL/'refit_complete.json',dict(runs=6,matched_batch_orders=True,selection_sha256=sha(root/REL/'selection.json'),test_accessed=False))


def evaluate(root,ssd):
    import torch
    from .model import Gate
    out=root/REL;cfg=read(out/'config.json');selection=read(out/'selection.json');done=read(out/'refit_complete.json')
    assert sha(out/'selection.json')==done['selection_sha256']
    # Val first accessed AFTER Train-only selection and all refits completed.
    refs=sorted(read(root/DATA/'ready.json')['scenes'],key=lambda r:r['scene_id']);assert{r['split']for r in refs}=={'train','val'}
    scenes=[load_scene(root/DATA,r['scene_id'],root,ssd)for r in refs];results={}
    torch.set_num_threads(1)
    for name in DECAYS:
        for seed in cfg['seeds']:
            folder=out/'full'/f'{name}_seed{seed}_foldfull';receipt=read(folder/'complete.json')
            for phase,e in [('early',selection['early'][name]['epoch']),('fixed40',40)]:
                ckpt=folder/f'epoch{e:03}.pt';assert sha(ckpt)==receipt['saved'][str(e)]
                state=torch.load(ckpt,map_location='cpu',weights_only=False);assert state['selection_sha256']==sha(out/'selection.json')
                model=Gate(58,hidden=(128,64));model.load_state_dict(state['model']);model.eval()
                for s in scenes:
                    s['xt']=torch.tensor((s['x'][:,list(range(58))]-state['mean'])/state['std'],dtype=torch.float32)
                    s['tt']=torch.tensor(np.log(s['interior_target'])-np.log1p(-s['interior_target']),dtype=torch.float32)
                ev={split:evaluate_model(model,[s for s in scenes if s['split']==split],True)for split in('train','val')}
                if name=='wd0'and phase=='fixed40':
                    old=read(root/CAPACITY/'runs'/f'128_64_seed{seed}'/'complete.json')['evaluated']['last']
                    for split in('train','val'):
                        assert ev[split]['predictions']==old[split]['predictions']
                        assert ev[split]['micro']==old[split]['micro']
                        assert ev[split]['loss']==old[split]['loss']
                results[f'{name}_seed{seed}_{phase}']=dict(decay_name=name,decay=DECAYS[name],seed=seed,phase=phase,epoch=e,
                    cv_selected=selection['winners'][phase]==name,checkpoint_sha256=sha(ckpt),**ev)
    atomic_json(out/'evaluation.json',dict(methods=results,selection_sha256=sha(out/'selection.json'),
        old_wd0_last_reproduced_exactly=True,val_used_for_selection=False,test_accessed=False))
    atomic_json(out/'status.json',dict(stage='evaluated',test_accessed=False));print('EVALUATED: wd0 Last reproduced exactly',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','cv','select','refit','evaluate']);p.add_argument('--root',type=Path,required=True);p.add_argument('--ssd',type=Path,required=True);p.add_argument('--workers',type=int,default=6);a=p.parse_args()
    if a.mode in('prepare','select'):globals()[a.mode](a.root)
    elif a.mode=='evaluate':evaluate(a.root,a.ssd)
    else:globals()[a.mode](a.root,a.ssd,a.workers)

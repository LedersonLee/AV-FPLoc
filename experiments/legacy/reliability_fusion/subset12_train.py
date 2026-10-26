"""Automatic prerequisite gate -> matched Train12 fitting -> Val3 report only."""
import csv
import fcntl
from pathlib import Path
import random
import shutil
import time
import traceback

import numpy as np
import torch
from PIL import Image

from acoustic_branch.contracts import atomic_json,immutable_json,sha
from acoustic_branch.data import read
from acoustic_branch.pilot_evaluate import heat_values
from .model import Gate
from .revision_model import initialize
from .success_set import metrics,predict,set_loss
from .subset12_data import REL
from .subset12_scores import load_scenes,attach_targets


def losses(raw,good,grid,target_logit,kind):
    # Keep the target in logit space; sigmoid->logit at FP32 could round
    # a narrow endpoint interval to0/1 and create an infinite loss.
    if kind=='logit_regression':return (raw-target_logit).square()
    if kind=='success_set':return set_loss(torch.sigmoid(raw),good,grid)[0]
    raise ValueError('Unknown objective')


@torch.inference_mode()
def evaluate(model,scenes,mean,std,kind,grid,resident=None):
    poses=[];truth=[];per_scene={};loss_values=[];weights=[]
    for s in scenes:
        a=s['arrays'];raw=model.net(torch.tensor((a['features']-mean)/std,dtype=torch.float32,device='cuda')).squeeze(-1);w=torch.sigmoid(raw)
        actual=predict(s,w.cpu().numpy(),resident[s['scene_id']] if resident else None)
        fixed=s['grid_poses'][:,2];m=metrics(actual,a['truth'],fixed);per_scene[s['scene_id']]=m
        valid=torch.tensor(s['valid'],device='cuda')
        if valid.any():
            target=s['interior_target'][s['valid']]
            target_logit=torch.tensor(np.log(target)-np.log1p(-target),dtype=torch.float32,device='cuda')
            loss_values.extend(losses(raw[valid],torch.tensor(s['good'][s['valid']],device='cuda'),grid,target_logit,kind).cpu().tolist())
        poses.extend(actual);truth.extend(a['truth']);weights.extend(w.cpu().tolist())
    fixed_all=np.concatenate([s['grid_poses'][:,2] for s in scenes])
    return dict(micro=metrics(poses,truth,fixed_all),per_scene=per_scene,
        macro_R1=float(np.mean([r['recall_1m'] for r in per_scene.values()])),macro_joint=float(np.mean([r['recall_1m_30deg'] for r in per_scene.values()])),
        loss=float(np.mean(loss_values)),weight_mean=float(np.mean(weights)),weight_p10=float(np.quantile(weights,.1)),weight_p90=float(np.quantile(weights,.9)))


def train_run(root,scenes,config,mean,std,kind,seed,resident,status,tracker):
    out=root/REL;folder=out/'runs'/f'{kind}_seed{seed}';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():return read(folder/'complete.json')
    train=[s for s in scenes if s['split']=='train'];val=[s for s in scenes if s['split']=='val']
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=Gate(14).cuda();initialize(model,.1);optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
    grid=torch.tensor(config['weight_grid'],dtype=torch.float32,device='cuda');best=evaluate(model,val,mean,std,kind,grid,resident)
    initial=best;history=[];start=1;best_epoch=0;updates=0
    if (folder/'last.pt').exists():
        previous=torch.load(folder/'last.pt',map_location='cpu',weights_only=False);assert previous['config_sha256']==sha(out/'config.json') and previous['kind']==kind and previous['seed']==seed
        model.load_state_dict(previous['model']);optimizer.load_state_dict(previous['optimizer']);history=previous['history'];best=previous['best'];best_epoch=previous['best_epoch'];updates=previous['updates'];start=previous['epoch']+1
        random.setstate(previous['python_rng']);np.random.set_state(previous['numpy_rng']);torch.set_rng_state(previous['torch_rng']);torch.cuda.set_rng_state_all(previous['cuda_rng'])
    def save(path,epoch):
        state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),mean=mean,std=std,config_sha256=sha(out/'config.json'),kind=kind,seed=seed,
            epoch=epoch,best=best,best_epoch=best_epoch,updates=updates,history=history,test_accessed=False,
            python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
        temporary=path.with_suffix('.tmp.pt');torch.save(state,temporary);temporary.replace(path)
    if start==1:save(folder/'best.pt',0)
    last=evaluate(model,val,mean,std,kind,grid,resident);total=sum(s['valid'].sum() for s in train)
    for epoch in range(start,config['epochs']+1):
        batches=[]
        for s in train:
            ids=np.flatnonzero(s['valid']);np.random.shuffle(ids)
            batches.extend((s,ids[i:i+config['batch_size']]) for i in range(0,len(ids),config['batch_size']))
        random.shuffle(batches);sumloss=0.;count=0
        for s,ids in batches:
            x=torch.tensor((s['arrays']['features'][ids]-mean)/std,dtype=torch.float32,device='cuda');raw=model.net(x).squeeze(-1)
            target=torch.tensor(np.log(s['interior_target'][ids])-np.log1p(-s['interior_target'][ids]),dtype=torch.float32,device='cuda')
            value=losses(raw,torch.tensor(s['good'][ids],device='cuda'),grid,target,kind)
            objective=value.mean()*total/(len(train)*int(s['valid'].sum()))
            optimizer.zero_grad(set_to_none=True);objective.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(grad):raise ValueError('Nonfinite gradient')
            optimizer.step();updates+=1;sumloss+=float(value.detach().sum());count+=len(ids)
        last=evaluate(model,val,mean,std,kind,grid,resident);key=lambda m:(m['macro_R1'],m['macro_joint'],-m['loss'])
        improved=key(last)>key(best)
        if improved:best=last;best_epoch=epoch
        history.append(dict(epoch=epoch,updates=updates,train_loss=sumloss/count,val=last))
        if improved:save(folder/'best.pt',epoch)
        save(folder/'last.pt',epoch);atomic_json(folder/'history.json',history)
        status('training',kind=kind,seed=seed,epoch=epoch,best_epoch=best_epoch,val_R1=last['micro']['recall_1m'],val_joint=last['micro']['recall_1m_30deg'])
        if tracker:tracker.log({f'{kind}/seed{seed}/epoch':epoch,f'{kind}/seed{seed}/train_loss':sumloss/count,
            **{f'{kind}/seed{seed}/val/{k}':v for k,v in last['micro'].items()},f'{kind}/seed{seed}/weight_mean':last['weight_mean']})
    result=dict(kind=kind,seed=seed,best_epoch=best_epoch,last_epoch=config['epochs'],initial_val=initial,best_val=best,last_val=last,
        best_sha256=sha(folder/'best.pt'),last_sha256=sha(folder/'last.pt'),updates=updates,test_accessed=False)
    atomic_json(folder/'complete.json',result);return result


def report_results(root,ssd,report,scenes,config,training,mean,std,resident,status):
    val=[s for s in scenes if s['split']=='val'];train=[s for s in scenes if s['split']=='train'];out=root/REL
    policies={'vision_only':None,'acoustic_only':None,'equal_fusion':None,'fixed_01':None};labels=dict(vision_only='Vision-only · 고정 encoder',
        acoustic_only='Acoustic-only · Train12 encoder',equal_fusion='고정1:1 fusion',fixed_01='고정 acoustic w=0.1')
    train_diagnostics={};grid=torch.tensor(config['weight_grid'],dtype=torch.float32,device='cuda')
    for r in training:
        folder=out/'runs'/f'{r["kind"]}_seed{r["seed"]}'
        for phase in ('best','last'):
            state=torch.load(folder/(phase+'.pt'),map_location='cpu',weights_only=False)
            assert sha(folder/(phase+'.pt'))==r[phase+'_sha256'];model=Gate(14).cuda().eval();model.load_state_dict(state['model'])
            name=f'{r["kind"]}_seed{r["seed"]}_{phase}';policies[name]=model
            labels[name]=('Logit 회귀' if r['kind']=='logit_regression' else '성공 집합 · 동일Train12')+f' · seed{r["seed"]} · {phase} ep{state["epoch"]}'
            status('frozen_train_diagnostics',method=name)
            train_diagnostics[name]=evaluate(model,train,mean,std,r['kind'],grid)
    selected={}
    for kind in config['runs']:
        candidates=[r for r in training if r['kind']==kind];chosen=max(candidates,key=lambda r:(r['best_val']['macro_R1'],r['best_val']['macro_joint'],-r['best_val']['loss']))
        selected[kind]=dict(seed=chosen['seed'],epoch=chosen['best_epoch'],criterion=config['model_selection'])
    immutable_json(out/'selection_frozen.json',dict(selected=selected,config_sha256=sha(out/'config.json'),test_accessed=False))
    records=[];banks={};(report/'maps').mkdir(exist_ok=True);(report/'rgb').mkdir(exist_ok=True);(report/'heatmaps').mkdir(exist_ok=True)
    for s in val:
        a=s['arrays'];sid=s['scene_id'];q=read(s['folder']/'queries.json');weights={};poses={}
        for name,model in policies.items():
            with torch.no_grad():weights[name]=model(torch.tensor((a['features']-mean)/std,dtype=torch.float32,device='cuda')).cpu().numpy() if model else np.full(len(q),0 if name=='vision_only' else 1 if name=='acoustic_only' else .5 if name=='equal_fusion' else .1)
            poses[name]=predict(s,weights[name],resident[sid])
        # Fixed Val baseline must reproduce the already frozen grid controls.
        np.testing.assert_allclose(poses['fixed_01'],s['grid_poses'][:,2],atol=1e-7,rtol=0)
        meta=read(root/'2DFP'/sid/'floorplan_metadata.json');banks[sid]=dict(xy=a['xy'].tolist(),metadata=meta)
        shutil.copy2(root/'2DFP'/sid/'map.png',report/'maps'/(sid+'.png'))
        for i,query in enumerate(q):
            filename=sid+'_'+str(query['sample_index']);dest=report/'rgb'/(filename+'.png')
            if not dest.exists():shutil.copy2(ssd/query['image_path'],dest)
            v,ac=a['scores'][i];layers=[]
            for name in policies:layers.append(heat_values(((1-weights[name][i])*v+weights[name][i]*ac).max(1)))
            np.stack(layers).tofile(report/'heatmaps'/(filename+'.bin'))
            records.append(dict(scene_id=sid,sample_index=query['sample_index'],gt=a['truth'][i].tolist(),image='rgb/'+filename+'.png',heatmap='heatmaps/'+filename+'.bin',
                methods={n:dict(pose=poses[n][i].tolist(),weight=float(weights[n][i])) for n in policies}))
        status('val_report',scene_id=sid,queries=len(records))
    assert len(records)==1704
    result={}
    for name in policies:
        result[name]=metrics([r['methods'][name]['pose'] for r in records],[r['gt'] for r in records],[r['methods']['fixed_01']['pose'] for r in records])
        result[name]['per_scene']={sid:metrics([r['methods'][name]['pose'] for r in records if r['scene_id']==sid],
            [r['gt'] for r in records if r['scene_id']==sid],[r['methods']['fixed_01']['pose'] for r in records if r['scene_id']==sid]) for sid in config['selection']['val']}
    from .verify_overfit_probe import calculate
    checks=0;truth=np.array([r['gt'] for r in records]);fixed=np.array([r['methods']['fixed_01']['pose'] for r in records])
    assert len({(r['scene_id'],r['sample_index']) for r in records})==1704
    assert set(r['scene_id'] for r in records)==set(config['selection']['val'])
    for name,m in result.items():
        independently,_=calculate(np.array([r['methods'][name]['pose'] for r in records]),truth,fixed)
        for key,value in independently.items():np.testing.assert_allclose(value,m[key],atol=1e-12,rtol=0);checks+=1
    verification=dict(status='passed',metric_checks=checks,queries=1704,train_scenes=12,val_scenes=3,test_accessed=False,
        baseline_reproduced_queries=1704,notice='Independent metric formulas; same fixed-Val grid predictions reproduced; no test query files opened')
    atomic_json(out/'verification.json',verification);atomic_json(report/'verification.json',verification)
    summary=dict(status='complete',config=config,training=training,selected=selected,methods=result,labels=labels,method_order=list(policies),train_diagnostics=train_diagnostics,
        targets=read(out/'targets_ready.json')['scenes'],test_accessed=False,notice='Matched Train12/Val3 model development; Val selected checkpoints, not independent Test performance.')
    atomic_json(out/'evaluation.json',summary);atomic_json(report/'summary.json',summary);atomic_json(report/'records.json',records);atomic_json(report/'banks.json',banks)
    fields=[k for k in result['fixed_01'] if k not in ('per_scene','queries')]
    with (report/'metrics.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['method',*fields])
        for name in policies:writer.writerow([labels[name],*[result[name][k] for k in fields]])
    atomic_json(report/'histories.json',{f'{r["kind"]}_seed{r["seed"]}':read(out/'runs'/f'{r["kind"]}_seed{r["seed"]}'/'history.json') for r in training})


def run(root,ssd,cache,report):
    root,ssd,cache,report=map(Path,(root,ssd,cache,report));out=root/REL;config=read(out/'config.json');report.mkdir(parents=True,exist_ok=True)
    lock=(out/'pipeline.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);tracker=None
    def status(stage,**extra):
        value=dict(status='running',stage=stage,updated_unix=time.time(),test_accessed=False,**extra);atomic_json(out/'status.json',value);atomic_json(report/'status.json',value)
    try:
        shutil.copy2(Path(__file__).with_name('subset12_report.html'),report/'index.html');torch.set_num_threads(4);failures=[0,0]
        while not all((out/'preprocessing'/f'part{i}/features_ready.json').exists() for i in (0,1)):
            parts=[]
            for i in (0,1):
                p=out/'preprocessing'/f'part{i}/status.json';row=read(p) if p.exists() else dict(status='starting')
                render=out/'preprocessing'/f'part{i}/render/status.json'
                if render.exists():row=dict(row,render=read(render))
                failures[i]=failures[i]+1 if row['status']=='failed' else 0
                if failures[i]>=4:raise RuntimeError('CPU partition failed persistently: '+str(row))
                parts.append(row)
            status('wait_verified_train12_banks',partitions=parts);time.sleep(15)
        status('build_train12_scores');scenes=load_scenes(root,ssd,cache,config,status);attach_targets(root,cache,scenes,config,status)
        allx=np.concatenate([s['arrays']['features'] for s in scenes if s['split']=='train']);mean=allx.mean(0);std=allx.std(0);std=np.where(std>1e-6,std,1.)
        atomic_json(out/'normalization.json',dict(mean=mean.tolist(),std=std.tolist(),fit_split='train',queries=len(allx),features=14,test_accessed=False))
        resident={s['scene_id']:torch.tensor(np.asarray(s['arrays']['scores']),device='cuda') for s in scenes if s['split']=='val'}
        import wandb
        tracker=wandb.init(project='hssd-reliability-fusion',name='subset12-logit-vs-success-set-v1',dir=str(out),config=config)
        atomic_json(out/'wandb.json',dict(id=tracker.id,url=tracker.url));training=[]
        for seed in config['seeds']:
            for kind in config['runs']:
                training.append(train_run(root,scenes,config,mean,std,kind,seed,resident,status,tracker));atomic_json(out/'training_progress.json',training)
        report_results(root,ssd,report,scenes,config,training,mean,std,resident,status)
        atomic_json(out/'status.json',dict(status='complete',updated_unix=time.time(),test_accessed=False));atomic_json(report/'status.json',read(out/'status.json'))
    except Exception:
        value=dict(status='failed',error=traceback.format_exc(),updated_unix=time.time(),test_accessed=False);atomic_json(out/'status.json',value);atomic_json(report/'status.json',value);raise
    finally:
        if tracker:tracker.finish()


if __name__=='__main__':
    import sys
    run(*sys.argv[1:])

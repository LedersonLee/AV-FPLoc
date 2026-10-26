"""Independent-scalar supervision check -> unchanged core14 success-set gate.

Reuse frozen scores only. Train creates multi-valued weight targets and fits
normalization; val selects checkpoints. Test opens only after frozen selection.
"""
import argparse
import copy
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
from acoustic_branch.pilot_evaluate import heat_values
from .core import FEATURE_NAMES
from .model import Gate
from .revision_model import initialize
from .revision_data import REL as SOURCE

REL='ReliabilityFusion/pilots/success_set_v1'


def set_loss(weight,good,grid):
    """Nearest acceptable grid weight, not the mean of disconnected solutions.

    Empty sets have zero loss/gradient; never remove them from evaluation.
    Returns per-query loss and eligibility so normalization can be explicit.
    """
    valid=good.any(1)
    distance=(weight[:,None]-grid[None])**2
    # min (not amin) selects one nearest target on ties. Averaging gradients
    # across disconnected equidistant solutions can leave w stuck between them.
    best=distance.masked_fill(~good,torch.inf).min(1).values
    return torch.where(valid,best,torch.zeros_like(best)),valid


def good_weights(poses,truth):
    d=np.linalg.norm(poses[:,:,:2]-truth[:,None,:2],axis=2)
    a=abs((poses[:,:,2]-truth[:,None,2]+180)%360-180)
    return (d<=1)&(a<=30)


def prepare(root):
    root=Path(root);source=read(root/SOURCE/'config.json');ready=read(root/SOURCE/'scores_ready.json')
    config=dict(version='success_set_v1',source_config_sha256=sha(root/SOURCE/'config.json'),
        source_ready_sha256=sha(root/SOURCE/'scores_ready.json'),selection=source['selection'],
        split_sha256=source['split_sha256'],feature_columns=source['feature_columns'],
        feature_names=[FEATURE_NAMES[i] for i in source['feature_columns']],weight_grid=source['weight_grid'],
        seed=42,lr=.001,weight_decay=.001,batch_size=16,epochs=40,patience=8,initial_weight=.1,
        target='all searched weights achieving XY<=1m AND yaw<=30deg',
        loss='min squared distance to acceptable weight set; no added anchor regularizer',
        empty_target='zero supervised loss; retain every query in val/test metrics',
        balance='equal expected contribution per scene among eligible train queries',
        model_selection='val scene-macro R1; ties joint recall then lower within-arm set loss',
        sanity=dict(optimizer='projected SGD',lr=.5,steps=8,split='train only',learned_gate=False),
        no_new_RIRs=True,no_encoder_updates=True,test_notice='Repeated exploratory3test, not blind; no test selection')
    if ready['test_accessed'] or len(config['feature_names'])!=14:raise ValueError('Train/val core14 contract required')
    immutable_json(root/REL/'config.json',config);return config


def load_scenes(root,cache,config,test=False):
    if test and not (root/REL/'selection_frozen.json').exists():raise ValueError('Test before frozen selection')
    ready=read(root/SOURCE/('test_scores_ready.json' if test else 'scores_ready.json'))
    out=[]
    for e in ready['entries']:
        if (e['split']=='test')!=test or e['scene_id'] not in config['selection'][e['split']]:raise ValueError('Split mismatch')
        p=cache/e['cache_directory'];receipt=read(p/'receipt.json')
        if sha(p/'receipt.json')!=e['receipt_sha256']:raise ValueError('Source receipt changed')
        for name,digest in receipt['files'].items():
            if sha(p/name)!=digest:raise ValueError('Source array changed: '+name)
        arrays={k:np.load(p/(k+'.npy'),mmap_mode='r') for k in ('scores','features','truth','xy')}
        scene=dict(**e,folder=p,arrays=arrays)
        if not test:
            with np.load(p/'weight_curve.npz') as blob:poses=blob['poses'].copy()
            scene['good']=good_weights(poses,arrays['truth']);scene['grid_poses']=poses
        out.append(scene)
    return out


def metrics(poses,truth,reference=None):
    poses,truth=map(np.asarray,(poses,truth));d=np.linalg.norm(poses[:,:2]-truth[:,:2],axis=1);a=abs((poses[:,2]-truth[:,2]+180)%360-180)
    m={f'recall_{t:g}m':float((d<=t).mean()) for t in (.1,.5,1)}
    m.update({f'yaw_within_{t}deg':float((a<=t).mean()) for t in (10,30,50)})
    m.update(queries=len(d),recall_1m_30deg=float(((d<=1)&(a<=30)).mean()),median_xy_m=float(np.median(d)),median_yaw_deg=float(np.median(a)))
    if reference is not None:
        reference=np.asarray(reference);rd=np.linalg.norm(reference[:,:2]-truth[:,:2],axis=1);ra=abs((reference[:,2]-truth[:,2]+180)%360-180)
        before,after=rd<=1,d<=1;bj,aj=before&(ra<=30),after&(a<=30)
        m.update(repair_vs_fixed_xy=int((~before&after).sum()),harm_vs_fixed_xy=int((before&~after).sum()),
            repair_vs_fixed_joint=int((~bj&aj).sum()),harm_vs_fixed_joint=int((bj&~aj).sum()))
    return m


@torch.inference_mode()
def predict(scene,weights,resident=None):
    a=scene['arrays'];poses=[]
    for first in range(0,len(weights),16):
        ws=torch.tensor(weights[first:first+16],dtype=torch.float64,device='cuda')
        bank=resident[first:first+16] if resident is not None else torch.tensor(np.asarray(a['scores'][first:first+16]),device='cuda')
        fused=(1-ws[:,None,None])*bank[:,0]+ws[:,None,None]*bank[:,1]
        ids=fused.flatten(1).argmax(1).cpu().numpy();poses.extend(np.column_stack([a['xy'][ids//36],ids%36*10]))
    return np.asarray(poses)


def scalar_sanity(root,scenes,config,status):
    grid=torch.tensor(config['weight_grid'],dtype=torch.float64,device='cuda');rows=[];valid_count=0;distance_max=0.
    for s in scenes:
        if s['split']!='train':continue
        good=torch.tensor(s['good'],device='cuda');w=torch.nn.Parameter(torch.full((len(good),),.1,dtype=torch.float64,device='cuda'))
        optimizer=torch.optim.SGD([w],lr=config['sanity']['lr'])
        for _ in range(config['sanity']['steps']):
            loss,valid=set_loss(w,good,grid);optimizer.zero_grad();loss.sum().backward();optimizer.step()
            with torch.no_grad():w.clamp_(0,1)
        ws=w.detach().cpu().numpy();poses=predict(s,ws);truth=s['arrays']['truth']
        actual=(np.linalg.norm(poses[:,:2]-truth[:,:2],axis=1)<=1)&(abs((poses[:,2]-truth[:,2]+180)%360-180)<=30)
        allowed=s['good'].any(1);valid_count+=int(allowed.sum())
        discrepancy=int((allowed&~actual).sum());distance_max=max(distance_max,float(loss[valid].max().detach()) if valid.any() else 0.)
        rows.append(dict(scene_id=s['scene_id'],queries=len(ws),eligible=int(allowed.sum()),success_after=int(actual.sum()),
            failures_despite_successful_grid=discrepancy,optimized_weights=ws.tolist()))
        status('independent_scalar_check',scene_id=s['scene_id'],eligible=int(allowed.sum()),misses=discrepancy)
    result=dict(status='passed' if sum(r['failures_despite_successful_grid'] for r in rows)==0 else 'failed',
        train_queries=sum(r['queries'] for r in rows),eligible_queries=valid_count,per_scene=rows,max_final_set_loss=distance_max,
        note='Independent parameter per train query; GT-dependent debugging only, NOT held-out localization performance',test_accessed=False)
    atomic_json(root/REL/'scalar_sanity.json',result)
    if result['status']!='passed':raise ValueError('Independent scalar failed; do not start shared gate training')
    return result


@torch.inference_mode()
def evaluate(model,scenes,mean,std,resident,grid):
    model.eval();allposes=[];alltruth=[];per_scene={};losses=[];ws_all=[]
    for s in scenes:
        x=(s['arrays']['features']-mean)/std;ws=model(torch.tensor(x,dtype=torch.float32,device='cuda'))
        poses=predict(s,ws.cpu().numpy(),resident[s['scene_id']]);truth=s['arrays']['truth'];m=metrics(poses,truth)
        good=torch.tensor(s['good'],device='cuda');loss,valid=set_loss(ws,good,grid)
        losses.append(float(loss[valid].mean()) if valid.any() else 0.);per_scene[s['scene_id']]=m
        allposes.extend(poses);alltruth.extend(truth);ws_all.extend(ws.cpu().tolist())
    return dict(micro=metrics(allposes,alltruth),per_scene=per_scene,
        macro_R1=float(np.mean([m['recall_1m'] for m in per_scene.values()])),
        macro_joint=float(np.mean([m['recall_1m_30deg'] for m in per_scene.values()])),
        set_loss=float(np.mean(losses)),weight_mean=float(np.mean(ws_all)),weight_p10=float(np.quantile(ws_all,.1)),weight_p90=float(np.quantile(ws_all,.9)))


def key(m):return m['macro_R1'],m['macro_joint'],-m['set_loss']


def train(root,scenes,config,status):
    out=root/REL;run=out/'run';run.mkdir(exist_ok=True)
    if (run/'complete.json').exists():return read(run/'complete.json')
    train_scenes=[s for s in scenes if s['split']=='train'];val=[s for s in scenes if s['split']=='val']
    x=np.concatenate([s['arrays']['features'] for s in train_scenes]);mean=x.mean(0);std=x.std(0);std=np.where(std>1e-6,std,1.)
    random.seed(config['seed']);np.random.seed(config['seed']);torch.manual_seed(config['seed']);torch.cuda.manual_seed_all(config['seed'])
    model=Gate(14).cuda();initialize(model,.1);optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
    grid=torch.tensor(config['weight_grid'],dtype=torch.float32,device='cuda')
    resident={s['scene_id']:torch.tensor(np.asarray(s['arrays']['scores']),device='cuda') for s in val}
    best=evaluate(model,val,mean,std,resident,grid);history=[];bad=0;best_epoch=0;start=1
    if (run/'last.pt').exists():
        state=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
        if state['config_sha256']!=sha(out/'config.json'):raise ValueError('Resume contract changed')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);best=state['best'];history=state['history'];bad=state['bad'];best_epoch=state['best_epoch'];start=state['epoch']+1
        random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
    def save(path,epoch):
        value=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,best=best,best_epoch=best_epoch,bad=bad,history=history,
            mean=mean,std=std,config_sha256=sha(out/'config.json'),feature_names=config['feature_names'],test_accessed=False,
            python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
        tmp=path.with_suffix('.tmp.pt');torch.save(value,tmp);tmp.replace(path)
    if start==1:save(run/'best.pt',0)
    tracker=None
    try:
        import wandb
        tracker=wandb.init(project='hssd-reliability-fusion',name='success-set-core14-v1',dir=str(run),config=config,
            id=read(run/'wandb.json')['id'] if (run/'wandb.json').exists() else None,resume='allow')
        atomic_json(run/'wandb.json',dict(id=tracker.id,url=tracker.url))
        last=evaluate(model,val,mean,std,resident,grid)
        for epoch in range(start,config['epochs']+1):
            if bad>=config['patience']:break
            model.train();batches=[]
            for s in train_scenes:
                ids=np.flatnonzero(s['good'].any(1));np.random.shuffle(ids)
                batches.extend((s,ids[i:i+config['batch_size']]) for i in range(0,len(ids),config['batch_size']))
            random.shuffle(batches);total=sum(int(s['good'].any(1).sum()) for s in train_scenes);losses=[]
            for s,ids in batches:
                x=(s['arrays']['features'][ids]-mean)/std;w=model(torch.tensor(x,dtype=torch.float32,device='cuda'))
                loss,_=set_loss(w,torch.tensor(s['good'][ids],device='cuda'),grid);objective=loss.mean()
                balanced=objective*total/(len(train_scenes)*int(s['good'].any(1).sum()))
                optimizer.zero_grad(set_to_none=True);balanced.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();losses.append(float(objective.detach()))
            last=evaluate(model,val,mean,std,resident,grid)
            if key(last)>key(best):best=last;best_epoch=epoch;bad=0;save(run/'best.pt',epoch)
            else:bad+=1
            history.append(dict(epoch=epoch,train_set_loss=float(np.mean(losses)),val=last));save(run/'last.pt',epoch)
            atomic_json(run/'history.json',history);status('training',epoch=epoch,best_epoch=best_epoch,val=last)
            tracker.log(dict(epoch=epoch,train_set_loss=float(np.mean(losses)),val_set_loss=last['set_loss'],val_R1=last['micro']['recall_1m'],
                val_joint=last['micro']['recall_1m_30deg'],weight_mean=last['weight_mean']))
            print('success_set',epoch,last,flush=True)
        result=dict(status='complete',best_epoch=best_epoch,last_epoch=history[-1]['epoch'],best_val=best,last_val=last,
            best_sha256=sha(run/'best.pt'),last_sha256=sha(run/'last.pt'),test_accessed=False)
        atomic_json(run/'complete.json',result);return result
    finally:
        if tracker:tracker.finish()
        del resident


def evaluate_test(root,cache,report,config,training,status):
    out=root/REL;source_selection=read(root/SOURCE/'selection_frozen.json');margin=next(r for r in source_selection['results'] if r['name']=='adaptive_margin_joint')
    states={}
    for name,path,digest in [('set_best',out/'run/best.pt',training['best_sha256']),('set_last',out/'run/last.pt',training['last_sha256']),
        ('margin_best',root/SOURCE/'runs/adaptive_margin_joint/best.pt',margin['best_sha256']),
        ('margin_last',root/SOURCE/'runs/adaptive_margin_joint/last.pt',margin['last_sha256'])]:
        if sha(path)!=digest:raise ValueError('Frozen evaluation checkpoint changed')
        state=torch.load(path,map_location='cpu',weights_only=False);model=Gate(14).cuda().eval();model.load_state_dict(state['model']);states[name]=(model,state)
    methods=['vision_only','fixed_01','margin_best','margin_last','set_best','set_last'];rows=[];banks={}
    original_report=Path('/reference_report');old={(r['scene_id'],r['sample_index']):r for r in read(original_report/'records.json')}
    scenes=load_scenes(root,cache,config,test=True);(report/'heatmaps').mkdir(exist_ok=True)
    for s in scenes:
        a=s['arrays'];queries=read(s['folder']/'queries.json');weights={};predictions={}
        for name,(model,state) in states.items():
            with torch.no_grad():w=model(torch.tensor((a['features']-state['mean'])/state['std'],dtype=torch.float32,device='cuda')).cpu().numpy()
            weights[name]=w;predictions[name]=predict(s,w)
        weights['fixed_01']=np.full(len(queries),.1);weights['vision_only']=np.zeros(len(queries))
        for n in ('fixed_01','vision_only'):predictions[n]=predict(s,weights[n])
        for i,q in enumerate(queries):
            ref=old[(s['scene_id'],q['sample_index'])]
            for name,oldname in [('fixed_01','train_fixed'),('vision_only','vision_only'),('margin_best','control_adaptive_margin_joint')]:
                np.testing.assert_allclose(predictions[name][i],ref['methods'][oldname]['pose_xzyaw'],atol=1e-7,rtol=0)
            row=dict(scene_id=s['scene_id'],sample_index=q['sample_index'],gt=a['truth'][i].tolist(),
                methods={n:dict(pose=predictions[n][i].tolist(),weight=float(weights[n][i])) for n in methods})
            v,ac=a['scores'][i];filename=s['scene_id']+'_'+str(q['sample_index'])+'.bin'
            np.stack([heat_values(((1-weights[n][i])*v+weights[n][i]*ac).max(1)) for n in methods]).tofile(report/'heatmaps'/filename)
            row['heatmap']='heatmaps/'+filename;rows.append(row)
        status('test_evaluation',scene_id=s['scene_id'],completed_queries=len(rows))
    result={}
    for n in methods:
        result[n]=metrics([r['methods'][n]['pose'] for r in rows],[r['gt'] for r in rows],[r['methods']['fixed_01']['pose'] for r in rows])
        result[n]['per_scene']={sid:metrics([r['methods'][n]['pose'] for r in rows if r['scene_id']==sid],
            [r['gt'] for r in rows if r['scene_id']==sid],[r['methods']['fixed_01']['pose'] for r in rows if r['scene_id']==sid]) for sid in config['selection']['test']}
    if len(rows)!=1472:raise ValueError('Test coverage mismatch')
    labels=dict(vision_only='Vision-only · 참고',fixed_01='고정 acoustic w=0.1',margin_best='XY·yaw margin · Best ep'+str(margin['best_epoch']),
        margin_last='XY·yaw margin · Last ep'+str(margin['last_epoch']),set_best='성공 집합 · Best ep'+str(training['best_epoch']),set_last='성공 집합 · Last ep'+str(training['last_epoch']))
    summary=dict(status='complete',methods=result,labels=labels,method_order=methods,training=training,queries=1472,config=config,
        comparisons='Repair/Harm relative to fixed w=.1, NOT Vision',test_notice=config['test_notice'])
    atomic_json(report/'summary.json',summary);atomic_json(report/'records.json',rows);shutil.copy2(original_report/'banks.json',report/'banks.json')
    import csv
    fields=[k for k in result['fixed_01'] if k not in ('per_scene','queries')]
    with (report/'metrics.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['method',*fields])
        for n,m in result.items():writer.writerow([labels[n],*[m[k] for k in fields]])
    atomic_json(out/'evaluation.json',summary)


def run(root,cache,report):
    root,cache,report=map(Path,(root,cache,report));out=root/REL;config=read(out/'config.json');out.mkdir(parents=True,exist_ok=True);report.mkdir(parents=True,exist_ok=True)
    lock=(out/'pipeline.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def status(stage,**extra):
        value=dict(status='running',stage=stage,updated_unix=time.time(),test_accessed=stage=='test_evaluation',**extra)
        atomic_json(out/'status.json',value);atomic_json(report/'status.json',value)
    shutil.copy2(Path(__file__).with_name('success_set_report.html'),report/'index.html')
    try:
        if sha(root/SOURCE/'config.json')!=config['source_config_sha256'] or sha(root/SOURCE/'scores_ready.json')!=config['source_ready_sha256']:
            raise ValueError('Frozen source changed')
        torch.set_num_threads(4);status('verify_train_val_scores');scenes=load_scenes(root,cache,config)
        sanity=scalar_sanity(root,scenes,config,status);atomic_json(report/'scalar_sanity.json',sanity)
        training=train(root,scenes,config,status)
        immutable_json(out/'selection_frozen.json',dict(config_sha256=sha(out/'config.json'),training=training,test_accessed=False))
        evaluate_test(root,cache,report,config,training,status)
        from .fusion_latency import benchmark
        latency=benchmark(scenes,out/'run/best.pt',config);atomic_json(out/'latency.json',latency);atomic_json(report/'latency.json',latency)
        atomic_json(out/'status.json',dict(status='complete',updated_unix=time.time()));atomic_json(report/'status.json',read(out/'status.json'))
    except Exception:
        value=dict(status='failed',error=traceback.format_exc(),updated_unix=time.time());atomic_json(out/'status.json',value);atomic_json(report/'status.json',value);raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','cache','report'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();run(a.root,a.cache,a.report)

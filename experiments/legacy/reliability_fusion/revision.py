"""Frozen E+C+N / 14-feature fusion controls, then one post-selection test pass.

Old results are immutable. Train fits normalization, constant weight and gates;
val selects checkpoints/arm. Previously inspected test is exploratory only.
"""
import argparse
import copy
import csv
import fcntl
import math
import random
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
from numpy.lib.format import open_memmap
import torch

from acoustic_branch.contracts import atomic_json,immutable_json,sha
from acoustic_branch.data import read
from acoustic_branch.evaluation import select,standardized,diagnose
from acoustic_branch.input_ablation_dense import embeddings
from acoustic_branch.pilot_evaluate import feature_arrays,heat_values
from acoustic_branch.train import AcousticEncoder
from .core import features,target_statistics,FEATURE_NAMES
from .prepare import REL as OLD
from .revision_data import REL,MODEL
from .revision_model import Gate,initialize,objective,arm


def set_status(out,stage,**kwargs):
    value=dict(status='running',stage=stage,updated_unix=time.time(),**kwargs)
    atomic_json(out/'status.json',value)
    if Path('/report').is_dir():atomic_json(Path('/report/status.json'),value)


def checked_old(root,entry):
    folder=root/entry['directory'];receipt=read(folder/'receipt.json')
    if sha(folder/'receipt.json')!=entry['receipt_sha256']:raise ValueError('Old receipt changed')
    for name,digest in receipt['files'].items():
        if sha(folder/name)!=digest:raise ValueError('Old score cache changed: '+name)
    return folder,receipt


def envelope_arrays(root,out,entry,ready):
    result={}
    for role in ('FP','Observed'):
        row=next(e for e in ready['entries'] if (e['scene_id'],e['role'])==(entry['scene_id'],role))
        arrays=[]
        for chunk in row['chunks']:
            path=root/chunk['path'];receipt=path.parent/'receipt.json'
            if sha(receipt)!=chunk['receipt_sha256'] or sha(path)!=read(receipt)['sha256']:raise ValueError('Envelope changed')
            arrays.append(np.load(path,allow_pickle=False))
        result[role]=np.concatenate(arrays)
    return result


def build_scores(root,cache,config,test=False):
    out=root/REL;dest=cache/('test' if test else 'train_val');dest.mkdir(parents=True,exist_ok=True)
    ready_path=out/('test_scores_ready.json' if test else 'scores_ready.json')
    if ready_path.exists():
        entries=read(ready_path)['entries']
        for e in entries:
            folder=cache/e['cache_directory'];receipt=read(folder/'receipt.json')
            if sha(folder/'receipt.json')!=e['receipt_sha256']:raise ValueError('Revised cache receipt changed')
            for name,digest in receipt['files'].items():
                if sha(folder/name)!=digest:raise ValueError('Revised score payload changed')
        return entries
    if test:
        frozen=read(out/'selection_frozen.json')
        if frozen['config_sha256']!=sha(out/'config.json'):raise ValueError('Selection not frozen')
        index=read(root/'AcousticContrastive/pilots/scene_subset_v1/corrected_yup_v2/evaluation/bank_index.json')
        sources=read(root/OLD/'test_scores_ready.json')['entries']
        emb=root/MODEL/'dense_test/embeddings/all'
    else:
        index=read(root/OLD/'bank_index.json');sources=read(root/OLD/'scores_ready.json')['entries']
        old_ready=read(root/OLD/'ready.json');env_ready=read(out/'envelope_ready.json')
        if env_ready['config_sha256']!=sha(out/'config.json'):raise ValueError('Envelope config mismatch')
        state=torch.load(root/config['acoustic_checkpoint'],map_location='cpu',weights_only=False)
        model=AcousticEncoder(**state['config']['model']).cuda().eval();model.load_state_dict(state['model'])
    results=[]
    for entry in index['scenes']:
        split='test' if test else entry['split'];scene=entry['scene_id']
        if scene not in config['selection'][split]:raise ValueError('Scene outside frozen split')
        folder=dest/split/scene;folder.mkdir(parents=True,exist_ok=True)
        rp=folder/'receipt.json'
        if not rp.exists():
            source=next(e for e in sources if e['scene_id']==scene);old,old_receipt=checked_old(root,source)
            xy=np.load(old/'xy.npy');truth=np.load(old/'truth.npy');queries=read(old/'queries.json')
            np.testing.assert_allclose(truth,[[q['centre_xyz_m'][0],q['centre_xyz_m'][2],q['world_yaw_deg']] for q in queries],atol=1e-7,rtol=0)
            if test:
                contract=read(root/MODEL/'dense_test/contract.json')
                if contract['models']['all']!=config['acoustic_sha256']:raise ValueError('Test embeddings use different model')
                candidate=np.load(emb/(scene+'_candidate.npy'),mmap_mode='r');observed=np.load(emb/(scene+'_observed.npy'))
            else:
                fp=feature_arrays(root,old_ready,scene,'FP');obs=feature_arrays(root,old_ready,scene,'Observed')
                envelopes=envelope_arrays(root,out,entry,env_ready)
                candidate,observed=embeddings(model,entry,fp,obs,envelopes,queries,256)
            np.testing.assert_allclose(xy,entry['candidate_xz'],atol=1e-8,rtol=0)
            if candidate.shape!=(len(xy),36,128) or observed.shape!=(len(queries),128):raise ValueError('Embedding shape mismatch')
            old_scores=np.load(old/'scores.npy',mmap_mode='r')
            scores=open_memmap(folder/'scores.tmp.npy',mode='w+',dtype=np.float64,shape=old_scores.shape)
            small={k:[] for k in ('features','target_expectations')}
            cg=torch.tensor(np.asarray(candidate).reshape(-1,128),dtype=torch.float64,device='cuda')
            for first in range(0,len(queries),8):
                audio=(torch.tensor(observed[first:first+8],dtype=torch.float64,device='cuda')@cg.T).cpu().numpy().reshape(-1,len(xy),36)
                for j,a in enumerate(audio):
                    i=first+j;v=np.asarray(old_scores[i,0]);scores[i,0]=v;scores[i,1]=standardized(a)
                    f=features(v,a,xy,np.ones(11))[config['feature_columns']]
                    small['features'].append(f)
                    expected,_=target_statistics(v,scores[i,1],xy,truth[i]);small['target_expectations'].append(expected)
                set_status(out,'score_cache',split=split,scene_id=scene,queries=min(first+8,len(queries)),total=len(queries),test_accessed=test)
            del cg;old_scores._mmap.close();scores.flush();scores._mmap.close()
            (folder/'scores.tmp.npy').replace(folder/'scores.npy')
            for name,value in small.items():np.save(folder/(name+'.npy'),np.asarray(value))
            for name in ('xy.npy','truth.npy','queries.json'):shutil.copy2(old/name,folder/name)
            names=['scores.npy','features.npy','target_expectations.npy','xy.npy','truth.npy','queries.json']
            immutable_json(rp,dict(files={n:sha(folder/n) for n in names},config_sha256=sha(out/'config.json'),
                source_receipt_sha256=source['receipt_sha256'],queries=len(queries),scene_id=scene,split=split,
                acoustic_checkpoint_sha256=config['acoustic_sha256'],test_accessed=test))
        receipt=read(rp)
        if receipt['config_sha256']!=sha(out/'config.json'):raise ValueError('Partial cache config mismatch')
        results.append(dict(scene_id=scene,split=split,cache_directory=str(folder.relative_to(cache)),receipt_sha256=sha(rp)))
    atomic_json(ready_path,dict(entries=results,status='complete',test_accessed=test))
    if not test:del model
    return results


def load(cache,entries):
    result=[]
    for entry in entries:
        folder=cache/entry['cache_directory'];a={k:np.load(folder/(k+'.npy'),mmap_mode='r') for k in
            ('scores','features','target_expectations','truth','xy')}
        result.append(dict(**entry,arrays=a,folder=folder))
    return result


def summarize(poses,truth,vision=None):
    poses=np.asarray(poses);truth=np.asarray(truth)
    distance=np.linalg.norm(poses[:,:2]-truth[:,:2],axis=1);yaw=abs((poses[:,2]-truth[:,2]+180)%360-180)
    r={f'recall_{t:g}m':float((distance<=t).mean()) for t in (.1,.5,1)}
    r.update(recall_1m_30deg=float(((distance<=1)&(yaw<=30)).mean()),median_xy_m=float(np.median(distance)),
        median_yaw_deg=float(np.median(yaw)),queries=len(poses))
    r.update({f'yaw_within_{t}deg':float((yaw<=t).mean()) for t in (10,30,50)})
    if vision is not None:
        vo=np.linalg.norm(np.asarray(vision)[:,:2]-truth[:,:2],axis=1)<=1
        r.update(repair=int((~vo&(distance<=1)).sum()),harm=int((vo&(distance>1)).sum()))
    return r


def aggregate(rows):
    poses=np.concatenate([r['poses'] for r in rows]);truth=np.concatenate([r['truth'] for r in rows])
    return dict(micro=summarize(poses,truth),scene_macro_recall_1m=float(np.mean([summarize(r['poses'],r['truth'])['recall_1m'] for r in rows])),
        scene_macro_joint=float(np.mean([summarize(r['poses'],r['truth'])['recall_1m_30deg'] for r in rows])))


def curve(root,scenes,config):
    out=root/REL; weights=np.asarray(config['weight_grid']);result={}
    for scene in scenes:
        path=scene['folder']/'weight_curve.npz'
        if not path.exists():
            a=scene['arrays'];values=[];nll=[]
            for i,gt in enumerate(a['truth']):
                v,ac=a['scores'][i];w=torch.tensor(weights,dtype=torch.float64,device='cuda')[:,None,None]
                s=(1-w)*torch.tensor(v,device='cuda')+w*torch.tensor(ac,device='cuda')
                win=s.flatten(1).argmax(1).cpu().numpy()
                values.append(np.column_stack([a['xy'][win//36],win%36*10]))
                expected=(1-weights)*a['target_expectations'][i,0]+weights*a['target_expectations'][i,1]
                nll.append(np.stack([(torch.logsumexp(s.flatten(1)/t,dim=1).cpu().numpy()-expected/t) for t in config['score_temperature_grid']]))
                if i%100==0:set_status(out,'weight_oracle_controls',split=scene['split'],scene_id=scene['scene_id'],query=i,total=len(a['truth']),test_accessed=False)
            np.savez_compressed(path,poses=np.asarray(values),nll=np.asarray(nll))
        with np.load(path) as blob:result[scene['scene_id']]={k:blob[k].copy() for k in blob.files}
    return result


def fixed_and_diagnosis(scenes,curves,config):
    weights=np.array(config['weight_grid']);diagnosis={};train_rows=[]
    for split in ('train','val'):
        subset=[s for s in scenes if s['split']==split];curve_rows=[]
        for wi,w in enumerate(weights):
            rows=[dict(poses=curves[s['scene_id']]['poses'][:,wi],truth=s['arrays']['truth']) for s in subset]
            curve_rows.append(dict(weight=float(w),**aggregate(rows)))
        diagnosis[split]=dict(fixed_curve=curve_rows,oracles={})
        for name in ('success_xy','success_joint',*['nll_T'+str(t) for t in config['score_temperature_grid']]):
            rows=[]
            for s in subset:
                c=curves[s['scene_id']];poses=c['poses'];truth=s['arrays']['truth']
                d=np.linalg.norm(poses[:,:,:2]-truth[:,None,:2],axis=2);y=abs((poses[:,:,2]-truth[:,None,2]+180)%360-180)
                value=(d<=1).astype(float) if name=='success_xy' else ((d<=1)&(y<=30)).astype(float) if name=='success_joint' else -c['nll'][:,config['score_temperature_grid'].index(float(name[5:]))]
                order=np.argsort(abs(weights-.5),kind='stable');ids=order[value[:,order].argmax(1)]
                rows.append(dict(poses=poses[np.arange(len(ids)),ids],truth=truth))
            diagnosis[split]['oracles'][name]=aggregate(rows)
        if split=='train':train_rows=curve_rows
    winner=max(range(len(weights)),key=lambda i:(train_rows[i]['scene_macro_recall_1m'],train_rows[i]['scene_macro_joint'],-abs(weights[i]-.5)))
    return float(weights[winner]),diagnosis


def mask(xy,truth,joint):
    d=np.linalg.norm(xy[None,:,:]-truth[:,None,:2],axis=2)
    m=np.broadcast_to(d[:,:,None]<=1,(len(truth),len(xy),36)).copy()
    if joint:m &= abs((np.arange(36)[None,None,:]*10-truth[:,None,2,None]+180)%360-180)<=30
    return m.reshape(len(truth),-1)


@torch.no_grad()
def evaluate(model,scenes,mean,std,spec):
    model.eval();rows=[];losses=[];weights=[]
    for s in scenes:
        a=s['arrays'];x=(a['features']-mean)/std
        ws=model(torch.tensor(x,dtype=torch.float32,device='cuda')).cpu().numpy();poses=[];ll=[]
        for first in range(0,len(ws),16):
            stop=first+16;w=ws[first:stop];bank=np.asarray(a['scores'][first:stop])
            fused=(1-w[:,None,None])*bank[:,0]+w[:,None,None]*bank[:,1]
            ids=fused.reshape(len(w),-1).argmax(1)
            poses.extend(np.column_stack([a['xy'][ids//36],ids%36*10]))
            ins=mask(a['xy'],a['truth'][first:stop],spec['margin_kind']=='joint') if spec['margin_kind'] else None
            ll.extend(objective(torch.tensor(w,device='cuda'),torch.tensor(bank[:,0].reshape(len(w),-1),device='cuda'),
                torch.tensor(bank[:,1].reshape(len(w),-1),device='cuda'),torch.tensor(a['target_expectations'][first:stop],device='cuda'),
                spec['temperature'],None if ins is None else torch.tensor(ins,device='cuda')).cpu().numpy())
        rows.append(dict(poses=np.asarray(poses),truth=a['truth']));losses.append(float(np.mean(ll)));weights.extend(ws.tolist())
    result=aggregate(rows);result.update(loss=float(np.mean(losses)),weight_mean=float(np.mean(weights)),
        weight_p10=float(np.quantile(weights,.1)),weight_p90=float(np.quantile(weights,.9)))
    return result


def selection_key(result):
    return result['scene_macro_recall_1m'],result['scene_macro_joint'],-result['loss']


def train_arm(root,scenes,config,fixed,name,mean,std):
    out=root/REL;folder=out/'runs'/name;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():return read(folder/'complete.json')
    spec=arm(name);train=[s for s in scenes if s['split']=='train'];val=[s for s in scenes if s['split']=='val']
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    model=Gate(14,spec['constant']).cuda();initialize(model,.5 if spec['half_initialization'] else float(np.clip(fixed,.01,.99)))
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
    best=evaluate(model,val,mean,std,spec);history=[];best_epoch=0;bad=0;start=1
    if (folder/'last.pt').exists():
        checkpoint=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
        if checkpoint['config_sha256']!=sha(out/'config.json'):raise ValueError('Resume config changed')
        model.load_state_dict(checkpoint['model']);optimizer.load_state_dict(checkpoint['optimizer'])
        best=checkpoint['best'];best_epoch=checkpoint['best_epoch'];bad=checkpoint['bad'];history=checkpoint['history'];start=checkpoint['epoch']+1
        random.setstate(checkpoint['python_rng']);np.random.set_state(checkpoint['numpy_rng'])
        torch.set_rng_state(checkpoint['torch_rng']);torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
    def save(path,epoch):
        state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,best_epoch=best_epoch,best=best,bad=bad,
            mean=mean,std=std,history=history,spec=spec,config_sha256=sha(out/'config.json'),
            python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),test_accessed=False)
        tmp=path.with_suffix('.tmp.pt');torch.save(state,tmp);tmp.replace(path)
    if start==1:save(folder/'best.pt',0)
    tracker=None
    try:
        import wandb
        tracker=wandb.init(project='hssd-reliability-fusion',entity='aeonian-ulsan-national-institute-of-science-and-technology',
            name='ecn14-v1-'+name,dir=str(folder),config=dict(config,arm=spec),reinit=True,resume='allow',
            id=read(folder/'wandb.json')['id'] if (folder/'wandb.json').exists() else None)
        atomic_json(folder/'wandb.json',dict(id=tracker.id,url=tracker.url))
        epoch=start-1;last=evaluate(model,val,mean,std,spec)
        for epoch in range(start,config['epochs']+1):
            if bad>=config['patience']:break
            model.train();batches=[]
            for scene in train:
                ids=np.random.permutation(len(scene['arrays']['features']))
                batches.extend((scene,ids[i:i+config['batch_size']]) for i in range(0,len(ids),config['batch_size']))
            random.shuffle(batches);losses=[];total=sum(len(s['arrays']['features']) for s in train)
            for scene,ids in batches:
                a=scene['arrays'];bank=torch.tensor(np.asarray(a['scores'][ids]),dtype=torch.float32,device='cuda').flatten(2)
                x=torch.tensor((a['features'][ids]-mean)/std,dtype=torch.float32,device='cuda');w=model(x)
                ins=mask(a['xy'],a['truth'][ids],spec['margin_kind']=='joint') if spec['margin_kind'] else None
                losses_per_query=objective(w,bank[:,0],bank[:,1],torch.tensor(a['target_expectations'][ids],dtype=torch.float32,device='cuda'),
                    spec['temperature'],None if ins is None else torch.tensor(ins,device='cuda'),config['score_margin'])
                loss=losses_per_query.mean();balanced=loss*total/(len(train)*len(a['features']))
                optimizer.zero_grad(set_to_none=True);balanced.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
                losses.append(float(loss.detach()))
            last=evaluate(model,val,mean,std,spec)
            if selection_key(last)>selection_key(best):best=last;best_epoch=epoch;bad=0;save(folder/'best.pt',epoch)
            else:bad+=1
            row=dict(epoch=epoch,train_objective=float(np.mean(losses)),val=last);history.append(row)
            save(folder/'last.pt',epoch);atomic_json(folder/'history.json',history)
            set_status(out,'training',arm=name,epoch=epoch,best_epoch=best_epoch,val=last,test_accessed=False)
            tracker.log(dict(epoch=epoch,train_objective=row['train_objective'],val_loss=last['loss'],val_R1=last['micro']['recall_1m'],
                val_joint=last['micro']['recall_1m_30deg'],weight=last['weight_mean']))
            if epoch==1 or epoch%5==0:
                from .train import figure
                import matplotlib.pyplot as plt
                fig=figure(model,val,mean,std,torch.device('cuda'))
                fig.savefig(folder/f'likelihood_{epoch:03d}.png',dpi=120)
                tracker.log(dict(epoch=epoch,val_likelihood=wandb.Image(fig)));plt.close(fig)
            print('fusion_train',name,epoch,last,flush=True)
        result=dict(name=name,best_epoch=best_epoch,last_epoch=history[-1]['epoch'] if history else 0,best_val=best,
            last_val=last,best_sha256=sha(folder/'best.pt'),last_sha256=sha(folder/'last.pt'),spec=spec,test_accessed=False)
        atomic_json(folder/'complete.json',result);return result
    finally:
        if tracker:tracker.finish()


def test_report(root,vision,cache,report,config,entries,selection):
    out=root/REL;scenes=load(cache,entries);winner=selection['selected_arm'];policies=['vision_only','acoustic_only','equal_weight_fusion','visual_topk_acoustic_rerank','train_fixed','adaptive_best','adaptive_last']
    policies += ['control_'+n for n in config['runs'] if n!=winner]
    models={}
    for policy,name,which in [('adaptive_best',winner,'best'),('adaptive_last',winner,'last')]+[('control_'+n,n,'best') for n in config['runs'] if n!=winner]:
        state=torch.load(out/'runs'/name/(which+'.pt'),map_location='cpu',weights_only=False)
        model=Gate(14,state['spec']['constant']).cuda().eval();model.load_state_dict(state['model']);models[policy]=(model,state)
    records=[];report.mkdir(parents=True,exist_ok=True);(report/'heatmaps').mkdir(exist_ok=True)
    baseline_report=vision/'AcousticContrastive/pilots/scene_subset_v1/evaluation/report/input-ablation-dense/all'
    original={(r['scene_id'],r['sample_index']):r for r in read(baseline_report/'records.json')}
    for scene in scenes:
        a=scene['arrays'];queries=read(scene['folder']/'queries.json');ws={}
        for policy,(model,state) in models.items():
            with torch.no_grad():ws[policy]=model(torch.tensor((a['features']-state['mean'])/state['std'],dtype=torch.float32,device='cuda')).cpu().numpy()
        for i,q in enumerate(queries):
            v,ac=a['scores'][i];gt=a['truth'][i];row=copy.deepcopy(original[(scene['scene_id'],q['sample_index'])])
            choices=diagnose(v,ac,a['xy'],gt,50)['methods']
            for method in choices:
                np.testing.assert_allclose(choices[method]['pose_xzyaw'],row['methods'][method]['pose_xzyaw'],atol=1e-7,rtol=0)
            row['methods']=choices;weights={'train_fixed':selection['train_fixed_weight'],**{k:float(w[i]) for k,w in ws.items()}}
            for name,w in weights.items():
                cell,yaw=np.unravel_index(((1-w)*v+w*ac).argmax(),v.shape);pose=[*a['xy'][cell].tolist(),int(yaw)*10]
                choices[name]=dict(pose_xzyaw=pose,xy_error_m=float(np.linalg.norm(np.array(pose[:2])-gt[:2])),
                    yaw_error_deg=float(abs((pose[2]-gt[2]+180)%360-180)))
            row['weights']=weights
            # Four baseline heatmaps plus fixed/best/last; all other controls in numeric records.
            top=select(v,ac,50)[1];sparse=np.full(len(a['xy']),np.nan);sparse[top]=ac.max(1)[top]
            layers=[v.max(1),ac.max(1),(.5*v+.5*ac).max(1),sparse]
            for name in ('train_fixed','adaptive_best','adaptive_last'):
                w=weights[name];layers.append(((1-w)*v+w*ac).max(1))
            filename=scene['scene_id']+'_'+str(q['sample_index'])+'.bin'
            np.stack([heat_values(x) for x in layers]).tofile(report/'heatmaps'/filename);row['heatmap']='heatmaps/'+filename
            records.append(row)
    results={}
    for method in policies:
        rr=[r['methods'][method]['pose_xzyaw'] for r in records];gt=[r['gt_pose_xzyaw'] for r in records];vv=[r['methods']['vision_only']['pose_xzyaw'] for r in records]
        m=summarize(rr,gt,vv);m['per_scene']={s:summarize([r['methods'][method]['pose_xzyaw'] for r in records if r['scene_id']==s],
            [r['gt_pose_xzyaw'] for r in records if r['scene_id']==s],[r['methods']['vision_only']['pose_xzyaw'] for r in records if r['scene_id']==s]) for s in config['selection']['test']}
        results[method]=m
    # Recompute recall directly from saved errors as an independent serialization check.
    for name,m in results.items():
        errors=np.array([r['methods'][name]['xy_error_m'] for r in records])
        if not np.isclose(m['recall_1m'],(errors<=1).mean()):raise ValueError('Metric reproduction failed')
    summary=dict(status='complete',methods=results,selection=selection,queries=len(records),feature_names=[FEATURE_NAMES[i] for i in config['feature_columns']],
        test_fitting=False,notice=config['test_notice'],baseline_reproduced_queries=len(records))
    atomic_json(report/'summary.json',summary);atomic_json(report/'records.json',records)
    for name in ('banks.json',):shutil.copy2(baseline_report/name,report/name)
    fields=['recall_0.1m','recall_0.5m','recall_1m','recall_1m_30deg','yaw_within_10deg','yaw_within_30deg','yaw_within_50deg','median_xy_m','median_yaw_deg','repair','harm']
    with (report/'metrics.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['method',*fields])
        for name,m in results.items():writer.writerow([name,*[m[k] for k in fields]])
    atomic_json(out/'evaluation_complete.json',dict(status='complete',summary_sha256=sha(report/'summary.json'),records_sha256=sha(report/'records.json'),test_fitting=False))


def run(root,vision,report,cache):
    root,vision,report,cache=map(Path,(root,vision,report,cache));out=root/REL;config=read(out/'config.json')
    lock=(out/'pipeline.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cache.mkdir(parents=True,exist_ok=True);report.mkdir(parents=True,exist_ok=True)
    shutil.copy2(Path(__file__).with_name('revision_report.html'),report/'index.html')
    try:
        audit=read(root/'ReliabilityFusion/preflight/desdf_range_v1.json')
        if audit['status']!='passed' or audit['split_sha256']!=config['split_sha256']:
            raise ValueError('DESDF range preflight failed')
        range_checks=[]
        for split in ('train','val'):
            for scene in config['selection'][split]:
                relative=Path('UnLoc/structural_targets')/scene/'desdf.npy'
                if sha(root/relative)!=sha(vision/relative):raise ValueError('DESDF replica mismatch')
                payload=np.load(vision/relative,allow_pickle=True).item()
                if payload['max_range_m']!=10 or payload['desdf_spacing_m']!=.1 or payload['desdf'].shape[-1]!=36:
                    raise ValueError('DESDF cap/grid/yaw changed')
                range_checks.append(dict(scene_id=scene,split=split,radial_cap_m=10,grid_m=.1,yaw_deg=10,sha256=sha(vision/relative)))
        immutable_json(out/'range_preflight.json',dict(status='passed',source_audit_sha256=sha(root/'ReliabilityFusion/preflight/desdf_range_v1.json'),
            checks=range_checks,test_accessed=False))
        while not (out/'envelope_ready.json').exists():
            current=read(out/'preprocess_status.json') if (out/'preprocess_status.json').exists() else {}
            if current.get('status')=='failed':raise RuntimeError('CPU preprocessing failed: '+str(current))
            set_status(out,'waiting_for_verified_envelope',preprocessing=current,test_accessed=False)
            atomic_json(report/'status.json',read(out/'status.json'));time.sleep(20)
        if sha(root/config['acoustic_checkpoint'])!=config['acoustic_sha256']:raise ValueError('Frozen acoustic changed')
        if sha(root/MODEL/'configs/all.json')!=config['acoustic_config_sha256']:raise ValueError('Acoustic input configuration changed')
        if sha(root/OLD/'scores_ready.json')!=config['old_scores_sha256']:raise ValueError('Old train/val score index changed')
        if sha(root/'manifests/scene_splits_v1.json')!=config['split_sha256']:raise ValueError('Splits changed')
        torch.set_num_threads(4)
        entries=build_scores(root,cache,config);scenes=load(cache,entries)
        curves=curve(root,scenes,config);fixed,diagnosis=fixed_and_diagnosis(scenes,curves,config)
        atomic_json(out/'diagnosis.json',dict(train_fixed_weight=fixed,diagnosis=diagnosis,test_accessed=False))
        atomic_json(report/'diagnosis.json',read(out/'diagnosis.json'))
        x=np.concatenate([s['arrays']['features'] for s in scenes if s['split']=='train']);mean=x.mean(0);std=x.std(0);std=np.where(std>1e-6,std,1)
        results=[train_arm(root,scenes,config,fixed,name,mean,std) for name in config['runs']]
        # Loss magnitudes are incomparable across NLL temperatures/margin arms.
        winner=max(results,key=lambda r:(r['best_val']['scene_macro_recall_1m'],
            r['best_val']['scene_macro_joint'],-config['runs'].index(r['name'])))
        selection=dict(config_sha256=sha(out/'config.json'),selected_arm=winner['name'],train_fixed_weight=fixed,
            results=results,test_accessed=False,selected_before_test=True,notice=config['test_notice'],
            across_arm_ties='scene-macro R1, then joint, then preregistered run order; losses not compared across objectives')
        immutable_json(out/'selection_frozen.json',selection);atomic_json(report/'selection.json',selection)
        test_entries=build_scores(root,cache,config,test=True)
        test_report(root,vision,cache,report,config,test_entries,selection)
        atomic_json(out/'status.json',dict(status='complete',stage='test_report',report=str(report),updated_unix=time.time()))
        atomic_json(report/'status.json',read(out/'status.json'))
    except Exception:
        atomic_json(out/'status.json',dict(status='failed',error=traceback.format_exc(),updated_unix=time.time()))
        atomic_json(report/'status.json',read(out/'status.json'));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','vision','report','cache'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();run(a.root,a.vision,a.report,a.cache)

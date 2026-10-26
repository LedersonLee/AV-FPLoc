"""Canonical dense Val verification and matched-input localization report."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import csv
from pathlib import Path
import shutil
import numpy as np

from acoustic_branch.contracts import atomic_json,sha
from .input_supervision_audit import read,resolve,SOURCE,pose,yawdiff
from .response_input_localization import REL,ARMS,load_scene,metrics
from .weight_group_discovery import mask_of

LABELS={'core14':'기존 신뢰도14개','processed58':'기존14 + 가공44 = 58개','core14_response15':'기존14 + 새 MAP 반응15 = 29개','processed58_response15':'기존58 + 새 MAP 반응15 = 73개'}


def verify_scene(task):
    root,ssd,sid,method_names,ws,expected=task;root,ssd=map(Path,(root,ssd));out=root/REL
    s=load_scene(out,sid,root,ssd);bank=np.load(s['source']/'scores.npy',mmap_mode='r');n=len(ws);direct=np.zeros_like(expected);checks=0;unique=0
    for i in range(n):
        v,a=bank[i];values,inv=np.unique(ws[i],return_inverse=True);pp=[]
        for w in values:pp.append(pose(int(np.argmax((1-w)*v+w*a)),s['xy']))
        direct[i]=np.array(pp)[inv];checks+=len(method_names);unique+=len(values)
    np.testing.assert_allclose(direct,expected,atol=1e-8,rtol=0)
    print('DENSE VERIFIED',sid,checks,'policies /',unique,'unique',flush=True)
    return dict(scene_id=sid,queries=n,policy_pose_checks=checks,unique_dense_argmax=unique)


def independent_metrics(pred,truth,fixed):
    dx=pred[:,0]-truth[:,0];dz=pred[:,1]-truth[:,1];distance=np.sqrt(dx*dx+dz*dz)
    angle=np.minimum(np.remainder(pred[:,2]-truth[:,2],360),np.remainder(truth[:,2]-pred[:,2],360))
    fd=np.sqrt(np.sum((fixed[:,:2]-truth[:,:2])**2,axis=1));fa=np.minimum(np.remainder(fixed[:,2]-truth[:,2],360),np.remainder(truth[:,2]-fixed[:,2],360))
    d={f'recall_{r:g}m':float(np.count_nonzero(distance<=r)/len(distance))for r in(.1,.5,1)}
    d.update({f'yaw_within_{r}deg':float(np.count_nonzero(angle<=r)/len(angle))for r in(10,30,50)})
    d.update(recall_1m_30deg=float(np.mean((distance<=1)&(angle<=30))),median_xy_m=float(np.median(distance)),median_yaw_deg=float(np.median(angle)))
    for name,before,after in [('xy',fd<=1,distance<=1),('joint',(fd<=1)&(fa<=30),(distance<=1)&(angle<=30))]:
        d['repair_vs_fixed_'+name]=int(np.sum(~before&after));d['harm_vs_fixed_'+name]=int(np.sum(before&~after))
    return d


def main(args):
    root,ssd,report=args.root,args.ssd,args.report;out=root/REL;report.mkdir(parents=True,exist_ok=True)
    config=read(out/'config.json');ready=read(out/'ready.json');refs=sorted(ready['scenes'],key=lambda r:r['scene_id']);scenes={r['scene_id']:load_scene(out,r['scene_id'],root,ssd)for r in refs}
    assert len({s for s in config['selection']['train']})==12 and len(set(config['selection']['val']))==3
    runs=[read(p)for p in sorted((out/'runs').glob('*/complete.json'))];assert len(runs)==16
    reference_run=next(r for r in runs if r['arm']=='processed58'and r['objective']=='logit_regression'and r['seed']==42)
    matched_mean=float(reference_run['evaluated']['best']['train']['weight_mean'])
    allmethods={};methodmeta={};perquery={};global_methods=[]
    for split in('train','val'):
        ss=[s for s in scenes.values()if s['split']==split];truth=np.concatenate([s['truth']for s in ss]);fixed=np.concatenate([s['grid_poses'][:,2]for s in ss]);cohorts=[]
        # The candidate grids already contain independently verified controls.
        policies={name:dict(poses=np.concatenate([s['grid_poses'][:,j]for s in ss]),weights=np.full(len(truth),w))for name,j,w in [('vision',0,0),('acoustic',20,1),('equal',10,.5),('fixed01',2,.1)]}
        from .response_input_localization import predict
        policies['matched_mean58']=dict(poses=np.concatenate([predict(s,np.full(len(s['truth']),matched_mean))for s in ss]),weights=np.full(len(truth),matched_mean))
        for r in runs:
            stem=f"{r['arm']}__{r['objective']}__seed{r['seed']}"
            for phase in('best','last'):
                name=stem+'__'+phase;ev=r['evaluated'][phase][split];ps=ev['predictions']
                policies[name]=dict(poses=np.concatenate([ps[s['scene_id']]['poses']for s in ss]),weights=np.concatenate([ps[s['scene_id']]['weights']for s in ss]))
                methodmeta[name]=dict(arm=r['arm'],label=LABELS[r['arm']],objective=r['objective'],seed=r['seed'],phase=phase,epoch=r['evaluated'][phase]['epoch'])
                assert len(policies[name]['poses'])==len(truth)
        names=list(policies);global_methods=names
        for name,m in policies.items():
            m['metrics']=metrics(m['poses'],truth,fixed);independent=independent_metrics(m['poses'],truth,fixed)
            for key,value in independent.items():np.testing.assert_allclose(value,m['metrics'][key],atol=1e-12,rtol=0)
        allmethods[split]={};perquery[split]=[]
        offset=0
        for s in ss:
            n=len(s['truth']);sl=slice(offset,offset+n)
            for i,q in enumerate(s['queries']):
                perquery[split].append(dict(scene_id=s['scene_id'],sample_index=q['sample_index'],pose_id=q['pose_id'],image_path=q['image_path'],truth=s['truth'][i].tolist(),success_intervals=q['success']['joint'],mask=mask_of(q['success']['joint']),methods={name:dict(pose=m['poses'][offset+i].tolist(),weight=float(m['weights'][offset+i]))for name,m in policies.items()}))
            for name,m in policies.items():
                entry=allmethods[split].setdefault(name,dict(micro=m['metrics'],per_scene={},weight_mean=float(m['weights'].mean()),weight_p10=float(np.quantile(m['weights'],.1)),weight_p90=float(np.quantile(m['weights'],.9))))
                entry['per_scene'][s['scene_id']]=metrics(m['poses'][sl],truth[sl],fixed[sl])
            offset+=n
        for entry in allmethods[split].values():entry['macro_R1']=float(np.mean([m['recall_1m']for m in entry['per_scene'].values()]))
    valrows=perquery['val'];assert len(valrows)==1704 and len(perquery['train'])==6387
    old=read(root/SOURCE/'evaluation.json')
    for name,oldname in [('vision','vision_only'),('acoustic','acoustic_only'),('equal','equal_fusion'),('fixed01','fixed_01')]:
        for k,v in allmethods['val'][name]['micro'].items():np.testing.assert_allclose(v,old['methods'][oldname][k],atol=1e-12,rtol=0)
    verify_path=out/'dense_verification.json'
    if verify_path.exists()and read(verify_path).get('method_names')==global_methods:
        verify=read(verify_path);assert verify['training_sha256']==sha(out/'training_complete.json') and verify['run_hashes']=={p.parent.name:sha(p)for p in sorted((out/'runs').glob('*/complete.json'))}
    else:
        tasks=[]
        for sid in config['selection']['val']:
            rr=[r for r in valrows if r['scene_id']==sid];tasks.append((str(root),str(ssd),sid,global_methods,np.array([[r['methods'][m]['weight']for m in global_methods]for r in rr]),np.array([[r['methods'][m]['pose']for m in global_methods]for r in rr])))
        with ProcessPoolExecutor(max_workers=3)as pool:receipts=[f.result()for f in as_completed([pool.submit(verify_scene,t)for t in tasks])]
        verify=dict(status='passed',receipts=receipts,method_names=global_methods,training_sha256=sha(out/'training_complete.json'),run_hashes={p.parent.name:sha(p)for p in sorted((out/'runs').glob('*/complete.json'))},test_accessed=False)
        atomic_json(verify_path,verify)
    # Paired scene-cluster intervals of selected checkpoints are descriptive:
    # the same Val already selected epochs; they are not independent tests.
    rng=np.random.default_rng(20261021);ci={};sids=config['selection']['val'];draw=rng.integers(3,size=(2000,3))
    fixed=allmethods['val']['fixed01']['per_scene']
    for name,entry in allmethods['val'].items():
        delta=[]
        for sid in sids:
            m=entry['per_scene'][sid];base=fixed[sid];delta.append([m['queries']*(m[k]-base[k])for k in('recall_1m','recall_1m_30deg')])
        n=np.array([fixed[s]['queries']for s in sids]);delta=np.array(delta);samples=delta[draw].sum(1)/n[draw].sum(1)[:,None]
        ci[name]=np.quantile(samples,[.025,.975],axis=0).T.tolist()
    feature_families=[dict(name='기존 신뢰도14',count=14,content='Vision 5 + Acoustic 5: peak gap·공간 분산·주변 질량·XY entropy·yaw entropy; 상호4: MAP 거리·yaw 차이·XY 분포 겹침·top50 겹침',rationale='개별 모달리티의 집중도와 두 예측의 합의/불일치'),
        dict(name='예측 depth4',count=4,content='평균·CV·범위·인접 ray 변화량',rationale='현재 RGB에서 추정한 구조적 depth가 식별력 있는 형태인지'),
        dict(name='교차 후보 지지14',count=14,content='상대 MAP의 순위·점수 손실·주변 질량·공간 모드; 두 XY 점수 상관·상위1% 겹침',rationale='한 모달리티가 상대 후보를 지지하거나 배척하는 방식'),
        dict(name='기존 MAP 반응14',count=14,content='w=.05/.15/.2/.3/.5에서 고정.1 MAP 대비 거리·yaw 변화10 + 동일 후보 유지 폭/좌우 여유/전환 수4',rationale='가중치를 바꿀 때 결합 argmax가 얼마나 민감한지'),
        dict(name='후보 ray 일치12',count=12,content='Vision/Acoustic/고정.1 후보 각4: RGB 예측 ray 대비 MAE·중앙 AE·0.5m 이내 비율·상대 MAE',rationale='후보 도면 구조가 현재 RGB 예측 구조와 실제로 맞는지; GT depth 아님'),
        dict(name='새 주변영역 MAP 반응15',count=15,content='XY/Joint 주변영역의 좌우 유지폭·전체 지지폭·재진입8; 큰 XY/yaw/Joint 전환·작은 전환4; 모드 수·최대 비중·entropy3',rationale='0.2m/10° 격자의 작은 argmax 흔들림과 다른 위치·방향으로의 전환을 구별; 고정.1 예측 MAP 기준, GT 기준 아님')]
    # All epochs/runs remain public; default is seed42, not a cherry-picked seed.
    selected='processed58__logit_regression__seed42__best';controls=['vision','acoustic','fixed01',selected]
    gallery=[];chosen=[]
    for criterion in('xy','joint'):
        for category in('repair','harm','both_success','both_failure'):
            candidates=[]
            for i,r in enumerate(valrows):
                def ok(name):
                    p=np.array(r['methods'][name]['pose']);gt=np.array(r['truth']);return np.linalg.norm(p[:2]-gt[:2])<=1 and(criterion=='xy'or yawdiff(p[2],gt[2])<=30)
                before,after=ok('fixed01'),ok(selected);cat='both_success'if before and after else 'repair'if after else 'harm'if before else 'both_failure'
                if cat==category:candidates.append(i)
            seen=set();take=[]
            for i in candidates:
                r=valrows[i];key=(r['scene_id'],r['pose_id'])
                if key in seen:continue
                seen.add(key);take.append(i)
                if len(take)>=3:break
            for i in take:
                if i not in chosen:chosen.append(i)
    for d in('maps','rgb','heatmaps'):(report/d).mkdir(exist_ok=True)
    banks={}
    for i in chosen:
        r=valrows[i];sid=r['scene_id'];s=scenes[sid];qi=next(j for j,q in enumerate(s['queries'])if q['sample_index']==r['sample_index']);v,a=np.load(s['source']/'scores.npy',mmap_mode='r')[qi]
        name=sid+'_'+str(r['sample_index']);layers=np.array([((1-r['methods'][m]['weight'])*v+r['methods'][m]['weight']*a).max(1)for m in controls],dtype='<f4');layers.tofile(report/'heatmaps'/(name+'.bin'))
        shutil.copy2(ssd/r['image_path'],report/'rgb'/(name+'.png'));shutil.copy2(root/'2DFP'/sid/'map.png',report/'maps'/(sid+'.png'))
        meta=read(root/'2DFP'/sid/'floorplan_metadata.json');banks[sid]=dict(xy=s['xy'].tolist(),metadata=meta)
        gallery.append(dict(**r,image='rgb/'+name+'.png',heatmap='heatmaps/'+name+'.bin'))
    explanations=dict(none='927/6387: only frozen score fields and scalar fusion, candidate resolution .2m/10deg, success XY1m/yaw30deg; NOT intrinsic impossibility',fixed01_success=4493,fixed01_lm=917,fixed01_lmh=3576,lmh_endpoint_cases=dict(both_success=740,vision_only=1580,acoustic_only=526,both_failure=730),h_only_median_success_width=.225)
    response_diagnosis={}
    for row in valrows:
        def ok(method):
            p=np.array(row['methods'][method]['pose']);t=np.array(row['truth']);return np.linalg.norm(p[:2]-t[:2])<=1 and yawdiff(p[2],t[2])<=30
        before,after=ok('fixed01'),ok(selected)
        category='both_success'if before and after else'repair'if after else'harm'if before else'both_failure'
        radius=max([min(.1-lo,hi-.1)for lo,hi in row['success_intervals']if lo<=.1<=hi]or[0.])
        response_diagnosis.setdefault(category,[]).append(dict(weight=row['methods'][selected]['weight'],mask=row['mask'],safe_radius_at01=radius))
    response_diagnosis={c:dict(queries=len(rr),median_weight=float(np.median([r['weight']for r in rr])),weights_below01=sum(r['weight']<.1 for r in rr),median_GT_safe_radius=float(np.median([r['safe_radius_at01']for r in rr])),mask_counts={str(m):sum(r['mask']==m for r in rr)for m in range(8)})for c,rr in response_diagnosis.items()}
    result=dict(config=config,methods=allmethods,metadata=methodmeta,labels=LABELS,runs=[{k:v for k,v in r.items()if k not in('evaluated','history')}for r in runs],method_order=global_methods,
        feature_families=feature_families,feature_names=refs[0]['features'],arm_columns=ARMS,diagnostic_explanations=explanations,val_scene_bootstrap_delta_ci95=ci,
        selected_gallery_method=selected,gallery_layer_methods=controls,response_diagnosis=response_diagnosis,matched_mean_control=dict(weight=matched_mean,source='Mean TRAIN prediction of processed58/logit/seed42/Val-best. Same checkpoint chosen by Val, so this is a conditional matched-mean control, NOT an independently Train-selected optimum.'),notice='Repeated exploratory Val3 development, Val-selects epochs; not independent Test improvement. Frozen branch Train predictions in-sample. Input count is the only intended arm difference; CPU controls refit concurrently, not asserted bitwise equal to old GPU Last.',test_accessed=False)
    atomic_json(out/'evaluation.json',result);atomic_json(report/'summary.json',result);atomic_json(report/'records.json',valrows);atomic_json(report/'gallery.json',gallery);atomic_json(report/'banks.json',banks)
    atomic_json(report/'histories.json',{f"{r['arm']}__{r['objective']}__seed{r['seed']}":r['history']for r in runs})
    atomic_json(out/'train_predictions.json',perquery['train'])
    fields=list(allmethods['val']['fixed01']['micro'])
    with(report/'metrics.csv').open('w')as f:
        writer=csv.writer(f);writer.writerow(['split','method',*fields])
        for split,methods_ in allmethods.items():
            for name,m in methods_.items():writer.writerow([split,name,*[m['micro'][k]for k in fields]])
    verification=dict(status='passed',train_queries=6387,val_queries=1704,train_scenes=12,val_scenes=3,runs=16,input_variants=4,seeds=[42,43],epochs=40,
        canonical_dense_pose_checks=sum(r['policy_pose_checks']for r in verify['receipts']),independent_metrics=True,baseline_reproduced=True,test_accessed=False,
        config_sha256=sha(out/'config.json'),evaluation_sha256=sha(out/'evaluation.json'),source_code_sha256=sha(Path(__file__).with_name('response_input_localization.py')))
    atomic_json(out/'verification.json',verification);atomic_json(report/'verification.json',verification)
    shutil.copy2(Path(__file__).with_name('response_input_report.html'),report/'index.html');atomic_json(out/'status.json',dict(stage='complete',test_accessed=False));print('REPORT COMPLETE',report,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--ssd',type=Path,required=True);p.add_argument('--report',type=Path,required=True);main(p.parse_args())

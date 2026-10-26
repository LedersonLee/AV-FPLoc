"""Explain frozen fusion rows and add explicitly GT-dependent oracle diagnostics.

Never retrain, tune on test, rewrite original numeric results, or replace model
selection. Supplemental Last checkpoints and test oracles are post-hoc diagnostics.
"""
import argparse
import csv
from pathlib import Path
import shutil
import time

import numpy as np

from acoustic_branch.contracts import atomic_json,immutable_json,sha
from acoustic_branch.data import read
from .revision_data import REL

FIELDS=['recall_0.1m','recall_0.5m','recall_1m','recall_1m_30deg',
    'yaw_within_10deg','yaw_within_30deg','yaw_within_50deg','median_xy_m','median_yaw_deg','repair','harm']
ARM_LABELS=dict(adaptive_nll_T1_half='Query별 가중치 · NLL T=1 · 초기 w=0.5',
    adaptive_nll_T1='Query별 가중치 · NLL T=1 · 초기 w=0.1',
    adaptive_nll_T05='Query별 가중치 · NLL T=0.5 · 초기 w=0.1',
    adaptive_nll_T025='Query별 가중치 · NLL T=0.25 · 초기 w=0.1',
    adaptive_margin_xy='Query별 가중치 · XY margin · 초기 w=0.1',
    adaptive_margin_joint='Query별 가중치 · XY·yaw margin · 초기 w=0.1',
    constant_nll_T1='전체 query 공통 상수 · NLL T=1 · 초기 w=0.1')
ORACLES=dict(oracle_xy='GT 성공 oracle · 위치 1m 목표',oracle_joint='GT 성공 oracle · 위치 1m·yaw30° 목표',
    oracle_nll_T025='GT NLL oracle · T=0.25',oracle_nll_T05='GT NLL oracle · T=0.5',oracle_nll_T1='GT NLL oracle · T=1')


def metric(poses,truth,vision):
    poses,truth,vision=map(np.asarray,(poses,truth,vision))
    d=np.linalg.norm(poses[:,:2]-truth[:,:2],axis=1);y=abs((poses[:,2]-truth[:,2]+180)%360-180)
    v=np.linalg.norm(vision[:,:2]-truth[:,:2],axis=1)<=1
    values={f'recall_{t:g}m':float((d<=t).mean()) for t in (.1,.5,1)}
    values.update({f'yaw_within_{t}deg':float((y<=t).mean()) for t in (10,30,50)})
    values.update(recall_1m_30deg=float(((d<=1)&(y<=30)).mean()),median_xy_m=float(np.median(d)),
        median_yaw_deg=float(np.median(y)),repair=int((~v&(d<=1)).sum()),harm=int((v&(d>1)).sum()),queries=len(d))
    return values


def select_oracles(poses,nll,truth,weights,temperatures):
    """One w per query; GT-dependent, stable tie nearest .5. No model fitting."""
    poses,truth,weights=map(np.asarray,(poses,truth,weights))
    d=np.linalg.norm(poses[:,:,:2]-truth[:,None,:2],axis=2)
    y=abs((poses[:,:,2]-truth[:,None,2]+180)%360-180)
    values=dict(oracle_xy=(d<=1).astype(float),oracle_joint=((d<=1)&(y<=30)).astype(float))
    for name,t in [('oracle_nll_T025',.25),('oracle_nll_T05',.5),('oracle_nll_T1',1.)]:
        values[name]=-nll[:,list(temperatures).index(t)]
    order=np.argsort(abs(weights-.5),kind='stable')
    return {name:order[value[:,order].argmax(1)] for name,value in values.items()}


def descriptors(selection):
    rows={r['name']:r for r in selection['results']};winner=rows[selection['selected_arm']]
    common='각 query에 동일한 상수 w를 사용합니다.' if winner['spec']['constant'] else '현재 query의 14개 특징으로 w를 출력합니다.'
    w=selection['train_fixed_weight']
    out={
        'vision_only':dict(label='Vision-only',kind='baseline',description='UnLoc epoch20 시각 점수 지도만으로 pose 선택. Acoustic w=0.'),
        'acoustic_only':dict(label='Acoustic-only · E+C+N',kind='baseline',description='고정된 subset E+C+N 음향 점수 지도만으로 pose 선택. Acoustic w=1.'),
        'equal_weight_fusion':dict(label='고정 1:1 fusion · w=0.5',kind='baseline',description='각 점수 지도를 query별 z-score로 정규화하고 0.5:0.5로 결합. 학습하지 않은 상수.'),
        'visual_topk_acoustic_rerank':dict(label='Visual top-50 음향 재정렬 · 참고',kind='baseline',description='시각 top-50 위치를 음향 점수로 재정렬하고 선택 위치의 시각 yaw를 사용. 공식 제안 fusion이 아닌 참고 조건.'),
        'train_fixed':dict(label=f'Train 선택 고정 가중치 · w={w:.2f}',kind='constant',description='Train scene-macro R@1m으로 0:0.05:1 격자에서 선택한 전역 상수. 모든 query에 동일하며 test GT를 사용하지 않음.')}
    for key,which in [('adaptive_best','best'),('adaptive_last','last')]:
        r=winner;weight=r[which+'_val']['weight_mean'];ep=r[which+'_epoch']
        name='학습 공통 상수' if r['spec']['constant'] else 'Query별 adaptive 가중치'
        out[key]=dict(label=f'{name} · {which.title()} ep{ep} · w'+('='+f'{weight:.4f}' if r['spec']['constant'] else '는 query별'),
            kind='constant' if r['spec']['constant'] else 'adaptive',arm=r['name'],checkpoint=which,epoch=ep,
            description=f'Val에서 선택된 조건 {r["name"]}의 {which.title()} checkpoint. '+common+
                (' Best는 val scene-macro R1, joint 순으로 선택.' if which=='best' else ' Last는 조기 종료 시점이며 성능이 가장 좋다는 의미가 아님.'))
    for name,r in rows.items():
        if name==winner['name']:continue
        ep=r['best_epoch'];initial=' · 학습 전 초기값' if ep==0 else ''
        out['control_'+name]=dict(label=ARM_LABELS[name]+f' · Best ep{ep}'+initial,kind='adaptive',arm=name,checkpoint='best',epoch=ep,
            description=ARM_LABELS[name]+'. '+('Best가 epoch0이므로 학습된 adaptive 효과가 아니라 초기 상수 예측을 표시. Last는 아래 조건별 표에서 별도로 확인.' if ep==0 else 'Val에서 선택된 query별 adaptive checkpoint.'))
    return out


def get_curve(folder,out,scene,weights,temperatures,status):
    import torch
    dest=out/'test_curves'/(scene+'.npz');dest.parent.mkdir(parents=True,exist_ok=True)
    source_sha=sha(folder/'receipt.json')
    if dest.exists():
        with np.load(dest) as blob:
            if str(blob['source_sha'])!=source_sha:raise ValueError('Oracle curve source changed')
            return blob['poses'].copy(),blob['nll'].copy()
    scores=np.load(folder/'scores.npy',mmap_mode='r');xy=np.load(folder/'xy.npy');expected=np.load(folder/'target_expectations.npy')
    poses=[];nll=[];w=torch.tensor(weights,dtype=torch.float64,device='cuda')[:,None,None]
    for i in range(len(scores)):
        v,a=scores[i];s=(1-w)*torch.tensor(v,device='cuda')+w*torch.tensor(a,device='cuda')
        indices=s.flatten(1).argmax(1).cpu().numpy();poses.append(np.column_stack([xy[indices//36],indices%36*10]))
        exp=(1-weights)*expected[i,0]+weights*expected[i,1]
        nll.append(np.stack([torch.logsumexp(s.flatten(1)/t,dim=1).cpu().numpy()-exp/t for t in temperatures]))
        if i%100==0:status(scene_id=scene,query=i,total=len(scores))
    scores._mmap.close();poses,nll=np.asarray(poses),np.asarray(nll)
    np.savez_compressed(dest,poses=poses,nll=nll,source_sha=np.array(source_sha));return poses,nll


def run(root,cache,report):
    import torch
    from .model import Gate
    root,cache,report=map(Path,(root,cache,report));base=root/REL;out=base/'review_v2';out.mkdir(parents=True,exist_ok=True)
    config=read(base/'config.json');selection=read(base/'selection_frozen.json');summary=read(report/'summary.json');records=read(report/'records.json')
    completed=read(base/'evaluation_complete.json')
    if sha(report/'summary.json')!=completed['summary_sha256'] or sha(report/'records.json')!=completed['records_sha256']:
        raise ValueError('Original evaluation changed')
    contract=dict(original_summary_sha256=sha(report/'summary.json'),original_records_sha256=sha(report/'records.json'),
        selection_sha256=sha(base/'selection_frozen.json'),config_sha256=sha(base/'config.json'),
        no_retraining=True,no_deployable_selection_changed=True,oracle_uses_GT=True,
        test_accessed_for_diagnostics=True,test_used_for_deployable_fitting=False,
        grid=config['weight_grid'],tie_rule='Equal objective: weight closest to .5, stable grid order',
        note='Success ceilings only for their target criterion and searched grid, NOT continuous-weight or all-metric upper bounds.')
    immutable_json(out/'contract.json',contract)
    def status(**extra):atomic_json(out/'status.json',dict(status='running',updated_unix=time.time(),**extra))
    torch.set_num_threads(4);weights=np.array(config['weight_grid']);temperatures=config['score_temperature_grid']
    all_oracles={};oracle_rows={};test_extra={r['name']:[] for r in selection['results']};sources={}
    lut={(r['scene_id'],r['sample_index']):r for r in records}
    for split in ('val','test'):
        ready=read(base/('scores_ready.json' if split=='val' else 'test_scores_ready.json'))
        selected=[e for e in ready['entries'] if e['split']==split]
        if {e['scene_id'] for e in selected}!=set(config['selection'][split]):raise ValueError('Oracle split mismatch')
        collected={name:[] for name in ORACLES};truths=[];visions=[];ids=[]
        for entry in selected:
            scene=entry['scene_id'];folder=cache/entry['cache_directory'];receipt=read(folder/'receipt.json')
            if sha(folder/'receipt.json')!=entry['receipt_sha256']:raise ValueError('Score receipt changed')
            for name,digest in receipt['files'].items():
                if sha(folder/name)!=digest:raise ValueError('Score payload changed')
            sources[scene]=entry['receipt_sha256'];truth=np.load(folder/'truth.npy');xy=np.load(folder/'xy.npy');queries=read(folder/'queries.json')
            if split=='val':
                with np.load(folder/'weight_curve.npz') as blob:poses,nll=blob['poses'].copy(),blob['nll'].copy()
            else:poses,nll=get_curve(folder,out,scene,weights,temperatures,status)
            choices=select_oracles(poses,nll,truth,weights,temperatures)
            if split=='test':
                for i,q in enumerate(queries):
                    old=lut[(scene,q['sample_index'])]
                    np.testing.assert_allclose(truth[i],old['gt_pose_xzyaw'],atol=1e-7,rtol=0)
                    for endpoint,method in [(0,'vision_only'),(-1,'acoustic_only')]:
                        np.testing.assert_allclose(poses[i,endpoint],old['methods'][method]['pose_xzyaw'],atol=1e-7,rtol=0)
                # Include real Last models, not just their epoch0 Best, without selecting on test.
                f=np.load(folder/'features.npy');scores=np.load(folder/'scores.npy',mmap_mode='r')
                for trained in selection['results']:
                    name=trained['name'];path=base/'runs'/name/'last.pt'
                    if sha(path)!=trained['last_sha256']:raise ValueError('Frozen Last checkpoint changed')
                    state=torch.load(path,map_location='cpu',weights_only=False);model=Gate(14,state['spec']['constant']).cuda().eval()
                    model.load_state_dict(state['model'])
                    with torch.no_grad():ws=model(torch.tensor((f-state['mean'])/state['std'],dtype=torch.float32,device='cuda')).cpu().numpy()
                    for i,wgt in enumerate(ws):
                        v,a=scores[i];win=((1-float(wgt))*v+float(wgt)*a).argmax();pose=[*xy[win//36].tolist(),int(win%36)*10]
                        test_extra[name].append(dict(scene_id=scene,sample_index=queries[i]['sample_index'],pose=pose,truth=truth[i].tolist(),weight=float(wgt),
                            vision=poses[i,0].tolist()))
                    del model
                scores._mmap.close()
            for name,indices in choices.items():
                for i,wi in enumerate(indices):collected[name].append(dict(pose=poses[i,wi].tolist(),weight=float(weights[wi])))
            truths.extend(truth.tolist());visions.extend(poses[:,0].tolist());ids.extend((scene,q['sample_index']) for q in queries)
        all_oracles[split]={};oracle_rows[split]=[]
        for name,rows in collected.items():
            if len(rows)!=(1704 if split=='val' else 1472):raise ValueError('Incomplete oracle query coverage')
            m=metric([r['pose'] for r in rows],truths,visions)
            m.update(label=ORACLES[name],mean_weight=float(np.mean([r['weight'] for r in rows])),uses_GT=True)
            all_oracles[split][name]=m
        for i,(scene,sample) in enumerate(ids):oracle_rows[split].append(dict(scene_id=scene,sample_index=sample,gt=truths[i],
            methods={name:collected[name][i] for name in ORACLES}))
    conditions=[]
    for r in selection['results']:
        name=r['name'];rows=test_extra[name];last=metric([x['pose'] for x in rows],[x['truth'] for x in rows],[x['vision'] for x in rows])
        best_key='adaptive_best' if name==selection['selected_arm'] else 'control_'+name
        conditions.append(dict(name=name,label=ARM_LABELS[name],constant=r['spec']['constant'],spec=r['spec'],best_epoch=r['best_epoch'],
            last_epoch=r['last_epoch'],best_val=r['best_val'],last_val=r['last_val'],test_best=summary['methods'][best_key],test_last=last,
            initial_weight=.5 if r['spec']['half_initialization'] else selection['train_fixed_weight'],
            test_last_weight_mean=float(np.mean([x['weight'] for x in rows])),
            test_last_weight_min=float(min(x['weight'] for x in rows)),test_last_weight_max=float(max(x['weight'] for x in rows))))
    review=dict(status='complete',contract=contract,policies=descriptors(selection),conditions=conditions,oracles=all_oracles,
        oracle_descriptions=dict(oracle_xy='GT를 보고 R@1m 성공 가능 가중치를 선택. 성공이 여러 개면 0.5에 가까운 값을 선택. 해당 격자의 R@1m ceiling이며 다른 지표의 ceiling은 아님.',
            oracle_joint='GT를 보고 위치1m·yaw30° 공동 성공 가능 가중치를 선택. 해당 격자의 공동 성공률 ceiling.',
            nll='GT Gaussian target(.2m/10°)에 대한 soft-pose NLL 최소 가중치를 선택. 성공률을 최대화하지 않으므로 성능 상한이 아님.'),
        source_receipts=sources,query_counts=dict(val=1704,test=1472),
        explanation='w_A(q)는 현재 query의 acoustic weight 한 개; w_V(q)=1−w_A(q). 같은 query의 모든 XY/yaw 후보에 동일하게 적용. 위치·yaw 별도 weight나 후보별 weight는 아님.')
    atomic_json(out/'review.json',review);atomic_json(out/'oracle_records.json',oracle_rows);atomic_json(out/'last_records.json',test_extra)
    atomic_json(report/'review.json',review);atomic_json(report/'oracle_records.json',oracle_rows)
    with (report/'metrics_explained.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['split','kind','label','policy_id',*FIELDS])
        for name,m in summary['methods'].items():writer.writerow(['test','deployable',review['policies'][name]['label'],name,*[m[k] for k in FIELDS]])
        for c in conditions:writer.writerow(['test','frozen_control_last',c['label']+' · Last',c['name']+'_last',*[c['test_last'][k] for k in FIELDS]])
        for split,values in all_oracles.items():
            for name,m in values.items():writer.writerow([split,'GT_ORACLE_NOT_DEPLOYABLE',m['label'],name,*[m[k] for k in FIELDS]])
    # Original report/model numeric receipts must remain byte-identical.
    if sha(report/'summary.json')!=contract['original_summary_sha256'] or sha(report/'records.json')!=contract['original_records_sha256']:
        raise ValueError('Original results modified during review')
    shutil.copy2(Path(__file__).with_name('revision_explained_report.html'),report/'index.html')
    atomic_json(out/'verification.json',dict(status='passed',original_results_unchanged=True,no_model_or_selection_modified=True,
        test_oracle_queries=1472,val_oracle_queries=1704,checkpoint_last_policies=len(conditions),
        review_sha256=sha(report/'review.json'),oracle_records_sha256=sha(report/'oracle_records.json'),
        index_sha256=sha(report/'index.html'),code_sha256=sha(Path(__file__))))
    atomic_json(out/'status.json',dict(status='complete',updated_unix=time.time()))
    print({split:{n:dict(R1=m['recall_1m'],joint=m['recall_1m_30deg']) for n,m in values.items()} for split,values in all_oracles.items()},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','cache','report'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();run(a.root,a.cache,a.report)

"""Paired ABC metrics, weight distributions and fixed-baseline cohorts."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import csv
from pathlib import Path
import shutil

import numpy as np
from acoustic_branch.contracts import atomic_json,sha
from .input_supervision_audit import read,yawdiff
from .supervision_abc import REL,ARMS,fixed_good
from .response_input_localization import REL as DATA,load_scene,metrics
from .response_input_report import independent_metrics,verify_scene


def weight_stats(w):
    w=np.asarray(w,float);hist,edges=np.histogram(w,bins=np.linspace(0,1,21))
    return dict(outputs=len(w),mean=float(w.mean()),std=float(w.std()),median=float(np.median(w)),
        p10=float(np.quantile(w,.1)),p90=float(np.quantile(w,.9)),min=float(w.min()),max=float(w.max()),
        near01_fraction=float(np.mean(abs(w-.1)<=.01)),below01_fraction=float(np.mean(w<.1)),
        mean_abs_move_from01=float(np.mean(abs(w-.1))),histogram=hist.tolist(),edges=edges.tolist())


def main(args):
    root,ssd,report=args.root,args.ssd,args.report;out=root/REL;report.mkdir(parents=True,exist_ok=True)
    cfg=read(out/'config.json');ev=read(out/'evaluation.json');trained=read(out/'training_complete.json')
    assert ev['A_reproduced_exactly']and not ev['val_used_for_selection']and not ev['test_accessed']
    assert trained['same_supervised_mask']and trained['matched_initializations']and trained['matched_batch_orders']
    assert sha(root/DATA/'ready.json')==cfg['data_ready_sha256']
    refs=sorted(read(root/DATA/'ready.json')['scenes'],key=lambda r:r['scene_id']);assert{r['split']for r in refs}=={'train','val'}
    scenes=[load_scene(root/DATA,r['scene_id'],root,ssd)for r in refs]
    methods={};means={};arrays={};rows=[];scope={};histories={}
    for name,r in ev['methods'].items():
        rec=read(out/'runs'/name/'complete.json');assert rec['checkpoint_sha256']==r['checkpoint_sha256']
        assert sha(out/'runs'/name/'epoch017.pt')==r['checkpoint_sha256'];histories[name]=rec['history']
    for split in('train','val'):
        ss=[s for s in scenes if s['split']==split];truth=np.concatenate([s['truth']for s in ss]);fixed=np.concatenate([s['grid_poses'][:,2]for s in ss])
        baseline=np.concatenate([fixed_good(s)for s in ss]);valid=np.concatenate([s['valid']for s in ss])
        feasible=np.concatenate([[bool(q['success']['joint'])for q in s['queries']]for s in ss])
        cohort=np.where(baseline,'baseline_correct',np.where(feasible,np.where(valid,'repairable_supervised','repairable_not_supervised'),'no_continuous_success'))
        scope[split]=dict(scenes=len(ss),queries=len(truth),supervised=int(valid.sum()),cohorts={c:int(np.sum(cohort==c))for c in sorted(set(cohort))})
        pp={'fixed01':fixed};ws={'fixed01':np.full(len(truth),.1)};methods[split]={};means[split]={}
        for name,r in ev['methods'].items():
            p=r[split]['predictions'];pp[name]=np.concatenate([p[s['scene_id']]['poses']for s in ss]);ws[name]=np.concatenate([p[s['scene_id']]['weights']for s in ss])
        for name,p in pp.items():
            m=metrics(p,truth,fixed)
            if name!='fixed01':assert m==ev['methods'][name][split]['micro']
            for k,v in independent_metrics(p,truth,fixed).items():np.testing.assert_allclose(v,m[k],atol=1e-12,rtol=0)
            d=np.linalg.norm(p[:,:2]-truth[:,:2],axis=1);a=yawdiff(p[:,2],truth[:,2]);good=(d<=1)&(a<=30)
            per_scene={};offset=0
            for s in ss:
                n=len(s['truth']);sl=slice(offset,offset+n);per_scene[s['scene_id']]=metrics(p[sl],truth[sl],fixed[sl]);offset+=n
            bycohort={c:dict(queries=int(np.sum(cohort==c)),joint=float(good[cohort==c].mean()),weights=weight_stats(ws[name][cohort==c]))for c in sorted(set(cohort))}
            methods[split][name]=dict(micro=m,per_scene=per_scene,weights=weight_stats(ws[name]),cohorts=bycohort,
                own_loss=None if name=='fixed01'else ev['methods'][name][split]['own_loss'],
                common_A_loss=None if name=='fixed01'else ev['methods'][name][split]['common_A_loss'])
            if name!='fixed01':
                seed=ev['methods'][name]['seed'];againstA=metrics(p,truth,pp[f'A_seed{seed}'])
                methods[split][name]['relative_A']={k:v for k,v in againstA.items()if k.startswith(('repair_','harm_'))}
        for arm in ARMS:
            mm=[methods[split][f'{arm}_seed{s}']for s in cfg['seeds']]
            means[split][arm]=dict(micro={k:float(np.mean([m['micro'][k]for m in mm]))for k in mm[0]['micro']},
                pooled_weights=weight_stats(np.concatenate([ws[f'{arm}_seed{s}']for s in cfg['seeds']])),
                cohorts={c:dict(queries=scope[split]['cohorts'][c],joint=float(np.mean([m['cohorts'][c]['joint']for m in mm])),
                    pooled_weights=weight_stats(np.concatenate([ws[f'{arm}_seed{s}'][cohort==c]for s in cfg['seeds']])))for c in sorted(set(cohort))},
                own_loss=float(np.mean([m['own_loss']for m in mm])),common_A_loss=float(np.mean([m['common_A_loss']for m in mm])),
                relative_A={k:float(np.mean([m['relative_A'][k]for m in mm]))for k in mm[0]['relative_A']})
        names=list(pp);arrays[split]=dict(names=names,weights=np.array([ws[n]for n in names]).T,poses=np.stack([pp[n]for n in names],axis=1),scene_ids=np.concatenate([np.repeat(s['scene_id'],len(s['truth']))for s in ss]))
        offset=0
        for s in ss:
            for i,q in enumerate(s['queries']):
                j=offset+i
                for name in names:
                    p=pp[name][j];t=truth[j];rows.append(dict(split=split,scene_id=s['scene_id'],sample_index=q['sample_index'],cohort=str(cohort[j]),
                        supervised=bool(valid[j]),method=name,weight=float(ws[name][j]),x=float(p[0]),z=float(p[1]),yaw=float(p[2]),
                        xy_error=float(np.linalg.norm(p[:2]-t[:2])),yaw_error=float(yawdiff(p[2],t[2]))))
            offset+=len(s['truth'])
    assert scope['train']['supervised']==5364 and scope['train']['queries']==6387 and scope['val']['queries']==1704
    path=out/'dense_val_verification.json';fingerprint=sha(out/'evaluation.json')
    if path.exists():
        checked=read(path);assert checked['evaluation_sha256']==fingerprint
    else:
        tasks=[];a=arrays['val']
        for s in scenes:
            if s['split']!='val':continue
            mask=a['scene_ids']==s['scene_id'];tasks.append((str(root),str(ssd),s['scene_id'],a['names'],a['weights'][mask],a['poses'][mask]))
        with ProcessPoolExecutor(max_workers=3)as pool:receipts=[f.result()for f in as_completed([pool.submit(verify_scene,t)for t in tasks])]
        checked=dict(status='passed',evaluation_sha256=fingerprint,receipts=receipts,test_accessed=False);atomic_json(path,checked)
    result=dict(config=cfg,scope=scope,labels=ARMS,methods=methods,means=means,
        notes=['All Recall denominators include every query, even frozen-mask-ineligible queries.',
            'Repair/Harm reference FIXED w=.1, Joint criterion XY<=1m AND yaw<=30deg unless explicitly XY.',
            'Recall and Repair/Harm means average two seeds; pooled weight quantiles concatenate both seeds (each query twice), NOT independent examples.',
            'Own losses differ by supervision and cannot be ranked across arms. common_A_loss is only a shared point-regression diagnostic, not localization quality.',
            'No success-weight means no positive-width Joint-success interval in these frozen score fields and scalar fusion, not an intrinsically impossible scene.',
            'Val3 already repeatedly inspected. No checkpoint selection, epoch search, decay search, new training samples or Test evaluation in this comparison.'],
        test_accessed=False)
    atomic_json(out/'report_summary.json',result);atomic_json(report/'summary.json',result);atomic_json(report/'histories.json',histories)
    keys=list(methods['val']['fixed01']['micro'])
    with(report/'metrics.csv').open('w')as f:
        w=csv.writer(f);w.writerow(['split','method','aggregation',*keys])
        for split in methods:
            for n,m in methods[split].items():w.writerow([split,n,'single',*[m['micro'][k]for k in keys]])
            for n,m in means[split].items():w.writerow([split,n,'mean_of_two_seeds',*[m['micro'][k]for k in keys]])
    with(report/'weights.csv').open('w')as f:
        keysw=['outputs','mean','std','median','p10','p90','min','max','near01_fraction','below01_fraction','mean_abs_move_from01']
        w=csv.writer(f);w.writerow(['split','method','aggregation',*keysw])
        for split in methods:
            for n,m in methods[split].items():w.writerow([split,n,'single',*[m['weights'][k]for k in keysw]])
            for n,m in means[split].items():w.writerow([split,n,'pooled_two_seeds',*[m['pooled_weights'][k]for k in keysw]])
    with(report/'queries.csv').open('w')as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
    v=dict(status='passed',test_accessed=False,scope=scope,runs=6,epochs=17,weight_decay=.1,inputs=58,hidden=[128,64],
        A_reproduced_exactly=True,matched_initializations=True,matched_batch_order=True,same_eligible5364=True,
        no_val_selection=True,independent_metrics=True,canonical_dense_val_pose_checks=sum(r['policy_pose_checks']for r in checked['receipts']),
        config_sha256=sha(out/'config.json'),evaluation_sha256=fingerprint,source_sha256=sha(Path(__file__)),trainer_sha256=sha(Path(__file__).with_name('supervision_abc.py')))
    atomic_json(out/'verification.json',v);atomic_json(report/'verification.json',v)
    shutil.copy2(Path(__file__).with_suffix('.html'),report/'index.html');atomic_json(out/'status.json',dict(stage='complete',test_accessed=False))
    for arm,m in means['val'].items():
        mm=m['micro'];print(arm,'R1/Joint',mm['recall_1m'],mm['recall_1m_30deg'],'R/H',mm['repair_vs_fixed_joint'],mm['harm_vs_fixed_joint'],'weights',m['pooled_weights'],flush=True)
    print('REPORT',report,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--ssd',type=Path,required=True);p.add_argument('--report',type=Path,required=True);main(p.parse_args())

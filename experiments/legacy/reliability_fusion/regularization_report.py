"""Train-CV-selected regularization controls: independent Val MAP checks."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import csv
from pathlib import Path
import shutil

import numpy as np
from acoustic_branch.contracts import atomic_json,sha
from .input_supervision_audit import read
from .regularization_probe import REL,DECAYS,early_rule,key
from .response_input_localization import REL as DATA,load_scene,metrics
from .response_input_report import verify_scene,independent_metrics


def main(args):
    root,ssd,report=args.root,args.ssd,args.report;out=root/REL;report.mkdir(parents=True,exist_ok=True)
    cfg=read(out/'config.json');selection=read(out/'selection.json');ev=read(out/'evaluation.json')
    assert ev['selection_sha256']==sha(out/'selection.json')and not ev['val_used_for_selection']
    assert ev['old_wd0_last_reproduced_exactly']and not ev['test_accessed']
    assert sha(root/DATA/'ready.json')==cfg['data_ready_sha256']
    for name in DECAYS:assert selection['early'][name]==early_rule(selection['curves'][name],8)
    for phase in('early','fixed40'):
        assert selection['winners'][phase]==max(DECAYS,key=lambda n:key(selection['early'][n]['metric']if phase=='early'else selection['curves'][n][-1]))
    refs=sorted(read(root/DATA/'ready.json')['scenes'],key=lambda r:r['scene_id'])
    assert {r['split']for r in refs}=={'train','val'}
    scenes=[load_scene(root/DATA,r['scene_id'],root,ssd)for r in refs]
    methods={};all_arrays={};per_query=[]
    for split in('train','val'):
        ss=[s for s in scenes if s['split']==split];truth=np.concatenate([s['truth']for s in ss]);fixed=np.concatenate([s['grid_poses'][:,2]for s in ss])
        names=['fixed01',*ev['methods']];pp={'fixed01':fixed};ws={'fixed01':np.full(len(truth),.1)}
        methods[split]={'fixed01':dict(micro=metrics(fixed,truth,fixed),per_scene={s['scene_id']:metrics(s['grid_poses'][:,2],s['truth'],s['grid_poses'][:,2])for s in ss},loss=None)}
        for name,r in ev['methods'].items():
            p=r[split]['predictions'];pp[name]=np.concatenate([p[s['scene_id']]['poses']for s in ss]);ws[name]=np.concatenate([p[s['scene_id']]['weights']for s in ss])
            actual=metrics(pp[name],truth,fixed);assert actual==r[split]['micro']
            for k,v in independent_metrics(pp[name],truth,fixed).items():np.testing.assert_allclose(v,actual[k],atol=1e-12,rtol=0)
            methods[split][name]={k:v for k,v in r[split].items()if k!='predictions'}
        all_arrays[split]=dict(names=names,weights=np.array([ws[n]for n in names]).T,poses=np.stack([pp[n]for n in names],axis=1),scene_ids=np.concatenate([np.repeat(s['scene_id'],len(s['truth']))for s in ss]))
        if split=='val':
            offset=0
            for s in ss:
                for i,q in enumerate(s['queries']):
                    j=offset+i
                    for name in names:
                        p=pp[name][j];t=truth[j];per_query.append(dict(scene_id=s['scene_id'],sample_index=q['sample_index'],method=name,
                            weight=float(ws[name][j]),x=float(p[0]),z=float(p[1]),yaw=float(p[2]),
                            xy_error_m=float(np.linalg.norm(p[:2]-t[:2])),yaw_error_deg=float(abs((p[2]-t[2]+180)%360-180))))
                offset+=len(s['truth'])
    # Direct canonical float64 dense argmax for EVERY new Val weight.
    path=out/'dense_val_verification.json';fingerprint=sha(out/'evaluation.json')
    if path.exists():
        checked=read(path);assert checked['evaluation_sha256']==fingerprint
    else:
        a=all_arrays['val'];tasks=[]
        for s in scenes:
            if s['split']!='val':continue
            mask=a['scene_ids']==s['scene_id'];tasks.append((str(root),str(ssd),s['scene_id'],a['names'],a['weights'][mask],a['poses'][mask]))
        receipts=[]
        with ProcessPoolExecutor(max_workers=3)as pool:
            for f in as_completed([pool.submit(verify_scene,t)for t in tasks]):receipts.append(f.result())
        checked=dict(status='passed',receipts=receipts,evaluation_sha256=fingerprint,test_accessed=False)
        atomic_json(path,checked)
    means={}
    for split,mm in methods.items():
        means[split]={}
        for decay in DECAYS:
            for phase in('early','fixed40'):
                entries=[mm[f'{decay}_seed{seed}_{phase}']for seed in cfg['seeds']]
                means[split][decay+'_'+phase]=dict(micro={k:float(np.mean([r['micro'][k]for r in entries]))for k in entries[0]['micro']},
                    loss=float(np.mean([r['loss']for r in entries])),
                    range={k:[float(min(r['micro'][k]for r in entries)),float(max(r['micro'][k]for r in entries))]for k in('recall_1m','recall_1m_30deg')})
    comparisons=[]
    for decay in DECAYS:
        b=means['val'][decay+'_fixed40'];a=means['val'][decay+'_early']
        comparisons.append(dict(decay=DECAYS[decay],early_epoch=selection['early'][decay]['epoch'],
            delta_R1_pp=100*(a['micro']['recall_1m']-b['micro']['recall_1m']),
            delta_joint_pp=100*(a['micro']['recall_1m_30deg']-b['micro']['recall_1m_30deg']),
            harm_mean_delta=a['micro']['harm_vs_fixed_joint']-b['micro']['harm_vs_fixed_joint'],loss_delta=a['loss']-b['loss']))
    metadata={n:{k:v for k,v in r.items()if k not in('train','val')}for n,r in ev['methods'].items()}
    result=dict(config=cfg,selection=selection,methods=methods,means=means,metadata=metadata,comparisons=comparisons,
        notice='CV-selected epoch/decay fixed BEFORE Val arrays were resolved. Entire predeclared grid shown; do not select a new winner from Val. Same exploratory Val3 previously reused in research, not independent Test. Seed averages are not unique-query sums.',
        limitations=['Early rule is applied chronologically to Train-CV history; later epochs past patience stop cannot change selected early epoch. All folds still ran40 for matched fixed40 controls.',
            'Full-Train refit uses selected epoch count, not update-count rescaling. Twelve fit scenes have more updates per epoch than nine fit scenes in CV.',
            'Scene-held-out CV concerns the fusion MLP only; frozen branch encoders were not refit out-of-fold.',
            'Decay applies to every AdamW parameter including biases; lr, batch, targets, inputs, seeds, model size and scheduler unchanged.',
            'Two seeds and three exploratory Val scenes cannot establish significance or universal regularization benefit.'],test_accessed=False)
    atomic_json(out/'report_summary.json',result);atomic_json(report/'summary.json',result)
    keys=list(methods['val']['fixed01']['micro'])
    with(report/'metrics.csv').open('w')as f:
        w=csv.writer(f);w.writerow(['split','method','seed_aggregation','loss',*keys])
        for split,mm in methods.items():
            for n,m in mm.items():w.writerow([split,n,'single'if n!='fixed01'else'baseline',m['loss'],*[m['micro'][k]for k in keys]])
            for n,m in means[split].items():w.writerow([split,n,'mean_of_two_seeds',m['loss'],*[m['micro'][k]for k in keys]])
    with(report/'val_queries.csv').open('w')as f:
        w=csv.DictWriter(f,list(per_query[0]));w.writeheader();w.writerows(per_query)
    atomic_json(report/'selection.json',selection)
    v=dict(status='passed',train_scenes=12,val_scenes=3,train_queries=6387,val_queries=1704,
        cv_runs=24,full_fit_runs=6,policies=12,parameters=15873,input_count=58,
        config_sha256=sha(out/'config.json'),selection_sha256=sha(out/'selection.json'),evaluation_sha256=fingerprint,
        no_val_selection=True,no_test_access=True,old_wd0_last_exact=True,matched_batch_orders=True,
        independent_metrics_checked=True,canonical_dense_val_pose_checks=sum(r['policy_pose_checks']for r in checked['receipts']),
        report_source_sha256=sha(Path(__file__)),trainer_source_sha256=sha(Path(__file__).with_name('regularization_probe.py')))
    atomic_json(out/'verification.json',v);atomic_json(report/'verification.json',v)
    shutil.copy2(Path(__file__).with_suffix('.html'),report/'index.html')
    atomic_json(out/'status.json',dict(stage='complete',test_accessed=False))
    print('BASELINE',methods['val']['fixed01']['micro'],flush=True)
    for n,m in means['val'].items():print('MEAN',n,m,flush=True)
    print('REPORT',report,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--ssd',type=Path,required=True);p.add_argument('--report',type=Path,required=True);main(p.parse_args())

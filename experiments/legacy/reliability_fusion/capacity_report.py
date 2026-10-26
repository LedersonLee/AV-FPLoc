"""Train/Val-only capacity tables, independent metric/dense MAP checks."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import csv
from pathlib import Path
import shutil
import numpy as np

from acoustic_branch.contracts import atomic_json,sha
from .input_supervision_audit import read
from .capacity_probe import REL,WIDTHS
from .response_input_localization import REL as DATA,load_scene,metrics
from .response_input_report import independent_metrics,verify_scene


def checked_dense(task):
    root,ssd,sid=task[:3];root,ssd=map(Path,(root,ssd))
    s=load_scene(root/DATA,sid,root,ssd);ref=s['ref'];receipt=s['source']/'receipt.json'
    assert sha(receipt)==ref['receipt_sha256']
    for name,digest in read(receipt)['files'].items():assert sha(s['source']/name)==digest
    return verify_scene(task)


def main(args):
    root,ssd,report=args.root,args.ssd,args.report;out=root/REL;report.mkdir(parents=True,exist_ok=True)
    config=read(out/'config.json');done=read(out/'training_complete.json');assert done['small_model_reproduced']and done['batch_order_matched']
    refs=sorted(read(root/DATA/'ready.json')['scenes'],key=lambda r:r['scene_id']);scenes=[load_scene(root/DATA,r['scene_id'],root,ssd)for r in refs]
    assert {s['split']for s in scenes}=={'train','val'}
    runs={p.parent.name:read(p)for p in sorted((out/'runs').glob('*/complete.json'))};assert len(runs)==6
    spec={};methods={};arrays={};histories={}
    for name,r in runs.items():
        hidden=r['hidden'];parameters=(58+1)*hidden[0]+(hidden[0]+1)*hidden[1]+hidden[1]+1;assert parameters==r['parameters']
        spec[name]=dict(hidden=hidden,parameters=parameters,seed=r['seed'],best_epoch=r['evaluated']['best']['epoch'],last_epoch=40,
            batch_order_sha256=r['batch_order_sha256'],best_sha256=r['best_sha256'],last_sha256=r['last_sha256'])
        histories[name]=r['history']
    for split in('train','val'):
        ss=[s for s in scenes if s['split']==split];truth=np.concatenate([s['truth']for s in ss]);fixed=np.concatenate([s['grid_poses'][:,2]for s in ss])
        pp={'fixed01':fixed};weights={'fixed01':np.full(len(truth),.1)}
        for name,r in runs.items():
            for phase in('best','last'):
                ev=r['evaluated'][phase][split];ps=ev['predictions'];key=name+'__'+phase
                pp[key]=np.concatenate([ps[s['scene_id']]['poses']for s in ss]);weights[key]=np.concatenate([ps[s['scene_id']]['weights']for s in ss])
                actual=metrics(pp[key],truth,fixed)
                assert actual==ev['micro']
                independent=independent_metrics(pp[key],truth,fixed)
                for k,v in independent.items():np.testing.assert_allclose(v,actual[k],atol=1e-12,rtol=0)
        names=list(pp);methods[split]={};offset=0
        for name,p in pp.items():
            per_scene={};offset=0
            for s in ss:
                n=len(s['truth']);sl=slice(offset,offset+n);per_scene[s['scene_id']]=metrics(p[sl],truth[sl],fixed[sl]);offset+=n
            methods[split][name]=dict(micro=metrics(p,truth,fixed),per_scene=per_scene,weight_mean=float(weights[name].mean()),weight_std=float(weights[name].std()),weights_below01=int((weights[name]<.1).sum()))
            if name!='fixed01':
                run,phase=name.rsplit('__',1);methods[split][name]['loss']=runs[run]['evaluated'][phase][split]['loss']
        arrays[split]=dict(names=names,truth=truth,poses=np.stack([pp[n]for n in names],axis=1),weights=np.stack([weights[n]for n in names],axis=1),scene_ids=np.concatenate([np.repeat(s['scene_id'],len(s['truth']))for s in ss]),sample_indices=np.concatenate([[q['sample_index']for q in s['queries']]for s in ss]))
        np.savez_compressed(report/(split+'_predictions.npz'),**{k:np.array(v)for k,v in arrays[split].items()})
    # Hashes bind resumption to the exact frozen source and all six new runs.
    run_hashes={n:sha(out/'runs'/n/'complete.json')for n in runs};vp=out/'dense_verification.json'
    if vp.exists():
        checked=read(vp);assert checked['run_hashes']==run_hashes and checked['data_ready_sha256']==config['data_ready_sha256']
    else:
        tasks=[]
        for s in scenes:
            a=arrays[s['split']];take=a['scene_ids']==s['scene_id']
            tasks.append((str(root),str(ssd),s['scene_id'],a['names'],a['weights'][take],a['poses'][take]))
        receipts=[]
        with ProcessPoolExecutor(max_workers=6)as pool:
            for f in as_completed([pool.submit(checked_dense,t)for t in tasks]):
                receipts.append(f.result());atomic_json(out/'status.json',dict(stage='dense_verification',done=len(receipts),total=15,test_accessed=False))
        checked=dict(status='passed',receipts=receipts,run_hashes=run_hashes,data_ready_sha256=config['data_ready_sha256'],test_accessed=False)
        atomic_json(vp,checked)
    means=[]
    for name,hidden in WIDTHS.items():
        for phase in('best','last'):
            for split in('train','val'):
                rr=[methods[split][f'{name}_seed{seed}__{phase}']['micro']for seed in config['seeds']]
                means.append(dict(width=name,phase=phase,split=split,recall_1m_mean=float(np.mean([r['recall_1m']for r in rr])),joint_mean=float(np.mean([r['recall_1m_30deg']for r in rr]))))
    result=dict(config=config,runs=spec,methods=methods,means=means,
        notice='Best is selected on Val scene-macro R1, then Joint, then own loss, including epoch0. Train Best uses THAT SAME Val-selected checkpoint, not Train-optimal checkpoint. Val3 reused for model development, not independent Test or equivalence proof.',
        frozen_input_names=refs[0]['features'][:58],test_accessed=False)
    atomic_json(out/'evaluation.json',result);atomic_json(report/'summary.json',result);atomic_json(report/'histories.json',histories)
    keys=list(methods['val']['fixed01']['micro'])
    with(report/'metrics.csv').open('w')as f:
        writer=csv.writer(f);writer.writerow(['split','run','checkpoint',*keys])
        for split,mm in methods.items():
            for key,m in mm.items():writer.writerow([split,key.rsplit('__',1)[0],key.rsplit('__',1)[-1]if'__'in key else'fixed',*[m['micro'][k]for k in keys]])
    lines=[]
    for split in('train','val'):
        lines.append(split.upper());lines.append('| Width | seed | Best ep | Best R1/Joint | Best Joint R/H | Last R1/Joint | Last Joint R/H |')
        lines.append('|---|---:|---:|---:|---:|---:|---:|')
        for width in WIDTHS:
            for seed in config['seeds']:
                run=f'{width}_seed{seed}';b=methods[split][run+'__best']['micro'];l=methods[split][run+'__last']['micro']
                lines.append(f"| {width.replace('_','→')} | {seed} | {spec[run]['best_epoch']} | {b['recall_1m']*100:.2f}/{b['recall_1m_30deg']*100:.2f} | {b['repair_vs_fixed_joint']}/{b['harm_vs_fixed_joint']} | {l['recall_1m']*100:.2f}/{l['recall_1m_30deg']*100:.2f} | {l['repair_vs_fixed_joint']}/{l['harm_vs_fixed_joint']} |")
    print('\n'.join(lines),flush=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(12,4),constrained_layout=True);x=np.arange(3)
    for split,phase,color in [('train','last','#277da8'),('val','last','#d8922d'),('val','best','#8955b9')]:
        for k,metric in enumerate(['recall_1m','recall_1m_30deg']):
            vals=np.array([[methods[split][f'{n}_seed{s}__{phase}']['micro'][metric]*100 for s in config['seeds']]for n in WIDTHS])
            ax[k].plot(x,vals.mean(1),'o-',label=f'{split} {phase}',color=color);ax[k].fill_between(x,vals.min(1),vals.max(1),alpha=.12,color=color)
    for k,metric in enumerate(['recall_1m','recall_1m_30deg']):
        ax[k].axhline(methods['val']['fixed01']['micro'][metric]*100,color='gray',linestyle='--',label='Val fixed w=.1');ax[k].set(xticks=x,xticklabels=list(WIDTHS),ylabel='Recall (%)',title=metric);ax[k].legend()
    fig.savefig(report/'capacity.png',dpi=150);plt.close(fig)
    verification=dict(status='passed',runs=6,parameters={n:spec[n+'_seed42']['parameters']for n in WIDTHS},train_queries=6387,val_queries=1704,train_scenes=12,val_scenes=3,
        dense_pose_checks=sum(r['policy_pose_checks']for r in checked['receipts']),metric_independently_checked=True,source_files_sha256_checked=True,baseline_reproduced_exactly=True,batch_order_matched=True,
        evaluation_sha256=sha(out/'evaluation.json'),config_sha256=sha(out/'config.json'),trainer_sha256=sha(Path(__file__).with_name('response_input_localization.py')),test_accessed=False)
    atomic_json(out/'verification.json',verification);atomic_json(report/'verification.json',verification)
    shutil.copy2(Path(__file__).with_name('capacity_report.html'),report/'index.html');atomic_json(out/'status.json',dict(stage='complete',test_accessed=False))
    print('REPORT COMPLETE',report,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--ssd',type=Path,required=True);p.add_argument('--report',type=Path,required=True);main(p.parse_args())

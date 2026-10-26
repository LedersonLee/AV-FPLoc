"""Exploratory train-only endpoint precision control; baseline results preserved."""
import copy
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from acoustic_branch.contracts import atomic_json,immutable_json,sha
from acoustic_branch.data import read
from .overfit_probe import REL,prepare,fit


def run(root,cache,report):
    root,cache,report=map(Path,(root,cache,report));out=root/REL;control=out/'controls/logit_v1';control.mkdir(parents=True,exist_ok=True)
    if read(out/'status.json')['status']!='complete':raise ValueError('Finish frozen baseline matrix first')
    config,data=prepare(root,cache);baseline=read(out/'summary.json');config=copy.deepcopy(config)
    config['arms']={'interior_logit_mse':dict(loss='interior_logit',weight_decay=0.)}
    config['rationale']='Exploratory follow-up: narrow successful weight interval near0 and sigmoid/MSE gradient precision, after observing train failures. SAME input/MLP; no val/test.'
    config['loss']='MSE between pre-sigmoid model logit and logit of exact-success-interval midpoint; deployed output remains sigmoid scalar'
    immutable_json(control/'config.json',dict(config=config,base_summary_sha256=sha(out/'summary.json'),selection_sha256=sha(out/'selection.json')))
    def status(stage,**extra):
        value=dict(status='running',stage='logit_control_'+stage,updated_unix=time.time(),val_test_arrays_accessed=False,**extra)
        atomic_json(control/'status.json',value);atomic_json(report/'status.json',value)
    status('start');torch.set_num_threads(2);n=max(config['sizes']);width=max(s.shape[1] for s in data['scores'])
    bank=torch.full((n,2,width),-1e100,dtype=torch.float64,device='cuda')
    for i,s in enumerate(data['scores']):bank[i,:,:s.shape[1]]=torch.tensor(s,device='cuda')
    tracker=None
    try:
        import wandb
        tracker=wandb.init(project='hssd-reliability-fusion',name='core14-train-overfit-logit-control-v1',dir=str(control),config=config)
        atomic_json(control/'wandb.json',dict(id=tracker.id,url=tracker.url));results=[]
        for size in config['sizes']:
            for seed in config['seeds']:
                r=fit(data,config,size,'interior_logit_mse',seed,bank,control/'runs'/f'n{size}_seed{seed}',status,tracker)
                results.append(r)
        combined=dict(baseline,results=baseline['results']+results,supplement=dict(config=config,notice='Exploratory post-baseline training-only optimization control. Not an independent confirmation.'))
        atomic_json(control/'summary.json',dict(config=config,results=results));atomic_json(out/'summary_with_logit.json',combined);atomic_json(report/'summary.json',combined)
        shutil.copy2(Path(__file__).with_name('overfit_report.html'),report/'index.html')
        import csv
        fields=['recall_0.1m','recall_0.5m','recall_1m','recall_1m_30deg','yaw_within_10deg','yaw_within_30deg','yaw_within_50deg','median_xy_m','median_yaw_deg',
            'repair_vs_fixed_xy','harm_vs_fixed_xy','repair_vs_fixed_joint','harm_vs_fixed_joint','set_loss','interior_mse','objective_loss']
        with (report/'metrics.csv').open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['n','arm','seed','checkpoint','step',*fields])
            for r in combined['results']:
                for stage in ('initial','best','last'):writer.writerow([r['n'],r['arm'],r['seed'],stage,r[stage]['step'],*[r[stage][k] for k in fields]])
        done=dict(status='complete',updated_unix=time.time(),val_test_arrays_accessed=False,logit_control_included=True)
        atomic_json(control/'status.json',done);atomic_json(report/'status.json',done)
    except Exception:
        import traceback
        value=dict(status='failed',error=traceback.format_exc());atomic_json(control/'status.json',value);atomic_json(report/'status.json',value);raise
    finally:
        if tracker:tracker.finish()


if __name__=='__main__':
    import sys
    run(*sys.argv[1:])

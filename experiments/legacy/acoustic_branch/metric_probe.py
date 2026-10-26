"""Small pairwise metric learner: train-only normalization and InfoNCE fit.

Tests whether coarse time/frequency reflection matching generalizes better
than high-capacity shared embeddings. No pose, yaw, RGB, or scene IDs enter.
"""
import argparse
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from .contracts import atomic_json, sha
from .data import read, load_manifest, check_cache
from .train import PairDataset
from .diagnose import tensors


def pair_features(oe, oc, ce, cc):
    """Nonnegative distances [batch,8,feature], ordered channels retained."""
    oe, ce = torch.expm1(oe)/100, torch.expm1(ce)/100
    features=[]
    # 19 STFT bins = 562.5..3937.5 Hz; split without claiming exact cutoffs.
    for f in (slice(0,5),slice(5,11),slice(11,None)):
        for t in (slice(0,24),slice(24,48),slice(48,None)):
            o=oe[:,None,:,f,t]
            c=ce[:,:,:,f,t]
            # Relative amplitudes and arrival-pattern overlap; robust common gain.
            for op,cp in ((o,c), (o/o.amax((-1,-2,-3),keepdim=True).clamp_min(1e-12),
                                 c/c.amax((-1,-2,-3),keepdim=True).clamp_min(1e-12))):
                features.extend([(op-cp).abs().mean((-1,-2,-3)),
                                 (cp-op).clamp_min(0).mean((-1,-2,-3))])
            features.append((o.sqrt()-c.sqrt()).square().mean((-1,-2,-3)))
        for window in range(2):
            o=oc[:,None,window,:,:,f]
            c=cc[:,:,window,:,:,f]
            features.extend([(o-c).abs().mean((-1,-2,-3)),
                             (o.square().sum(-2)-c.square().sum(-2)).abs().mean((-1,-2))])
    return torch.stack(features,-1)


def run(root, config_path, output):
    torch.set_num_threads(4)
    torch.manual_seed(42)
    config=read(config_path)
    data={}
    for split in ('train','val'):
        queries,digest=load_manifest(root,split,config)
        locations,_=check_cache(root,config,split,digest,full=False)
        dataset=PairDataset(queries,locations)
        arrays=[]
        for start in range(0,len(queries),32):
            batch=tensors(dataset,range(start,min(start+32,len(queries))),'cpu')
            arrays.append(pair_features(*batch).numpy())
        data[split]=dict(queries=queries,x=np.concatenate(arrays))
        atomic_json(output/'metric_status.json',dict(status='running',stage='features_'+split,updated_unix=time.time()))
    x=data['train']['x']
    mean,std=x.mean((0,1)),np.maximum(x.std((0,1)),1e-12)
    fit=torch.from_numpy((x-mean)/std)
    val=torch.from_numpy((data['val']['x']-mean)/std)
    result=dict(test_accessed=False,config_sha256=sha(config_path),features=fit.shape[-1],
        protocol='linear pair-distance InfoNCE; five regularizations selected on val, no test access',sweep=[])
    for regularization in (.0001,.001,.01,.1,1.):
        weight=torch.zeros(fit.shape[-1],requires_grad=True)
        optimizer=torch.optim.Adam([weight],lr=.02)
        for step in range(501):
            optimizer.zero_grad()
            logits=fit@weight
            loss=F.cross_entropy(logits,torch.zeros(len(fit),dtype=torch.long))+regularization*weight.square().sum()
            if not torch.isfinite(loss): raise FloatingPointError('Metric nonfinite loss')
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            vs=(val@weight).numpy()
            ts=(fit@weight).numpy()
        record=dict(regularization=regularization,train_top1=float((ts.argmax(-1)==0).mean()),
                    val_top1=float((vs.argmax(-1)==0).mean()),weights=weight.detach().tolist(),
                    records=[dict(scene=q['scene_id'],sample_index=q['sample_index'],xyz=q['observed']['xyz'],
                                  top1_credit=float(s.argmax()==0),scores=s.tolist()) for q,s in zip(data['val']['queries'],vs)])
        result['sweep'].append(record)
        result['best_regularization']=max(result['sweep'],key=lambda r:r['val_top1'])['regularization']
        atomic_json(output/'metric_results.json',result)
        print({k:v for k,v in record.items() if k not in ('records','weights')},flush=True)
    np.savez(output/'metric_normalization.npz',mean=mean,std=std)
    atomic_json(output/'metric_status.json',dict(status='complete',test_accessed=False,updated_unix=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    try:run(a.root,a.config,a.output)
    except Exception as e:
        atomic_json(a.output/'metric_status.json',dict(status='failed',error=repr(e)))
        raise

"""Low-capacity cross-domain ridge probe; fit train only, select alpha on val."""
import argparse
from pathlib import Path
import time

import numpy as np
import torch

from .contracts import atomic_json
from .data import read, load_manifest, check_cache
from .train import FeatureStore


def vector(store, ref):
    e, c = store.get(ref)
    # Keep ordered microphones and coarse frequency/time structure.
    e = torch.nn.functional.adaptive_avg_pool2d(e, (5, 19)).numpy().ravel()
    return np.concatenate([e, c.numpy().ravel()])


def credit(scores):
    win = scores == scores.max(axis=-1, keepdims=True)
    return float(np.mean(win[:,0]/win.sum(-1)))


def run(root, config_path, output):
    torch.set_num_threads(1)
    config = read(config_path)
    data = {}
    for split in ('train', 'val'):
        queries, digest = load_manifest(root, split, config)
        locations, _ = check_cache(root, config, split, digest, full=False)
        store = FeatureStore(locations)
        obs, positive = [], []
        for q in queries:
            obs.append(vector(store, q['observed']))
            positive.append(vector(store, q['candidates'][0]))
        data[split] = dict(queries=queries, store=store,
            observed=np.asarray(obs), positive=np.asarray(positive))
        atomic_json(output/'linear_status.json', dict(status='running', stage='loaded_'+split, updated_unix=time.time()))
    result = dict(test_accessed=False, fitting='train positives only, separate observed-to-FP ridge map',
                  selection='alpha selected by official val eight-candidate top1; not a final test result', modes={})
    energy_dim = 6*5*19
    for mode, selection in dict(energy=slice(0,energy_dim), coherence=slice(energy_dim,None), both=slice(None)).items():
        x = data['train']['observed'][:,selection].astype(np.float64)
        y = data['train']['positive'][:,selection].astype(np.float64)
        mx, my = x.mean(0), y.mean(0)
        sx, sy = np.maximum(x.std(0),1e-6), np.maximum(y.std(0),1e-6)
        x, y = (x-mx)/sx, (y-my)/sy
        # Trace-scaled ridge is comparable across feature dimensionalities.
        gram, cross = x.T@x, x.T@y
        scale = np.trace(gram)/len(gram)
        val = data['val']
        vx = (val['observed'][:,selection]-mx)/sx
        candidates = np.asarray([[vector(val['store'], r)[selection] for r in q['candidates']] for q in val['queries']])
        candidates = (candidates-my)/sy
        candidates /= np.maximum(np.linalg.norm(candidates,axis=-1,keepdims=True),1e-12)
        entries=[]
        for alpha in (.01,.1,1.,10.,100.):
            w = np.linalg.solve(gram+alpha*scale*np.eye(len(gram)), cross)
            pred = vx@w
            pred /= np.maximum(np.linalg.norm(pred,axis=-1,keepdims=True),1e-12)
            scores = np.einsum('bd,bkd->bk', pred, candidates)
            spatial, orientation = [], []
            for q, s in zip(val['queries'], scores):
                spatial.append(credit(s[[0]+[i for i in range(1,8) if q['candidates'][i]['kind']!='orientation_hard']][None]))
                orientation.append(credit(s[[0]+[i for i in range(1,8) if q['candidates'][i]['kind']=='orientation_hard']][None]))
            row = dict(alpha=alpha, top1=credit(scores), spatial_top1=float(np.mean(spatial)), orientation_top1=float(np.mean(orientation)))
            entries.append(row)
            print(mode, row, flush=True)
        best = max(entries, key=lambda r:r['top1'])
        w = np.linalg.solve(gram+best['alpha']*scale*np.eye(len(gram)), cross)
        pred = vx@w
        pred /= np.maximum(np.linalg.norm(pred,axis=-1,keepdims=True),1e-12)
        final_scores = np.einsum('bd,bkd->bk', pred, candidates)
        winners = final_scores == final_scores.max(-1,keepdims=True)
        credits = winners[:,0]/winners.sum(-1)
        bank = (val['positive'][:,selection]-my)/sy
        bank /= np.maximum(np.linalg.norm(bank,axis=-1,keepdims=True),1e-12)
        records=[]
        for scene in sorted({q['scene_id'] for q in val['queries']}):
            ids=np.array([i for i,q in enumerate(val['queries']) if q['scene_id']==scene])
            nearest=ids[(pred[ids]@bank[ids].T).argmax(-1)]
            for i,j in zip(ids,nearest):
                q,r=val['queries'][i],val['queries'][j]
                xy=float(np.linalg.norm((np.array(q['observed']['xyz'])-r['observed']['xyz'])[[0,2]]))
                yaw=abs((q['observed']['yaw']-r['observed']['yaw']+180)%360-180)
                records.append(dict(scene=scene,sample_index=q['sample_index'],xyz=q['observed']['xyz'],
                    top1_credit=float(credits[i]),xy_error_m=xy,yaw_error_deg=yaw))
        result['modes'][mode] = dict(sweep=entries, best=best, records=records,
            captured_bank=dict(**{f'recall_{t:g}m':float(np.mean([r['xy_error_m']<=t for r in records])) for t in (.1,.5,1)},
                recall_1m_30deg=float(np.mean([r['xy_error_m']<=1 and r['yaw_error_deg']<=30 for r in records])),
                median_xy_error_m=float(np.median([r['xy_error_m'] for r in records]))))
        np.savez(output/f'linear_{mode}.npz',w=w,mx=mx,my=my,sx=sx,sy=sy,alpha=best['alpha'])
        atomic_json(output/'linear_results.json', result)
    atomic_json(output/'linear_status.json', dict(status='complete', test_accessed=False, updated_unix=time.time()))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    try:
        run(a.root,a.config,a.output)
    except Exception as e:
        atomic_json(a.output/'linear_status.json',dict(status='failed',error=repr(e)))
        raise

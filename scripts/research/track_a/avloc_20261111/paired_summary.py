"""Paired scene uncertainty; seeds are averaged within scenes, never independent scenes."""
import numpy as np

def matched_keys(rows):
 keys=[(r['scene'],int(r['query_id'])) for r in rows]
 if len(set(keys))!=len(keys):raise ValueError('Duplicate scene/query IDs')
 return keys

def align(rows,expected):
 keys=matched_keys(rows);lookup=dict(zip(keys,rows))
 if set(keys)!=set(expected):raise ValueError('Query cohort mismatch')
 return [lookup[k] for k in expected]

def paired(model,baseline,scene_ids,replicates=10000,seed=20261111):
 m=np.atleast_2d(np.asarray(model,dtype=float));b=np.atleast_2d(np.asarray(baseline,dtype=float));sc=np.asarray(scene_ids)
 assert m.shape[-1]==b.shape[-1]==len(sc) and len(sc)>0
 assert m.shape[0]==b.shape[0] or min(m.shape[0],b.shape[0])==1
 assert np.isfinite(m).all() and np.isfinite(b).all()
 diff=(m-b).mean(0);names=np.unique(sc);by=np.array([diff[sc==s].mean() for s in names]);counts=np.array([(sc==s).sum() for s in names]);rng=np.random.default_rng(seed);take=rng.integers(len(names),size=(replicates,len(names)));macro=by[take].mean(1);micro=(by[take]*counts[take]).sum(1)/counts[take].sum(1)
 repair=((m>0)&(b==0)).mean(0);harm=((m==0)&(b>0)).mean(0)
 return dict(micro_delta=float(diff.mean()),macro_delta=float(by.mean()),macro_ci95=np.quantile(macro,[.025,.975]).tolist(),micro_ci95=np.quantile(micro,[.025,.975]).tolist(),repair_count_seed_mean=float(repair.sum()),harm_count_seed_mean=float(harm.sum()),replicates=replicates,bootstrap_seed=seed,scenes=len(names),queries=len(sc),model_seeds=m.shape[0],baseline_seeds=b.shape[0],per_scene_delta={str(s):float(v) for s,v in zip(names,by)},scope='paired scene bootstrap, average seeds within scene; reused development, no independence from model selection implied')

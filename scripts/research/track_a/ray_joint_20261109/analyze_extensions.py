"""Scene-level paired summaries; seeds are repeated measurements, not independent N."""
import json,itertools
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent
METRICS=['joint','fine','R1','pair_accuracy','repair','harm','high_ambiguity_joint']
def summarize(rows,keys,metrics=METRICS):
 out={}
 for values in sorted(set(tuple(r[k] for k in keys) for r in rows)):
  rr=[r for r in rows if tuple(r[k] for k in keys)==values];sc=sorted(set(r['scene'] for r in rr));boot=np.random.default_rng(20261109).integers(len(sc),size=(20000,len(sc)));entry={}
  for metric in metrics:
   if metric not in rr[0]:continue
   v=np.array([np.mean([r[metric] for r in rr if r['scene']==s]) for s in sc]);entry[metric]=dict(mean=float(v.mean()),ci95=np.quantile(v[boot].mean(1),[.025,.975]).tolist(),scene_values=dict(zip(sc,v.tolist())))
  out['/'.join(map(str,values))]=entry
 return out
def difference(a,b):
 result={}
 for metric in set(a)&set(b):
  sc=sorted(a[metric]['scene_values']);assert sc==sorted(b[metric]['scene_values']);d=np.array([a[metric]['scene_values'][s]-b[metric]['scene_values'][s] for s in sc]);boot=np.random.default_rng(20261109).integers(len(sc),size=(20000,len(sc)));result[metric]=dict(mean=float(d.mean()),ci95=np.quantile(d[boot].mean(1),[.025,.975]).tolist())
 return result
out={'scope':'Exploratory reused scenes; seed means within scene; 20k scene bootstrap, unadjusted CIs.'}
f=json.loads((P/'fov_late_results.json').read_text());out['fov']=summarize(f['records'],['split','kind']);out['fov_rays']=summarize(f['ray_records'],['split','kind'],['in_fov_mae','out_fov_mae']);out['fov_contrasts']={}
for split in ['development','transfer']:
 names=[k for k in out['fov'] if k.startswith(split+'/')]
 for a,b in itertools.combinations(names,2):out['fov_contrasts'][a+' minus '+b]=difference(out['fov'][a],out['fov'][b])
for name,filename in [('conditioned','conditioned_candidate_results.json'),('transfer','transfer_evaluation.json')]:
 j=json.loads((P/filename).read_text());s=summarize(j['records'],['group','kind','control','method','fusion']);out[name]=s;out[name+'_contrasts']={}
 for key in s:
  g,k,c,m,fusion=key.split('/')
  if c=='matched':
   baseline='/'.join([g,k,c,'uniform',fusion])
   if m!='uniform' and baseline in s:out[name+'_contrasts'][key+' minus uniform']=difference(s[key],s[baseline])
   for ctrl in ['donor17','shuffled']:
    other='/'.join([g,k,ctrl,m,fusion])
    if other in s:out[name+'_contrasts'][key+' minus '+ctrl]=difference(s[key],s[other])
 if 'baselines' in j:out['transfer_baselines']=summarize(j['baselines'],['arm'])
(P/'extension_analysis.json').write_text(json.dumps(out,indent=2))
print(json.dumps({k:v['joint'] for k,v in out['fov_contrasts'].items()},indent=2))

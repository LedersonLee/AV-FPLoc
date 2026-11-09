"""All-severity matched comparison against the frozen independent F1/A controls."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

def read(p):return json.loads(p.read_text())
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--foundation',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 rows=read(a.run/'predictions.json');assert len(rows)==13824 and read(a.run/'normal_replay_mismatches.json')==[]
 a.out.mkdir(parents=True,exist_ok=False);rng=np.random.default_rng(20261109);pairs=[];controls=[]
 for level in [0,25,50,75]:
  source=a.foundation/('independent_fusion_anchor'+(f'_occlusion{level}' if level else '')+'_v1')
  ref=read(source/'predictions.json');rr=[r for r in rows if r['occlusion_percent']==level]
  lookup={method:{(r['seed'],r['scene'],r['query']):r for r in ref if r['method']==method} for method in ['fixed','acoustic']}
  for method in lookup:controls.append(dict(level=level,method=method,joint=float(np.mean([r['joint'] for r in lookup[method].values()]))))
  scenes=sorted({r['scene'] for r in rr})
  for mode in ['attention','concat','separable']:
   own=[r for r in rr if r['mode']==mode];keys={(r['seed'],r['scene'],r['query']) for r in own}
   assert keys==set(lookup['fixed'])==set(lookup['acoustic'])
   comparisons=dict(lookup)
   if mode=='attention':comparisons.update({m:{(r['seed'],r['scene'],r['query']):r for r in rr if r['mode']==m} for m in ['concat','separable']})
   for control,index in comparisons.items():
    for metric in ['joint','fine','nll']:
     d=np.array([np.mean([r[metric]-index[r['seed'],r['scene'],r['query']][metric] for r in own if r['scene']==s]) for s in scenes]);scale=1 if metric=='nll' else 100;b=d[rng.integers(len(d),size=(10000,len(d)))].mean(1)*scale
     pairs.append(dict(mode=mode,level=level,control=control,metric=metric,delta=float(d.mean()*scale),ci95=np.quantile(b,[.025,.975]).tolist(),units='nats' if metric=='nll' else 'pp',unit='scene, seeds averaged'))
 summary=read(a.run/'summary.json')
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(figsize=(7,4.5))
 for mode in ['attention','concat','separable']:
  r=[r for r in summary if r['mode']==mode];ax.plot([x['occlusion_percent'] for x in r],[100*x['joint'] for x in r],marker='o',label=mode)
 for method,label in [('fixed','Independent F1'),('acoustic','Acoustic only')]:
  r=[x for x in controls if x['method']==method];ax.plot([x['level'] for x in r],[100*x['joint'] for x in r],marker='s',linestyle='--',label=label)
 ax.set(xlabel='Center-image occlusion (%)',ylabel='Joint success (%)',xticks=[0,25,50,75],title='Frozen checkpoints: all declared occlusion levels')
 ax.legend();ax.grid(alpha=.25);fig.tight_layout();fig.savefig(a.out/'occlusion_comparison.png',dpi=160);plt.close(fig)
 for name,obj in [('paired',pairs),('controls',controls),('external_hashes',{x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in a.run.glob('*.json')})]:(a.out/f'{name}.json').write_text(json.dumps(obj,indent=2))
 print(json.dumps([r for r in pairs if r['mode']=='attention' and r['level']==75 and r['metric']=='joint']));print('COMPLETE')
if __name__=='__main__':main()

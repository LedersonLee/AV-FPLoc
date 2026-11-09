"""Paired final-versus-independent-initial control and query-audio intervention."""
import argparse,json
from pathlib import Path
import numpy as np

def read(p):return json.loads(p.read_text())
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 protocol=read(a.run/'protocol.json');ref=read(a.reference/'predictions.json');rp=read(a.reference/'protocol.json');assert rp['scene_ids']==protocol['data']['scenes']['val'] and rp['seeds']==protocol['seeds']
 assert rp['checkpoint_hashes']==protocol['pretrained_sha256']
 baseline={(r['seed'],r['scene'],r['query']):r for r in ref if r['method']=='fixed'};rng=np.random.default_rng(20261109);result=[]
 for mode in protocol['modes']:
  normal=[];shuffled=[]
  for seed in protocol['seeds']:
   r=read(a.run/f'{mode}_{seed}.json')['val']['rows']
   normal.extend([dict(seed=seed,**x) for x in r if x['arm']=='normal']);shuffled.extend([dict(seed=seed,**x) for x in r if x['arm']=='audio_shuffle'])
  assert {(r['seed'],r['scene'],r['query']) for r in normal}==set(baseline)
  shuffle={(r['seed'],r['scene'],r['query']):r for r in shuffled}
  for label,control in [('independent_initial',baseline),('audio_shuffle',shuffle)]:
   for metric in ['joint','fine','raw20','nll']:
    d=np.array([np.mean([r[metric]-control[r['seed'],r['scene'],r['query']][metric] for r in normal if r['scene']==s]) for s in rp['scene_ids']]);scale=1 if metric=='nll' else 100
    b=d[rng.integers(len(d),size=(10000,len(d)))].mean(1)*scale
    result.append(dict(mode=mode,control=label,metric=metric,delta=float(d.mean()*scale),ci95=np.quantile(b,[.025,.975]).tolist(),units='nats' if metric=='nll' else 'percentage points',sampling=f'{len(d)} paired scenes, seeds averaged'))
 (a.out/'paired.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));print('COMPLETE')
if __name__=='__main__':main()

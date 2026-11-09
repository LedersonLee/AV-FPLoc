"""Match exact uniform expectations to fitted model evaluation keys."""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
from track_b.scripts.evaluate_uniform_prior import inclusion

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--prior',type=Path,required=True);p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 checks=0
 for n in range(1,31):
  for k in range(n+1):
   d=min(n,20);expected=1-(math.comb(n-k,d)/math.comb(n,d) if n-k>=d else 0)
   assert abs(inclusion(n,k)-expected)<1e-12;checks+=1
 protocol=read(a.prior/'protocol.json');ep=read(a.evaluation/'protocol.json');assert protocol['cache_manifest_sha256']==ep['cache_manifest_sha256'] and protocol['scene_ids']==ep['scene_ids']
 rows=read(a.prior/'rows.json');rr=read(a.evaluation/'predictions.json');prior={(r['scene'],r['query']):r for r in rows};assert len(prior)==len(rows)==1216
 for method in ['visual','acoustic','fixed','fixed_audio_shuffle']:
  for seed in ep['seeds']:
   keys=[(r['scene'],r['query']) for r in rr if r['method']==method and r['seed']==seed];assert len(keys)==len(set(keys)) and set(keys)==set(prior)
 paired=[];rng=np.random.default_rng(20261109)
 for method in ['visual','acoustic','fixed','fixed_audio_shuffle']:
  for metric in ['joint','r1','fine','nll']:
   delta=np.array([np.mean([r[metric] for r in rr if r['method']==method and r['scene']==s])-np.mean([r[metric] for r in rows if r['scene']==s]) for s in protocol['scene_ids']]);scale=1 if metric=='nll' else 100;boot=delta[rng.integers(len(delta),size=(10000,len(delta)))].mean(1)*scale
   paired.append(dict(method=method,reference='exact uniform random pose expectation',metric=metric,delta=float(delta.mean()*scale),ci95=np.quantile(boot,[.025,.975]).tolist(),units='nats' if metric=='nll' else 'pp',unit='paired scene; fitted seeds averaged; no Monte Carlo randomness in prior'))
 a.out.mkdir(parents=True,exist_ok=False)
 (a.out/'paired.json').write_text(json.dumps(paired,indent=2));(a.out/'validation.json').write_text(json.dumps(dict(hypergeometric_checks=checks,queries=len(rows),scenes=len(protocol['scene_ids']),prior_rows_sha256=sha(a.prior/'rows.json'),evaluation_predictions_sha256=sha(a.evaluation/'predictions.json'),test_accessed=False),indent=2));print('COMPLETE',flush=True)
if __name__=='__main__':main()

"""Register a completed fixed scaling cell without choosing hyperparameters."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from track_b.eval.extended_metrics import reliability_statistics

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--pilot-cache',type=Path,required=True);p.add_argument('--condition',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 plan=read(a.root/'plan_v1/plan.json');condition=next(c for c in plan['conditions'] if c['name']==a.condition)
 setup=read(a.run/'protocol.json');summary=read(a.run/'summary.json');rows=read(a.run/'predictions.json')
 assert setup['fixed_only'] and setup['extended_metrics'] and read(a.run/'selection.json')=={} and summary['selection']=={}
 assert setup['scene_ids']==condition['val_scenes'] and setup['seeds']==plan['seeds']
 assert setup['cache_manifest_sha256']==sha(a.root/'cache_v1/manifest.json')
 assert setup['fixed']=={'visual_weight':.85,'temperature':1.}
 methods=['visual','acoustic','fixed','fixed_audio_shuffle'];assert set(summary['summary'])==set(methods)
 assert summary['unimodal_replay_checks']==19*3*64*2
 keys=[tuple(r[k] for k in ['scene','seed','query','method']) for r in rows];assert len(keys)==len(set(keys))==19*3*64*4
 base={(r['scene'],r['seed'],r['query']) for r in rows if r['method']=='visual'}
 for method in methods:assert {(r['scene'],r['seed'],r['query']) for r in rows if r['method']==method}==base
 for sid in setup['scene_ids']:
  for seed in setup['seeds']:assert len([k for k in base if k[0]==sid and k[1]==seed])==64
 old=read(a.pilot_cache/'protocol.json')['scenes']['val'];assert len(old)==6 and set(old)<=set(setup['scene_ids'])
 strata={'all19':setup['scene_ids'],'original6':old,'added13':[s for s in setup['scene_ids'] if s not in old]}
 metrics=['joint','r1','fine','raw20','diverse20','nll','entropy','raw5','raw10','strict_05_15','strict_02_10','normalized_entropy']
 stats=[];pairs=[];rng=np.random.default_rng(20261109)
 for stratum,scenes in strata.items():
  for method in methods:
   rr=[r for r in rows if r['scene'] in scenes and r['method']==method]
   stats.append(dict(stratum=stratum,method=method,scenes=len(scenes),cases=len(rr),mean={k:float(np.mean([r[k] for r in rr])) for k in metrics},per_seed={str(seed):{k:float(np.mean([r[k] for r in rr if r['seed']==seed])) for k in metrics} for seed in setup['seeds']},reliability=reliability_statistics(rr)))
  for left,right in [('fixed','visual'),('fixed','acoustic'),('fixed','fixed_audio_shuffle')]:
   for metric in ['joint','r1','fine','raw20','diverse20','nll']:
    delta=np.array([np.mean([r[metric] for r in rows if r['scene']==sid and r['method']==left])-np.mean([r[metric] for r in rows if r['scene']==sid and r['method']==right]) for sid in scenes]);scale=1 if metric=='nll' else 100;boot=delta[rng.integers(len(delta),size=(10000,len(delta)))].mean(1)*scale
    pairs.append(dict(stratum=stratum,left=left,right=right,metric=metric,delta=float(delta.mean()*scale),ci95=np.quantile(boot,[.025,.975]).tolist(),units='nats' if metric=='nll' else 'pp',unit='scene, seeds averaged'))
 a.out.mkdir(parents=True,exist_ok=False)
 for name,obj in [('stratified',stats),('paired',pairs),('validation',dict(condition=a.condition,rows=len(rows),unimodal_replay_checks=summary['unimodal_replay_checks'],plan_sha256=sha(a.root/'plan_v1/plan.json'),evaluation_protocol_sha256=sha(a.run/'protocol.json'),predictions_sha256=sha(a.run/'predictions.json'),strata=strata,test_accessed=False,evidence='Exploratory; added13 are historically exposed Val, not independent Test'))]:
  (a.out/f'{name}.json').write_text(json.dumps(obj,indent=2))
 print('COMPLETE',len(rows),flush=True)
if __name__=='__main__':main()

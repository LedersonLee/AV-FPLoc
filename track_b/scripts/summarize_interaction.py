"""Require all matched fits; verify common initial tensors and scene-paired outcomes."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
import torch

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--code-sha',required=True);a=ap.parse_args();protocol=read(a.run/'protocol.json');assert len(read(a.run/'summary.json'))==9,'Await all planned fits';a.out.mkdir(parents=True,exist_ok=False)
 initial=[];records=[];allrows=[];registry=[]
 for seed in protocol['seeds']:
  states={m:torch.load(a.run/f'{m}_{seed}_initial.pt',weights_only=True,map_location='cpu') for m in protocol['modes']};anchor=states[protocol['modes'][0]]
  for mode,state in states.items():
   assert set(state)==set(anchor) and all(torch.equal(v,state[k]) for k,v in anchor.items()),f'Initialization differs {mode}/{seed}'
   initial.append(dict(mode=mode,seed=seed,equal_to_anchor=True,sha256=sha(a.run/f'{mode}_{seed}_initial.pt')))
   rec=read(a.run/f'{mode}_{seed}.json');assert rec['history'][-1]['step']==protocol['steps']-1;assert rec['parameters']==2429659
   for arm,metrics in rec['val']['summary'].items():records.append(dict(mode=mode,seed=seed,arm=arm,**metrics))
   allrows += [dict(mode=mode,seed=seed,**r) for r in rec['val']['rows']]
   registry.append(dict(run_id=f'{mode}_{seed}',git_commit=a.code_sha,status='completed',train_queries=1024,val_queries=384,test_queries=0,steps=protocol['steps'],parameters=rec['parameters'],train_seconds=rec['training_seconds'],peak_vram_bytes=rec['peak_vram_bytes'],checkpoint_sha256=sha(a.run/f'{mode}_{seed}.pt')))
 rng=np.random.default_rng(20261109);scenes=protocol['data']['scenes']['val'];paired=[]
 for arm in ('normal','low_light','audio_shuffle'):
  for control in ('concat','separable'):
   for metric in ('joint','fine','raw20','nll'):
    delta=np.array([np.mean([r[metric] for r in allrows if r['scene']==sid and r['mode']=='attention' and r['arm']==arm])-np.mean([r[metric] for r in allrows if r['scene']==sid and r['mode']==control and r['arm']==arm]) for sid in scenes]);scale=1 if metric=='nll' else 100;boot=delta[rng.integers(0,len(scenes),(10000,len(scenes)))].mean(1)*scale
    paired.append(dict(arm=arm,left='attention',right=control,metric=metric,delta=float(delta.mean()*scale),ci95=np.quantile(boot,[.025,.975]).tolist(),units='nats' if metric=='nll' else 'percentage points',sampling='six scenes, seeds averaged within scene'))
 aggregate=[]
 for mode in protocol['modes']:
  for arm in ('normal','low_light','audio_shuffle'):
   rr=[r for r in records if r['mode']==mode and r['arm']==arm];aggregate.append(dict(mode=mode,arm=arm,**{k:float(np.mean([r[k] for r in rr])) for k in rr[0] if k not in ('mode','seed','arm')}))
 for filename,value in [('aggregate.json',aggregate),('paired_scene_bootstrap.json',paired),('initialization_validation.json',initial),('per_seed_summary.json',records)]: (a.out/filename).write_text(json.dumps(value,indent=2))
 with (a.out/'experiment_registry.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(registry[0]),lineterminator='\n');w.writeheader();w.writerows(registry)
 (a.out/'source_hashes.json').write_text(json.dumps({p.name:sha(p) for p in a.run.glob('*') if p.is_file()},indent=2));print(json.dumps(aggregate),flush=True)
if __name__=='__main__':main()

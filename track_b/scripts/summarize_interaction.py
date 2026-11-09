"""Require all matched fits; verify common initial tensors and scene-paired outcomes."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
import torch

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--code-sha',required=True,help='Aggregation code SHA, never substituted for training provenance');ap.add_argument('--historical-registry',type=Path);ap.add_argument('--expected-parameters',type=int,default=2429659);a=ap.parse_args();protocol=read(a.run/'protocol.json');assert len(read(a.run/'summary.json'))==9,'Await all planned fits';a.out.mkdir(parents=True,exist_ok=False)
 historical={}
 if a.historical_registry:
  historical_rows=list(csv.DictReader(a.historical_registry.open()))
  historical={r['run_id']:r for r in historical_rows}
  assert len(historical)==len(historical_rows), 'Duplicate historical run IDs'
 initial=[];records=[];allrows=[];registry=[];reference_keys=None
 for seed in protocol['seeds']:
  states={m:torch.load(a.run/f'{m}_{seed}_initial.pt',weights_only=True,map_location='cpu') for m in protocol['modes']};anchor=states[protocol['modes'][0]]
  for mode,state in states.items():
   assert set(state)==set(anchor) and all(torch.equal(v,state[k]) for k,v in anchor.items()),f'Initialization differs {mode}/{seed}'
   initial.append(dict(mode=mode,seed=seed,equal_to_anchor=True,sha256=sha(a.run/f'{mode}_{seed}_initial.pt')))
   rec=read(a.run/f'{mode}_{seed}.json');assert rec['history'][-1]['step']==protocol['steps']-1;assert rec['parameters']==a.expected_parameters
   run_id=f'{mode}_{seed}';checkpoint_hash=sha(a.run/f'{run_id}.pt');execution_sha=rec.get('execution_code_sha')
   if run_id in historical:
    old=historical[run_id];assert old['checkpoint_sha256']==checkpoint_hash, f'Historical checkpoint mismatch {run_id}'
    assert execution_sha is None or execution_sha==old['git_commit'], f'Conflicting execution SHA {run_id}'
    execution_sha=old['git_commit']
   assert execution_sha and len(execution_sha)==40 and all(c in '0123456789abcdef' for c in execution_sha), f'Missing valid training SHA {run_id}; supply hash-verified historical registry'
   rows=rec['val']['rows'];keys=[(r['arm'],r['scene'],r['query']) for r in rows]
   assert len(keys)==len(set(keys)), f'Duplicate evaluation rows {mode}/{seed}'
   normal={(r['scene'],r['query']) for r in rows if r['arm']=='normal'}
   scenes=protocol['data']['scenes']['val']
   assert {sid for sid,q in normal}==set(scenes)
   assert all(sum(s==sid for s,q in normal)==protocol['data']['queries_per_scene'] for sid in scenes)
   expected={(arm,sid,q) for arm in ('normal','low_light','audio_shuffle') for sid,q in normal}
   assert set(keys)==expected, f'Incomplete evaluation arms {mode}/{seed}'
   if reference_keys is None:reference_keys=expected
   assert set(keys)==reference_keys, f'Query cohort differs {mode}/{seed}'
   for arm,metrics in rec['val']['summary'].items():records.append(dict(mode=mode,seed=seed,arm=arm,**metrics))
   allrows += [dict(mode=mode,seed=seed,**r) for r in rec['val']['rows']]
   registry.append(dict(run_id=f'{mode}_{seed}',git_commit=execution_sha,timestamp='2026-11-09',method=mode,dataset_version='HSSD positive_pairs_v1',split_hash=hashlib.sha256(json.dumps(protocol['data'],sort_keys=True).encode()).hexdigest(),scene_count=sum(len(v) for v in protocol['data']['scenes'].values()),train_query_count=len(protocol['data']['scenes']['train'])*protocol['data']['queries_per_scene'],val_query_count=len(protocol['data']['scenes']['val'])*protocol['data']['queries_per_scene'],test_query_count=0,input_variant='frozen foundation RGB features + 6ch-STFT-IPD' if 'visual_cache_protocol_sha256' in protocol else 'RGB128x96+6ch-STFT-IPD',fp_variant='footprint+geometry+learned FP query',grid_xy=.25,yaw_bins=36,encoder_regime=rec.get('encoder_regime','joint'),pretrained_checkpoint_hash=json.dumps({k:v for k,v in protocol['pretrained_sha256'].items() if k.endswith(str(seed))}) if 'visual_cache_protocol_sha256' in protocol else protocol['pretrained_sha256'][str(seed)],parameter_count=rec['parameters'],trainable_parameter_count=rec.get('trainable_parameters',rec['parameters']),flops_est='NA not measured',gpu='RTX4090',seed=seed,train_steps=protocol['steps'],gpu_hours=rec['training_seconds']/3600,best_val_metric='NA no selection; scheduled final checkpoint',final_val_joint=rec['val']['summary']['normal']['joint'],test_metric='NA not evaluated',status='completed',note='Exploratory; common auxiliary pretraining cost reported separately; split_hash=canonical JSON of data protocol',train_seconds=rec['training_seconds'],peak_vram_bytes=rec['peak_vram_bytes'],checkpoint_sha256=sha(a.run/f'{mode}_{seed}.pt')))
 rng=np.random.default_rng(20261109);scenes=protocol['data']['scenes']['val'];paired=[]
 for arm in ('normal','low_light','audio_shuffle'):
  for control in ('concat','separable'):
   for metric in ('joint','fine','raw20','nll'):
    delta=np.array([np.mean([r[metric] for r in allrows if r['scene']==sid and r['mode']=='attention' and r['arm']==arm])-np.mean([r[metric] for r in allrows if r['scene']==sid and r['mode']==control and r['arm']==arm]) for sid in scenes]);scale=1 if metric=='nll' else 100;boot=delta[rng.integers(0,len(scenes),(10000,len(scenes)))].mean(1)*scale
    paired.append(dict(arm=arm,left='attention',right=control,metric=metric,delta=float(delta.mean()*scale),ci95=np.quantile(boot,[.025,.975]).tolist(),units='nats' if metric=='nll' else 'percentage points',sampling=f'{len(scenes)} scenes, seeds averaged within scene'))
 aggregate=[]
 for mode in protocol['modes']:
  for arm in ('normal','low_light','audio_shuffle'):
   rr=[r for r in records if r['mode']==mode and r['arm']==arm];aggregate.append(dict(mode=mode,arm=arm,**{k:float(np.mean([r[k] for r in rr])) for k in rr[0] if k not in ('mode','seed','arm')}))
 for filename,value in [('aggregate.json',aggregate),('paired_scene_bootstrap.json',paired),('initialization_validation.json',initial),('per_seed_summary.json',records)]: (a.out/filename).write_text(json.dumps(value,indent=2))
 with (a.out/'experiment_registry.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(registry[0]),lineterminator='\n');w.writeheader();w.writerows(registry)
 (a.out/'aggregation_provenance.json').write_text(json.dumps(dict(aggregation_code_sha=a.code_sha,historical_registry_sha256=sha(a.historical_registry) if a.historical_registry else None),indent=2))
 (a.out/'source_hashes.json').write_text(json.dumps({p.name:sha(p) for p in a.run.glob('*') if p.is_file()},indent=2));print(json.dumps(aggregate),flush=True)
if __name__=='__main__':main()

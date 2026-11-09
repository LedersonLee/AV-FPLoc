"""Register complete destructive controls; retain tie/roundoff diagnostics."""
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--training',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 rows=read(a.run/'predictions.json');assert len(rows)==17280
 assert read(a.run/'normal_replay_mismatches.json')==[],'Investigate numerical replay before interpreting controls'
 protocol=read(a.run/'protocol.json');tp=read(a.training/'protocol.json');a.out.mkdir(parents=True,exist_ok=False)
 registry=[];per_seed=[]
 for mode in tp['modes']:
  for seed in tp['seeds']:
   for condition in protocol['conditions']:
    rr=[r for r in rows if r['mode']==mode and r['seed']==seed and r['condition']==condition];assert len(rr)==384
    per_seed.append(dict(mode=mode,seed=seed,condition=condition,**{k:float(np.mean([r[k] for r in rr])) for k in ['joint','r1','fine','raw20','diverse20','nll']}))
    registry.append(dict(run_id=f'{mode}_{seed}_{condition}',git_commit=protocol['source_sha'],timestamp='2026-11-09',method=mode,dataset_version='HSSD positive_pairs_v1',split_hash=hashlib.sha256(json.dumps(tp['data'],sort_keys=True).encode()).hexdigest(),scene_count=6,train_query_count=0,val_query_count=384,test_query_count=0,input_variant=condition,fp_variant='cached learned dual-FP descriptors',grid_xy=.25,yaw_bins=36,encoder_regime='frozen checkpoint evaluation',pretrained_checkpoint_hash=sha(a.training/f'{mode}_{seed}.pt'),parameter_count=2834908,flops_est='NA',gpu='RTX4090',seed=seed,train_steps=0,gpu_hours='NA evaluation time not instrumented',best_val_metric='NA no selection',test_metric='NA',status='completed',note='destructive map evidence intervention; removed-field ranks are tie artifacts, not learned localization'))
 with (a.out/'experiment_registry.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(registry[0]),lineterminator='\n');w.writeheader();w.writerows(registry)
 spreads=read(a.run/'removed_field_spread.json');validation=dict(cases=len(rows),normal_replay_cases=3456,normal_replay_mismatches=0,max_removed_field_spread=float(max(spreads)),removed_field_semantics='constant candidate evidence; residual GPU differences and tie selection are not meaningful spatial evidence')
 for name,obj in [('per_seed',per_seed),('validation',validation),('external_hashes',{x.name:sha(x) for x in a.run.glob('*.json')})]:(a.out/f'{name}.json').write_text(json.dumps(obj,indent=2))
 print(json.dumps(validation));print('COMPLETE')
if __name__=='__main__':main()

"""All-grid paired comparison to strong F1 with identical scene/query/grid keys."""
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser()
 for k in ['run','reference','out']:p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();rows=read(a.run/'predictions.json');assert len(rows)==20736 and read(a.run/'baseline_replay_mismatches.json')==[]
 protocol=read(a.run/'protocol.json');rp=read(a.reference/'protocol.json');assert protocol['grid_manifest_sha256']==rp['grid_manifest_sha256']
 reference={(r['seed'],r['scene'],r['query'],r['variant']):r for r in read(a.reference/'predictions.json') if r['method']=='fixed'}
 a.out.mkdir(parents=True,exist_ok=False);rng=np.random.default_rng(20261109);pairs=[];registry=[]
 for mode in ['attention','concat','separable']:
  assert {(r['seed'],r['scene'],r['query'],r['variant']) for r in rows if r['mode']==mode}==set(reference)
  for variant,step,yaws,erosion in protocol['grid_protocol']['variants']:
   rr=[r for r in rows if r['mode']==mode and r['variant']==variant]
   for metric in ['joint','fine','raw20','diverse20','nll']:
    delta=np.array([np.mean([r[metric]-reference[r['seed'],r['scene'],r['query'],variant][metric] for r in rr if r['scene']==s]) for s in protocol['grid_protocol']['scenes']]);scale=1 if metric=='nll' else 100;b=delta[rng.integers(len(delta),size=(10000,len(delta)))].mean(1)*scale
    pairs.append(dict(mode=mode,variant=variant,control='fixed independent F1',metric=metric,delta=float(delta.mean()*scale),ci95=np.quantile(b,[.025,.975]).tolist(),units='nats' if metric=='nll' else 'pp',unit='scene; seeds averaged'))
   for seed in [42,43,44]:
    registry.append(dict(run_id=f'{mode}_{seed}_{variant}',git_commit=protocol['code_sha'],timestamp='2026-11-09',method=mode,dataset_version='HSSD positive_pairs_v1',split_hash=sha(a.run/'protocol.json'),scene_count=6,train_query_count=0,val_query_count=384,test_query_count=0,input_variant='fixed RGB+6chSTFT-IPD',fp_variant=variant,grid_xy=step,yaw_bins=yaws,encoder_regime='frozen scheduled joint checkpoint',pretrained_checkpoint_hash=protocol['checkpoint_hashes'][f'{mode}_{seed}'],parameter_count=2834908,flops_est='NA',gpu='RTX4090',seed=seed,train_steps=0,gpu_hours='NA evaluation not timed',best_val_metric='NA no grid selection',test_metric='NA',status='completed',note='same six grid variants as independent F1; all-query/valid-subset; no retraining'))
 with (a.out/'experiment_registry.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(registry[0]),lineterminator='\n');w.writeheader();w.writerows(registry)
 for name,obj in [('paired_vs_f1',pairs),('external_hashes',{x.name:sha(x) for x in a.run.glob('*.json')}),('validation',dict(cases=len(rows),normal_replay_cases=3456,normal_replay_mismatches=0,reference_query_keys_equal=True,reference_protocol_sha256=sha(a.reference/'protocol.json')))]:
  (a.out/f'{name}.json').write_text(json.dumps(obj,indent=2))
 print('COMPLETE')
if __name__=='__main__':main()

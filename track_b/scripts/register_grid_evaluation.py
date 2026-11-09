"""Archive a completed fixed-checkpoint sensitivity run with per-seed provenance."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--code-sha',required=True);a=ap.parse_args();protocol=read(a.run/'protocol.json');rows=read(a.run/'predictions.json');assert len(rows)==20736;a.out.mkdir(parents=True,exist_ok=False);registry=[];perseed=[]
 for name,step,yaws,erosion in protocol['grid_protocol']['variants']:
  for seed in protocol['seeds']:
   for method in ['visual','acoustic','fixed']:
    rr=[r for r in rows if r['variant']==name and r['seed']==seed and r['method']==method];assert len(rr)==384
    metrics={k:float(np.mean([r[k] for r in rr])) for k in ['joint','r1','fine','raw20','diverse20','nll','oracle_joint','oracle_fine']};perseed.append(dict(variant=name,seed=seed,method=method,**metrics));registry.append(dict(run_id=f'{name}/{method}/{seed}',git_commit=a.code_sha,timestamp='2026-11-09',method=method,dataset_version='HSSD positive_pairs_v1',split_hash=protocol['grid_protocol']['reference_protocol_sha256'],scene_count=6,train_query_count=0,val_query_count=384,test_query_count=0,input_variant='unchanged normal RGB foundation features and observed6ch STFT/IPD',fp_variant=f'original FP; validity clearance {erosion}m',grid_xy=step,yaw_bins=yaws,encoder_regime='frozen inference; independent anchored visual/acoustic checkpoints',pretrained_checkpoint_hash=json.dumps({k:v for k,v in protocol['checkpoint_hashes'].items() if k.endswith(str(seed))}),parameter_count='See source training registry; no new parameters',flops_est='NA',gpu='RTX4090',seed=seed,train_steps=0,gpu_hours='UNRECORDED evaluation wall time',best_val_metric='No selection or grid-specific calibration',test_metric='NA',status='completed',note='All-query; invalid grid membership flagged; source weights unchanged; partial map timing is not total inference cost',final_val_joint=metrics['joint']))
 with (a.out/'experiment_registry.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(registry[0]),lineterminator='\n');w.writeheader();w.writerows(registry)
 (a.out/'per_seed_summary.json').write_text(json.dumps(perseed,indent=2));(a.out/'external_hashes.json').write_text(json.dumps({p.name:sha(p) for p in a.run.glob('*.json')},indent=2));print('COMPLETE')
if __name__=='__main__':main()

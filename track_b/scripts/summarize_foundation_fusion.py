"""Report normal/darkened independent fusion without selecting on shifted data."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--normal-code',required=True);ap.add_argument('--shift-code',required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False);summaries={};pred={};registry=[];hashes={}
 for variant in ['anchor','frozen']:
  for condition in ['normal','lowlight']:
   run=f'independent_fusion_{variant}'+('_lowlight' if condition=='lowlight' else '')+'_v1';path=a.root/run;summary=read(path/'summary.json');protocol=read(path/'protocol.json');rows=read(path/'predictions.json');assert len(rows)==6912;assert summary['unimodal_replay_checks']==(2304 if condition=='normal' else 1152);summaries[run]=summary;pred[variant,condition]=rows
   for p in path.glob('*.json'):hashes[str(p.relative_to(a.root))]=sha(p)
   for seed in [42,43,44]:
    registry.append(dict(run_id=f'{run}/{seed}',git_commit=a.normal_code if condition=='normal' else a.shift_code,timestamp='2026-11-09',method='independent full-field visual/acoustic fusion',dataset_version='HSSD positive_pairs_v1',split_hash=protocol['cache_manifest_sha256'],scene_count=6,train_query_count=0,val_query_count=384,test_query_count=0,input_variant=condition,fp_variant='independent learned visual/acoustic FP descriptors',grid_xy=.25,yaw_bins=36,encoder_regime='inference frozen; visual training '+variant,pretrained_checkpoint_hash=json.dumps({k:v for k,v in protocol['checkpoint_hashes'].items() if k.endswith(str(seed))}),parameter_count='See source models; not matched capacity comparison',flops_est='NA',gpu='RTX4090',seed=seed,train_steps=0,gpu_hours='UNRECORDED',best_val_metric='No checkpoint selection; other-five-scene calibration on normal only',test_metric='NA',status='completed',note='Exploratory, reused development and upstream exposure; no RIR reference bank'))
 rng=np.random.default_rng(20261109);paired=[]
 for variant in ['anchor','frozen']:
  normal=pred[variant,'normal'];dark=pred[variant,'lowlight'];scenes=sorted({r['scene'] for r in normal})
  for method in ['visual','acoustic','fixed','calibrated']:
   for metric in ['joint','fine','raw20','diverse20']:
    d=np.array([np.mean([r[metric] for r in dark if r['scene']==s and r['method']==method])-np.mean([r[metric] for r in normal if r['scene']==s and r['method']==method]) for s in scenes]);boot=d[rng.integers(0,len(d),(10000,len(d)))].mean(1)*100;paired.append(dict(variant=variant,method=method,metric=metric,lowlight_minus_normal_pp=float(d.mean()*100),ci95_pp=np.quantile(boot,[.025,.975]).tolist(),unit='six scenes; seeds averaged'))
 for name,data in [('summary',summaries),('paired_brightness',paired),('external_hashes',hashes)]:(a.out/f'{name}.json').write_text(json.dumps(data,indent=2))
 with (a.out/'experiment_registry.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(registry[0]),lineterminator='\n');w.writeheader();w.writerows(registry)
 print(json.dumps({k:v['summary'] for k,v in summaries.items()}));print('COMPLETE')
if __name__=='__main__':main()

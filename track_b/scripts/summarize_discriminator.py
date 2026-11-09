"""Paired scene uncertainty, all completed fits and external artifact hashes."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from track_b.scripts.train_structure import read,sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(exist_ok=False,parents=True)
 rows=read(a.root/'discriminator_v1/predictions.json');scenes=sorted({r['scene'] for r in rows});rng=np.random.default_rng(20261109);pairs=[]
 for policy in ['raw20','diverse20']:
  for metric in ['joint','fine','r1']:
   delta=np.array([np.mean([r[metric] for r in rows if r['scene']==s and r['policy']==policy and r['method']=='geometry'])-np.mean([r[metric] for r in rows if r['scene']==s and r['policy']==policy and r['method']=='score_statistics_only']) for s in scenes])
   b=delta[rng.integers(len(delta),size=(10000,len(delta)))].mean(1)*100
   pairs.append(dict(policy=policy,metric=metric,left='geometry',right='score_statistics_only',delta_pp=float(delta.mean()*100),ci95_pp=np.quantile(b,[.025,.975]).tolist(),unit='scene, seeds averaged'))
 (a.out/'geometry_ablation_paired.json').write_text(json.dumps(pairs,indent=2))
 records=[]
 for d in sorted(a.root.glob('fold[01]_*_v1')):
  if not any(stage in d.name for stage in ['visual_pretrain','visual_joint','acoustic']):continue
  protocol=read(d/'protocol.json');rr=read(d/'summary.json');assert len(rr)==3
  for r in rr:
   path=d/f"{r['mode']}_{r['seed']}.pt";detail=read(d/f"{r['mode']}_{r['seed']}.json")
   assert detail['history'][-1]['step']==protocol['aux_steps']+protocol['dense_steps']-1
   records.append(dict(run_id=d.name+f"_{r['seed']}",git_commit='0aedb2ad9cb4690501c7f8ff0e648630241f983b',timestamp='2026-11-09',method=r['mode'],dataset_version='HSSD positive_pairs_v1',split_hash=sha(d/'protocol.json'),scene_count=16,train_query_count=512,val_query_count=512,test_query_count=0,input_variant='foundation RGB' if r['mode']=='visual' else '6ch STFT-IPD',fp_variant='footprint rays+trainable FP',grid_xy=.25,yaw_bins=36,encoder_regime='joint',pretrained_checkpoint_hash=json.dumps(protocol.get('pretrained_sha256',{})),parameter_count=r['parameters'],flops_est='NA',gpu='RTX4090',seed=r['seed'],train_steps=protocol['aux_steps']+protocol['dense_steps'],gpu_hours=r['training_seconds']/3600,best_val_metric='NA scheduled final; no selection',test_metric='NA',status='completed',note='OOF source training; frozen upstream not pristine',checkpoint_sha256=sha(path)))
 fits=read(a.root/'discriminator_v1/fits.json');assert len(fits)==12
 for r in fits:
  records.append(dict(run_id=r['run_id'],git_commit=r['code_sha'],timestamp='2026-11-09',method=r['arm'],dataset_version='HSSD positive_pairs_v1',split_hash=sha(a.root/'discriminator_v1/protocol.json'),scene_count=22,train_query_count=1024,val_query_count=384,test_query_count=0,input_variant='OOF '+r['policy']+' 133dim',fp_variant='frozen independent full-field candidates',grid_xy=.25,yaw_bins=36,encoder_regime='frozen upstream; new classifier',pretrained_checkpoint_hash='fresh classifier; scorer hashes in candidate protocols',parameter_count=r['parameters'],flops_est='NA',gpu='CPU',seed=r['seed'],train_steps=1000,gpu_hours=0,best_val_metric='NA scheduled final; no selection',test_metric='NA',status='completed',note='CPU seconds='+str(r['train_seconds'])+'; no-positive queries='+str(r['train_no_positive_queries']),checkpoint_sha256=r['checkpoint_sha256']))
 with (a.out/'experiment_registry.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(records[0]),lineterminator='\n');w.writeheader();w.writerows(records)
 hashes={str(p.relative_to(a.root)):sha(p) for p in a.root.rglob('*') if p.is_file() and a.out not in p.parents}
 (a.out/'external_hashes.json').write_text(json.dumps(hashes,indent=2));print(json.dumps(pairs));print('COMPLETE')

if __name__=='__main__':main()

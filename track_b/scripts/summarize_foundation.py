"""Complete foundation visual comparisons with scene-paired uncertainty."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--scratch',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--initial-code',required=True);ap.add_argument('--matched-code',required=True);ap.add_argument('--controls-code');ap.add_argument('--anchor-code');a=ap.parse_args();a.out.mkdir(exist_ok=False,parents=True);series={};summaries=[];registry=[]
 runs=['visual_supervised_v1','matched_frozen_v1','matched_partial_v1','matched_joint_v1']
 if a.controls_code:runs+=['fixed_fp_w10_v1','learned_fp_w1000_v1','fixed_fp_w1000_v1']
 if a.anchor_code:
  assert a.controls_code
  runs+=['learned_fp_w1000_anchor1000_v1']
 for run in runs:
  protocol=read(a.root/run/'protocol.json');assert len(read(a.root/run/'summary.json'))==3
  for seed in [42,43,44]:
   rec=read(a.root/run/f'visual_{seed}.json');assert rec['history'][-1]['step']==protocol['aux_steps']+protocol['dense_steps']-1;rows=rec['val']['rows'];assert len(rows)==384;series.setdefault(run,[]).extend(dict(seed=seed,**r) for r in rows);summaries.append(dict(run=run,seed=seed,**rec['val']['summary']['normal']))
   registry.append(dict(run_id=f'{run}/visual_{seed}',git_commit=a.initial_code if run=='visual_supervised_v1' else (a.anchor_code if run=='learned_fp_w1000_anchor1000_v1' else a.controls_code if run in runs[4:] else a.matched_code),timestamp='2026-11-09',method='foundation visual structure matching',dataset_version='HSSD positive_pairs_v1',split_hash=protocol['cache_protocol_sha256'],scene_count=22,train_query_count=1024,val_query_count=384,test_query_count=0,input_variant='upstream RGB features + observed RIR unused by visual score',fp_variant='fixed geometry' if protocol.get('fixed_fp',False) else 'geometry + learned residual',grid_xy=.25,yaw_bins=36,encoder_regime=protocol['encoder_regime'],pretrained_checkpoint_hash=protocol.get('pretrained_sha256',{}).get(f'visual_{seed}','upstream hash in feature protocol; adapter scratch'),parameter_count=rec['parameters'],flops_est='NA',gpu='RTX4090',seed=seed,train_steps=protocol['aux_steps']+protocol['dense_steps'],gpu_hours=rec['training_seconds']/3600,best_val_metric='NA scheduled final, no selection',test_metric='NA',status='completed',note='Nominal params include unused acoustic branch, exclude frozen upstream encoder; training time excludes feature generation',checkpoint_sha256=sha(a.root/run/f'visual_{seed}.pt')))
 for seed in [42,43,44]:
  old=read(a.scratch/f'visual_{seed}.json');series.setdefault('scratch',[]).extend(dict(seed=seed,**r) for r in old['val']['rows'] if r['arm']=='normal')
  aux=read(a.root/f'visual_supervised_v1/visual_{seed}_aux_eval.json');series.setdefault('aux_only_raw_geometry',[]).extend(dict(seed=seed,**r) for r in aux['rows'])
 aggregate=[]
 for name,rows in series.items():aggregate.append(dict(run=name,**{k:float(np.mean([r[k] for r in rows])) for k in ['joint','r1','fine','raw20','nll','ray_mse','ray_mae_m']},diverse20=float(np.mean([r['diverse20'] for r in rows])) if rows[0]['diverse20'] is not None else None))
 rng=np.random.default_rng(20261109);paired=[]
 pairs=[('visual_supervised_v1','scratch'),('matched_partial_v1','matched_joint_v1'),('matched_frozen_v1','matched_joint_v1')]
 if a.controls_code:pairs += [('fixed_fp_w10_v1','matched_joint_v1'),('learned_fp_w1000_v1','matched_joint_v1'),('fixed_fp_w1000_v1','fixed_fp_w10_v1'),('fixed_fp_w1000_v1','learned_fp_w1000_v1'),('learned_fp_w1000_v1','matched_frozen_v1')]
 if a.anchor_code:pairs += [('learned_fp_w1000_anchor1000_v1','learned_fp_w1000_v1'),('learned_fp_w1000_anchor1000_v1','fixed_fp_w1000_v1'),('learned_fp_w1000_anchor1000_v1','matched_frozen_v1')]
 for left,right in pairs:
  for metric in ['joint','fine','raw20']:
   scenes=sorted({r['scene'] for r in series[left]});d=np.array([np.mean([r[metric] for r in series[left] if r['scene']==s])-np.mean([r[metric] for r in series[right] if r['scene']==s]) for s in scenes]);boot=d[rng.integers(0,6,(10000,6))].mean(1)*100;paired.append(dict(left=left,right=right,metric=metric,delta_pp=float(d.mean()*100),ci95_pp=np.quantile(boot,[.025,.975]).tolist(),unit='six scenes; seeds averaged'))
 for name,data in [('aggregate.json',aggregate),('paired_scene_bootstrap.json',paired),('per_seed_summary.json',summaries)]: (a.out/name).write_text(json.dumps(data,indent=2))
 with (a.out/'experiment_registry.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(registry[0]),lineterminator='\n');w.writeheader();w.writerows(registry)
 (a.out/'external_hashes.json').write_text(json.dumps({str(p.relative_to(a.root)):sha(p) for p in a.root.rglob('*') if p.is_file() and p.suffix in ('.json','.pt')},indent=2));print(json.dumps(aggregate))
if __name__=='__main__':main()

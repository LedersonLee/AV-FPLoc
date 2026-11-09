"""Archive small summaries and scene-paired uncertainty without uploading observations."""
import argparse,csv,hashlib,json,subprocess
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 commits={'supervised_v1':'fb6900d','shuffled_v1':'efcb0cb','dense_only_v1':'efcb0cb','frozen_v1':'38f76f6','partial_v1':'38f76f6','matched_frozen_v1':'c20ce07','matched_partial_v1':'c20ce07','matched_joint_v1':'c20ce07'}
 full={k:subprocess.check_output(['git','rev-parse',v],text=True).strip() for k,v in commits.items()};rows=[];cells={};registry=[]
 for run,commit in full.items():
  p=a.root/run;protocol=read(p/'protocol.json');summary=read(p/'summary.json');assert len(summary)==(9 if run=='supervised_v1' else 3)
  for rec in summary:
   mode=rec['mode'];seed=rec['seed'];raw=read(p/f'{mode}_{seed}.json');normal=[r for r in raw['val']['rows'] if r['arm']=='normal'];key=f'{run}/{mode}';cells.setdefault(key,{})[seed]={s:np.mean([r['joint'] for r in normal if r['scene']==s]) for s in sorted({r['scene'] for r in normal})}
   rows.append(dict(method=key,seed=seed,**rec['val']['normal'],train_joint=rec['train']['normal']['joint'],training_seconds=rec['training_seconds'],peak_vram_bytes=rec['peak_vram_bytes']))
   registry.append(dict(run_id=f'{key}_{seed}',git_commit=commit,timestamp='2026-11-09',method=mode,dataset_version='HSSD observed positive_pairs_v1',split_hash=protocol['cache_protocol_sha256'],scene_count=22,train_query_count=1024,val_query_count=384,test_query_count=0,input_variant='RGB128x96+6ch-STFT-IPD16384samples',fp_variant='footprint+16rays+learned-residual',grid_xy=.25,yaw_bins=36,encoder_regime=protocol.get('encoder_regime','joint'),pretrained_checkpoint_hash=protocol.get('pretrained_sha256',{}).get(f'{mode}_{seed}','scratch-warmup'),parameter_count=rec['parameters'],flops_est='not measured',gpu='RTX4090',seed=seed,train_steps=protocol['aux_steps']+protocol['dense_steps'],gpu_hours=rec['training_seconds']/3600,best_val_metric='not selected; final step',test_metric='not measured',status='completed',note='Exploratory',checkpoint_sha256=sha(p/f'{mode}_{seed}.pt')))
 for name,data in [('results_master.csv',rows),('experiment_registry.csv',registry)]:
  with (a.out/name).open('w') as f:
   w=csv.DictWriter(f,fieldnames=list(data[0]),lineterminator="\n");w.writeheader();w.writerows(data)
 rng=np.random.default_rng(20261109);ci=[]
 comparisons=[('supervised_v1/fusion','dense_only_v1/fusion'),('supervised_v1/fusion','shuffled_v1/fusion'),('supervised_v1/fusion','supervised_v1/visual'),('matched_partial_v1/fusion','matched_joint_v1/fusion'),('matched_frozen_v1/fusion','matched_joint_v1/fusion')]
 for left,right in comparisons:
  diff=np.array([[cells[left][seed][s]-cells[right][seed][s] for s in cells[left][seed]] for seed in (42,43,44)]).mean(0)
  boot=diff[rng.integers(0,len(diff),size=(10000,len(diff)))].mean(1)*100
  ci.append(dict(left=left,right=right,delta_pp=100*diff.mean(),ci95_pp=np.quantile(boot,[.025,.975]).tolist(),scene_deltas_pp=(100*diff).tolist(),unit='six scenes; seeds averaged within each scene, not independent n'))
 (a.out/'paired_scene_bootstrap.json').write_text(json.dumps(ci,indent=2))
 aggregate=[]
 for key in cells:
  selected=[r for r in rows if r['method']==key];aggregate.append(dict(method=key,**{k:float(np.mean([r[k] for r in selected])) for k in selected[0] if k not in ('method','seed')}))
 (a.out/'aggregate.json').write_text(json.dumps(aggregate,indent=2))
 for source,dest in [('cache/protocol.json','data_protocol.json'),('cache/manifest.json','cache_manifest.json'),('profile.json','profile.json'),('extended_v1/summary.json','extended_summary.json'),('unloc_reference_v1/summary.json','unloc_summary.json'),('unloc_reference_v1/provenance.json','unloc_source.json')]:
  (a.out/dest).write_bytes((a.root/source).read_bytes())
 correction=read(a.root/'corrected_ray_diagnostics.json');correction.pop('rows',None);(a.out/'ray_correction.json').write_text(json.dumps(correction,indent=2))
 external={str(p.relative_to(a.root)):sha(p) for p in a.root.rglob('*') if p.is_file() and p.suffix in ('.json','.pt')}
 (a.out/'provenance.json').write_text(json.dumps(dict(code_commits=full,external_hashes=external,image_id='sha256:e51cb712100f25d0018aa65e371e63deaa51d12601c465ce698c0c446800c318',files=[]),indent=2))
if __name__=='__main__':main()

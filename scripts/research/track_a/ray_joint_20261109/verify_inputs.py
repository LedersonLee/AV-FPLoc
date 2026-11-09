from pathlib import Path
import json,hashlib
P=Path(__file__).resolve().parent;W=P.parent.parent;PR=json.loads((P/'PROTOCOL.json').read_text());H=Path('/file2/jeongeon/AV-FPLoc/datasets/HSSD')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def main():
 files=[];raw=0;zero=[]
 entries={r['scene_id']:r for r in json.loads((H/'ReliabilityFusion/expansions/train36_val8_test3_v1/bank_index.json').read_text())['scenes']}
 assert all(entries[s]['split']=='train' for s in PR['training_scenes']+PR['transfer_scenes'])
 for sid in PR['training_scenes']+PR['transfer_scenes']:
  receipt=json.loads((P/'audio'/sid/'receipt.json').read_text());assert sha(P/'audio'/sid/'features.npz')==receipt['feature_sha256'];assert sha(P/'chirp.npy')==receipt['chirp_sha256']
  for row in receipt['source']:
   path=Path(row['path']);assert sha(path)==row['sha256'];raw+=1
   if row.get('zero_tail_samples',0):zero.append(dict(scene=sid,query_id=row['query_id'],original_samples=row['original_samples'],zero_tail_samples=row['zero_tail_samples']))
  targets=[W/'experiments/directional_structure_20261109/candidates'/sid/'bank_geometry.npz',W/'experiments/hssd_scaling_20261108_1332/inputs'/sid/'windows.npy']
  for seed in PR['seeds']:
   targets.append(W/'experiments/directional_tribranch_20261109/data'/sid/f'candidates_seed{seed}.npz')
   if sid in PR['training_scenes']:
    fold=next(i for i,f in enumerate(PR['outer_folds']) if sid in f);targets.append(W/'experiments/directional_structure_20261109/discrimination'/f'pure_visual_seed{seed}_fold{fold}'/f'{sid}.npz')
  for path in targets:
   if path.exists():files.append(dict(path='${WORKSPACE}/'+str(path.relative_to(W)),bytes=path.stat().st_size,sha256=sha(path)))
 cfg=json.loads((H/'ReliabilityFusion/expansions/train36_val8_test3_v1/config.json').read_text());cp=H/cfg['acoustic_checkpoint'];assert sha(cp)==cfg['acoustic_sha256'];files.append(dict(path='${HSSD_ROOT}/'+str(cp.relative_to(H)),sha256=sha(cp),bytes=cp.stat().st_size))
 result=dict(raw_RIR_hashes_verified=raw,raw_source_metadata='Local audio/<scene>/receipt.json; receipt hashes are in input_inventory.json.',feature_hashes_verified=23,chirp_hashes_verified=23,zero_tail_queries=zero,additional_dependencies=files,scope='Only explicitly allowlisted training-source scenes. Reading config paths containing test3 is metadata, not Test observations/results.');(P/'input_verification.json').write_text(json.dumps(result,indent=2));print('Verified',raw,'rawRIRs,',len(files),'additional dependencies,',len(zero),'zero-tail inputs')
if __name__=='__main__':main()

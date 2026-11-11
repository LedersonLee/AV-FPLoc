"""Validate historical full-query baseline caches against audited AVLoc Val19 cohort."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--study',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 data=json.loads((a.study/'data_v1/manifest.json').read_text());rows=[]
 for r in data['records']:
  if r['split']!='val':continue
  sid=r['scene'];choices=[a.workspace/'experiments/mbt_remaining_suite_20261106/data'/sid,a.workspace/'experiments/fusion_vs_posterior_20261107_1300/independent_data'/sid];folder=next((p for p in choices if (p/'receipt.json').exists()),None)
  assert folder is not None,('Missing full baseline cache',sid);rec=json.loads((folder/'receipt.json').read_text());start=time.time()
  for name,digest in rec['files'].items():assert sha(folder/name)==digest,(sid,name)
  q=np.load(folder/'query.npz');lab=np.load(a.study/'data_v1'/sid/'labels.npz');assert len(q['sample_indices'])==r['queries'];np.testing.assert_array_equal(q['sample_indices'],lab['query_ids']);np.testing.assert_allclose(q['truth'],lab['truth'],atol=1e-6,rtol=0)
  bank=np.load(a.study/'banks_v1'/sid/'grid_20cm.npz');xy=np.load(folder/'xy.npy');np.testing.assert_allclose(xy,bank['xy'],atol=1e-7,rtol=0)
  fp=np.load(folder/'fp.npy',mmap_mode='r');windows=np.load(folder/'windows.npy',mmap_mode='r');assert fp.shape==(len(xy)*36,128) and windows.shape==(len(xy)*36,11)
  for key in q.files:assert np.isfinite(q[key]).all(),(sid,key)
  norms=np.linalg.norm(fp,axis=1);obsnorms=np.linalg.norm(q['observed'],axis=1);normerr=max(float(abs(norms-1).max()),float(abs(obsnorms-1).max()));assert normerr<1e-5
  assert np.isfinite(windows).all()
  rows.append(dict(scene=sid,queries=r['queries'],poses=len(fp),folder_relative=str(folder.relative_to(a.workspace)),receipt_sha256=sha(folder/'receipt.json'),files=rec['files'],pose_and_query_ids_match=True,grid_match=True,max_embedding_norm_error=normerr,array_shapes={k:list(q[k].shape) for k in q.files},seconds=time.time()-start));print('BASELINE CACHE',sid,r['queries'],flush=True)
 assert len(rows)==19 and sum(r['queries'] for r in rows)==9965
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(dict(complete=True,scenes=19,queries=9965,records=rows,source_sha256=sha(__file__),input_manifest_sha256=sha(a.study/'data_v1/manifest.json'),test_accessed=False,limitation='Historical cached-feature provenance preserved; this audit is not raw-image/RIR encoder re-execution.20cm only;10cm acoustic bank is not synthesized.'),indent=2)+'\n')
if __name__=='__main__':main()

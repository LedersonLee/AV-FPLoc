"""Frozen-distribution fusion controls. No GT input, fitting, or candidate dependencies."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from matching import FOV

def equal_fusion(vision,audio,joint=None):
 v,vb=[np.asarray(x,dtype=np.float64) for x in vision];a,ab=[np.asarray(x,dtype=np.float64) for x in audio];assert v.shape==vb.shape==(len(a),11) and a.shape==ab.shape==(len(a),36)
 experts=[(a,ab)] if joint is None else [(a,ab),tuple(np.asarray(x,dtype=np.float64) for x in joint)]
 for m,b in [(v,vb),*experts]:assert m.shape==b.shape and np.isfinite(m).all() and np.isfinite(b).all() and (b>0).all()
 if joint is not None:assert experts[1][0].shape==a.shape
 mu=np.empty_like(a);scale=np.empty_like(a);weights=np.zeros((len(a),36,3 if joint is not None else 2))
 for k in range(36):
  means=[m[:,k] for m,b in experts];bs=[b[:,k] for m,b in experts]
  if k in FOV:means.insert(0,v[:,int(np.flatnonzero(FOV==k)[0])]);bs.insert(0,vb[:,int(np.flatnonzero(FOV==k)[0])]);weights[:,k,:]=1/len(means)
  else:weights[:,k,1:]=1/len(means)
  m=np.stack(means,-1);b=np.stack(bs,-1);mean=m.mean(-1);variance=(2*b*b+(m-mean[:,None])**2).mean(-1);mu[:,k]=mean;scale[:,k]=np.sqrt(variance/2)
 return mu.astype('float32'),scale.astype('float32'),weights.astype('float32')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['independent_equal','A3_uniform'],required=True);p.add_argument('--first',type=Path,required=True);p.add_argument('--second',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args();m=json.loads((a.first/'manifest.json').read_text());assert m['complete'] and m['queries']==9965 and m['split']=='val'
 if a.mode=='independent_equal':
  assert m['arm']=='A0' and a.second is not None;n=json.loads((a.second/'manifest.json').read_text());assert n['complete'] and n['arm']=='A1' and n['seed']==m['seed'] and n['input_manifest_sha256']==m['input_manifest_sha256'] and n['queries']==9965 and n['split']=='val';other={r['scene']:r for r in n['records']};assert set(other)=={r['scene'] for r in m['records']}
 else:assert m['arm']=='A3' and a.second is None
 a.out.mkdir(parents=True,exist_ok=False);records=[]
 for r in m['records']:
  path=a.first/r['file'];assert sha(path)==r['sha256'];z=np.load(path);v=(z['V_mu'],z['V_scale'])
  if a.mode=='independent_equal':
   rr=other[r['scene']];path=a.second/rr['file'];assert sha(path)==rr['sha256'];zz=np.load(path);np.testing.assert_array_equal(z['query_ids'],zz['query_ids']);audio=(zz['A_mu'],zz['A_scale']);joint=None
  else:audio=(z['A_mu'],z['A_scale']);joint=(z['AV_mu'],z['AV_scale'])
  mu,b,w=equal_fusion(v,audio,joint);assert len(mu)==r['queries'];dest=a.out/r['file'];np.savez_compressed(dest,query_ids=z['query_ids'],F_mu=mu,F_scale=b,weights=w);records.append(dict(scene=r['scene'],queries=len(mu),file=dest.name,sha256=sha(dest)))
 assert len(records)==19 and sum(r['queries'] for r in records)==9965
 sources=[dict(arm=m['arm'],manifest_sha256=sha(a.first/'manifest.json'))]
 if a.second:sources.append(dict(arm=n['arm'],manifest_sha256=sha(a.second/'manifest.json')))
 (a.out/'manifest.json').write_text(json.dumps(dict(complete=True,split='val',arm=a.mode,seed=m['seed'],scenes=19,queries=9965,records=records,input_manifest_sha256=m['input_manifest_sha256'],source_predictions=sources,source_sha256=sha(__file__),policy='equal mixture moment-matched to Laplace; no V outside FoV',scope='post-hoc inference intervention with fixed learned distributions; not training with fixed router',test_accessed=False),indent=2)+'\n')
if __name__=='__main__':main()

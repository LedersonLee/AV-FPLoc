"""GT-free 2x2 decomposition of fixed fusion: FoV mixture versus outside audio."""
import argparse,json
from pathlib import Path
import numpy as np
from matching import FOV
from predict_hash import sha


def decompose(v,vb,a,ab,f,fb):
 v,vb,a,ab,f,fb=[np.asarray(x) for x in (v,vb,a,ab,f,fb)]
 n=len(a)
 assert v.shape==vb.shape==(n,11) and a.shape==ab.shape==f.shape==fb.shape==(n,36)
 assert all(np.isfinite(x).all() for x in [v,vb,a,ab,f,fb]) and all((x>0).all() for x in [vb,ab,fb])
 mu=a.copy();scale=ab.copy();mu[:,FOV]=v;scale[:,FOV]=vb
 return {'visual_inside_audio_outside':{'F_mu':mu,'F_scale':scale},'fused_fov':{'V_mu':f[:,FOV].copy(),'V_scale':fb[:,FOV].copy()}}


def main():
 p=argparse.ArgumentParser()
 for key in ['visual','audio','fused','out']:p.add_argument('--'+key,type=Path,required=True)
 args=p.parse_args();roots=[args.visual,args.audio,args.fused];manifests=[json.loads((p/'manifest.json').read_text()) for p in roots]
 for m,arm in zip(manifests,['A0','A1','independent_equal']):
  assert m['complete'] and m['arm']==arm and m['split']=='val' and m['queries']==9965 and m['scenes']==19
 assert len({m['seed'] for m in manifests})==len({m['input_manifest_sha256'] for m in manifests})==1
 expected={r['scene'] for r in manifests[0]['records']};assert len(expected)==19
 lookups=[]
 for m in manifests:
  assert len(m['records'])==19 and {r['scene'] for r in m['records']}==expected
  lookups.append({r['scene']:r for r in m['records']})
 assert manifests[2]['source_predictions']==[dict(arm=arm,manifest_sha256=sha(root/'manifest.json')) for arm,root in zip(['A0','A1'],roots[:2])]
 args.out.mkdir(parents=True,exist_ok=False);modes=['visual_inside_audio_outside','fused_fov'];records={k:[] for k in modes}
 for mode in modes:(args.out/mode).mkdir()
 for sid in sorted(expected):
  zs=[]
  for root,lookup in zip(roots,lookups):
   rec=lookup[sid];path=root/rec['file'];assert sha(path)==rec['sha256'];zs.append(np.load(path))
  v,a,f=zs;np.testing.assert_array_equal(v['query_ids'],a['query_ids']);np.testing.assert_array_equal(v['query_ids'],f['query_ids'])
  for mode,arrays in decompose(v['V_mu'],v['V_scale'],a['A_mu'],a['A_scale'],f['F_mu'],f['F_scale']).items():
   path=args.out/mode/f'{sid}.npz';np.savez_compressed(path,query_ids=v['query_ids'],**arrays);records[mode].append(dict(scene=sid,queries=len(v['query_ids']),file=path.name,sha256=sha(path)))
 for mode in modes:
  assert sum(r['queries'] for r in records[mode])==9965
  (args.out/mode/'manifest.json').write_text(json.dumps(dict(complete=True,split='val',arm=mode,seed=manifests[0]['seed'],scenes=19,queries=9965,input_manifest_sha256=manifests[0]['input_manifest_sha256'],records=records[mode],source_manifest_sha256={m['arm']:sha(p/'manifest.json') for p,m in zip(roots,manifests)},source_sha256=sha(__file__),scope='Fixed-distribution inference diagnostic, not training ablation; no GT input',test_accessed=False),indent=2)+'\n')
if __name__=='__main__':main()

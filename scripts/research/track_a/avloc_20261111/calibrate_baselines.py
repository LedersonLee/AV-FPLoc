"""Actual native baseline field temperature curves, all Val19, no new model selection."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
from temperature_calibration import curve,crossfit,TEMPERATURES

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--audit',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--seed',type=int,choices=[42,43],required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();sys.path.insert(0,str(a.workspace/'AV-FPLoc/src'));from avfploc.geometry.rays import angular_rays
 from avfploc.evaluation.baselines import standardized
 audit=json.loads(a.audit.read_text());assert audit['complete'];a.out.mkdir(parents=True,exist_ok=False);E=a.workspace/'experiments';scenes=[];values={};correct={};records=[]
 for r in (audit['records'][:1] if a.smoke else audit['records']):
  sid=r['scene'];start=time.time();folder=a.workspace/r['folder_relative'];assert sha(folder/'receipt.json')==r['receipt_sha256']
  for name,digest in r['files'].items():assert sha(folder/name)==digest
  q=np.load(folder/'query.npz');xy=np.load(folder/'xy.npy');win=np.load(folder/'windows.npy').astype(float);fp=np.load(folder/'fp.npy').astype(float)
  hs=E/f'hypothesis_discrimination_20261107_1538/cache_seed{a.seed}';hp=next(hs/s/(sid+'.json') for s in ['val','extra'] if (hs/s/(sid+'.json')).exists());hr=json.loads(hp.read_text());paths=[E/f'astra_fusion_v2_20261107_0225/support_seed{a.seed}/{s}/{sid}_adapted.npz' for s in ['val','extra_val']]+[E/f'fusion_vs_posterior_20261107_1300/{s}_seed{a.seed}/{sid}_adapted.npz' for s in ['cache','confirmation']];ap=next(p for p in paths if p.exists() and sha(p)==hr['adapted_sha256']);z=np.load(ap);np.testing.assert_array_equal(z['query_ids'],q['sample_indices'])
  n=min(4,len(q['mu'])) if a.smoke else len(q['mu']);local={};local_correct={};tops={}
  for i in range(n):
   mu,b=angular_rays(z['mu'][i].astype(float),z['scale'][i].astype(float));v=standardized(-(abs(win-mu)/b).sum(-1)/40);ac=standardized(fp@z['observed'][i].astype(float));fields={'b2':.85*v+.15*ac}
   if a.seed==42:
    mu,b=angular_rays(q['mu'][i].astype(float),q['scale'][i].astype(float));v=-(abs(win-mu)/b).sum(-1)/40;ac=fp@q['observed'][i].astype(float);fields.update(visual=v,acoustic=ac,fixed=.85*standardized(v)+.15*standardized(ac))
   for name,field in fields.items():
    cc,meta=curve(field,xy,q['truth'][i]);local.setdefault(name,[]).append(cc);local_correct.setdefault(name,[]).append(meta['joint']);tops.setdefault(name,[]).append(meta['top1_id'])
   if i%100==0:print('CURVES',a.seed,sid,i,n,flush=True)
  payload={'query_ids':q['sample_indices'][:n]}
  for name in local:
   payload[name+'_curve']=np.array(local[name]);payload[name+'_joint']=np.array(local_correct[name]);payload[name+'_top1']=np.array(tops[name]);values.setdefault(name,[]).extend(local[name]);correct.setdefault(name,[]).extend(local_correct[name])
  path=a.out/f'{sid}.npz';np.savez_compressed(path,**payload);scenes.extend([sid]*n);records.append(dict(scene=sid,queries=n,file=path.name,sha256=sha(path),input_receipt_sha256=r['receipt_sha256'],adapted_sha256=hr['adapted_sha256'],seconds=time.time()-start));print('SCENE CURVES',a.seed,sid,n,flush=True)
 summaries={} if a.smoke else {name:crossfit(scenes,values[name],correct[name]) for name in values};assert a.smoke or len(scenes)==9965
 (a.out/'summary.json').write_text(json.dumps(dict(complete=True,scope='4-query code check, no calibration fitted' if a.smoke else 'scene-family crossfit on reused Val19; checkpoint/teacher exposure persists',seed=a.seed,queries=len(scenes),temperatures=TEMPERATURES.tolist(),methods=summaries,records=records,input_audit_sha256=sha(a.audit),source_sha256={name:sha(Path(__file__).with_name(name)) for name in ['calibrate_baselines.py','temperature_calibration.py','matching.py']},test_accessed=False),indent=2)+'\n')
if __name__=='__main__':main()

"""Frozen full Val19 cached-input fields; raw arrays stay local, no training/GT selection."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
from matching import diagnostic,proposals,calibration

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--audit',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--seed',type=int,choices=[42,43],required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
 sys.path.insert(0,str(a.workspace/'AV-FPLoc/src'));from avfploc.geometry.rays import angular_rays
 from avfploc.evaluation.baselines import standardized
 audit=json.loads(a.audit.read_text());assert audit['complete'];a.out.mkdir(parents=True,exist_ok=False);E=a.workspace/'experiments';allrows=[];receipts=[]
 for r in (audit['records'][:1] if a.smoke else audit['records']):
  start=time.time();sid=r['scene'];folder=a.workspace/r['folder_relative'];assert sha(folder/'receipt.json')==r['receipt_sha256']
  for name,digest in r['files'].items():assert sha(folder/name)==digest
  q=np.load(folder/'query.npz');xy=np.load(folder/'xy.npy');win=np.load(folder/'windows.npy').astype(float);fp=np.load(folder/'fp.npy').astype(float)
  hs=E/f'hypothesis_discrimination_20261107_1538/cache_seed{a.seed}';hp=next(hs/s/(sid+'.npz') for s in ['val','extra'] if (hs/s/(sid+'.npz')).exists());hr=json.loads(hp.with_suffix('.json').read_text());assert sha(hp)==hr['output_sha256'];h=np.load(hp)
  paths=[E/f'astra_fusion_v2_20261107_0225/support_seed{a.seed}/{s}/{sid}_adapted.npz' for s in ['val','extra_val']]+[E/f'fusion_vs_posterior_20261107_1300/{s}_seed{a.seed}/{sid}_adapted.npz' for s in ['cache','confirmation']]
  ap=next(p for p in paths if p.exists() and sha(p)==hr['adapted_sha256']);z=np.load(ap);np.testing.assert_array_equal(z['query_ids'],q['sample_indices']);np.testing.assert_array_equal(h['query_ids'],q['sample_indices']);np.testing.assert_array_equal(h['truth'],q['truth']);rows=[]
  count=min(4,len(q['mu'])) if a.smoke else len(q['mu'])
  for i in range(count):
   mu,scale=angular_rays(z['mu'][i].astype(float),z['scale'][i].astype(float));v=standardized(-(abs(win-mu)/scale).sum(-1)/40);ac=standardized(fp@z['observed'][i].astype(float));fields={'b2':.85*v+.15*ac}
   if a.seed==42:
    mu,scale=angular_rays(q['mu'][i].astype(float),q['scale'][i].astype(float));vr=-(abs(win-mu)/scale).sum(-1)/40;ar=fp@q['observed'][i].astype(float);fields.update(visual=vr,acoustic=ar,fixed=.85*standardized(vr)+.15*standardized(ar))
   metrics={}
   for method,field in fields.items():
    ids=proposals(field,xy)
    if method=='b2':np.testing.assert_array_equal(ids,h['score_nms_centers'][i])
    d=diagnostic(field,xy,q['truth'][i],candidate_ids=ids);d['candidate_ids']=ids.tolist()
    if not d['gt_region_exists']:d['gt_region_nll']=None
    metrics[method]=d
   rows.append(dict(scene=sid,query_id=int(q['sample_indices'][i]),methods=metrics))
   if i%100==0:print('QUERY',a.seed,sid,i,count,flush=True)
  target=a.out/f'{sid}.json';target.write_text(json.dumps(rows)+'\n');allrows.extend(rows);receipts.append(dict(scene=sid,queries=count,file=target.name,sha256=sha(target),adapted_sha256=hr['adapted_sha256'],candidate_archive_sha256=hr['output_sha256'],input_receipt_sha256=r['receipt_sha256'],score_nms_replay_exact=True,seconds=time.time()-start));print('SCENE DONE',a.seed,sid,count,flush=True)
 keys=['r1','joint','fine','top5_raw','top20_raw','top5_nms','top20_nms','entropy_normalized','gt_region_mass','nearest_grid_nll'];methods=allrows[0]['methods'];summary={}
 for name in methods:
  values=[v['methods'][name] for v in allrows];summary[name]=dict(means={k:float(np.mean([v[k] for v in values])) for k in keys},joint_event_calibration=calibration([v['joint_confidence'] for v in values],[v['joint'] for v in values]),per_scene={sid:{k:float(np.mean([v['methods'][name][k] for v in allrows if v['scene']==sid])) for k in keys} for sid in sorted({v['scene'] for v in allrows})})
 assert a.smoke or len(allrows)==9965
 (a.out/'summary.json').write_text(json.dumps(dict(complete=True,scope='4-query implementation check' if a.smoke else 'full reused Val19 development',seed=a.seed,queries=len(allrows),scenes=len(receipts),grid_cm=20,temperature=1,score_units={'visual':'negative scaled ray distance/40','acoustic':'unit-embedding cosine','fixed':'.85z(V)+.15z(A)','b2':'adapted .85z(V)+.15z(A)'},warning='Softmax score mass is not demonstrated calibrated probability; native temperatures/score scales differ. GEOQ utility is not a pose posterior.',methods=summary,records=receipts,source_sha256={n:sha(Path(__file__).with_name(n)) for n in ['evaluate_baseline_fields.py','matching.py']},input_audit_sha256=sha(a.audit),test_accessed=False,training=False),indent=2)+'\n')
if __name__=='__main__':main()

"""Replay frozen GEOQ against historical full Val19 support candidates, without fitting."""
import argparse,hashlib,importlib.util,json,time
from pathlib import Path
import numpy as np,joblib
from matching import pose_values,circular_error

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--audit',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();audit=json.loads(a.audit.read_text());assert audit['complete'] and audit['queries']==9965
 root=a.workspace/'experiments/structural_evidence_20261107_1730';hs=a.workspace/'experiments/hypothesis_discrimination_20261107_1538';spec=importlib.util.spec_from_file_location('historical_geoq',root/'inference.py');geo=importlib.util.module_from_spec(spec);spec.loader.exec_module(geo)
 a.out.mkdir(parents=True,exist_ok=False);results=[]
 for seed in [42,43]:
  selection=json.loads((root/f'final_selection_seed{seed}.json').read_text());chosen=selection['selected'];assert chosen['arm']=='GEOQ';cp=root/f'GEOQ_seed{seed}/model.joblib';assert sha(cp)==selection['model_sha256'];model=joblib.load(cp);rows=[];sources=[];start=time.time()
  for r in audit['records']:
   sid=r['scene'];paths=[root/f'features_seed{seed}'/split/(sid+'.npz') for split in ['val','extra']];fpath=next(p for p in paths if p.exists());split=fpath.parent.name;rec=json.loads(fpath.with_suffix('.json').read_text());assert sha(fpath)==rec['sha256'];f=np.load(fpath)
   hpath=hs/f'cache_seed{seed}'/split/(sid+'.npz');h=np.load(hpath);hr=json.loads(hpath.with_suffix('.json').read_text());assert sha(hpath)==hr['output_sha256']==rec['candidate_sha256']
   folder=a.workspace/r['folder_relative'];assert sha(folder/'receipt.json')==r['receipt_sha256'];q=np.load(folder/'query.npz');xy=np.load(folder/'xy.npy');np.testing.assert_array_equal(f['query_ids'],q['sample_indices']);np.testing.assert_array_equal(f['truth'],q['truth']);np.testing.assert_array_equal(f['indices'],h['support_nms_indices']);np.testing.assert_array_equal(f['S'],h['support_nms_features'])
   choice,gain=geo.rank(model,f['S'],f['V'][...,:11],chosen['threshold'],f['V'][...,11:22]);old=np.load(root/f'GEOQ_seed{seed}/{split}_gains.npz')[sid];np.testing.assert_allclose(gain,old,rtol=0,atol=1e-10)
   assert len(choice)==r['queries'];candidate=f['indices'];poses=pose_values(xy,candidate.ravel()).reshape(len(choice),20,3);truth=q['truth'];dist=np.linalg.norm(poses[...,:2]-truth[:,None,:2],axis=-1);yaw=circular_error(poses[...,2],truth[:,None,2]);joint=(dist<=1)&(yaw<=30);np.testing.assert_array_equal(joint,f['labels'][...,1]);ii=np.arange(len(choice));picked=joint[ii,choice]
   for i,qid in enumerate(q['sample_indices']):
    rows.append(dict(scene=sid,query_id=int(qid),candidate_ids=candidate[i].tolist(),selected_slot=int(choice[i]),selected_id=int(candidate[i,choice[i]]),position_error_m=float(dist[i,choice[i]]),yaw_error_deg=float(yaw[i,choice[i]]),r1=bool(dist[i,choice[i]]<=1),joint=bool(picked[i]),fine=bool(dist[i,choice[i]]<=.1),support_joint=bool(joint[i,0]),top20_support=bool(joint[i].any()),top5_support=bool(joint[i,:5].any()),vision_joint=bool(f['vision_success'][i]),selection_failure=bool(joint[i].any() and not picked[i])))
   sources.append(dict(scene=sid,feature_sha256=rec['sha256'],candidate_sha256=rec['candidate_sha256'],gain_archive_sha256=sha(root/f'GEOQ_seed{seed}/{split}_gains.npz')))
  assert len(rows)==9965;summary=dict(seed=seed,threshold=chosen['threshold'],queries=len(rows),scenes=19,metrics={k:float(np.mean([v[k] for v in rows])) for k in ['r1','joint','fine','support_joint','top20_support','top5_support','selection_failure']},repair_vs_vision=sum(v['joint'] and not v['vision_joint'] for v in rows),harm_vs_vision=sum(not v['joint'] and v['vision_joint'] for v in rows),per_scene={sid:{k:float(np.mean([v[k] for v in rows if v['scene']==sid])) for k in ['r1','joint','fine','top20_support']} for sid in sorted({v['scene'] for v in rows})},exact_gain_rows=len(rows),model_sha256=sha(cp),threshold_selection_sha256=sha(root/f'final_selection_seed{seed}.json'),seconds=time.time()-start,sources=sources)
  (a.out/f'seed{seed}_queries.json').write_text(json.dumps(rows)+'\n');(a.out/f'seed{seed}_summary.json').write_text(json.dumps(summary,indent=2)+'\n');results.append(summary);print('GEOQ',seed,summary['metrics'],flush=True)
 (a.out/'manifest.json').write_text(json.dumps(dict(complete=True,scope='frozen historical full Val19 replay; reused/teacher-exposed development, not independent validation',seeds=[42,43],source_sha256=sha(__file__),historical_inference_sha256=sha(root/'inference.py'),audit_sha256=sha(a.audit),test_accessed=False,training=False,summary_files=[f'seed{s}_summary.json' for s in [42,43]]),indent=2)+'\n')
if __name__=='__main__':main()

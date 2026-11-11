"""Full Val19 geometry-only evaluation of extracted ray distributions.
Small per-query diagnostics are saved; raw dense score arrays are not persisted.
"""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from matching import laplace_field,proposals,diagnostic,calibration,FOV

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--predictions',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--banks',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--grid-cm',type=int,choices=[10,20],required=True);p.add_argument('--head',choices=['V','A','AV','F'],required=True);a=p.parse_args()
 manifest=json.loads((a.predictions/'manifest.json').read_text());banks=json.loads((a.banks/'manifest.json').read_text());assert manifest['complete'] and banks['complete'];assert manifest['split']=='val' and len(manifest['records'])==19
 assert {r['scene'] for r in banks['records']}=={r['scene'] for r in manifest['records']}
 assert sha(a.data/'manifest.json')==manifest['input_manifest_sha256'];a.out.mkdir(parents=True,exist_ok=False);allrows=[];scenes=[]
 bank_records={r['scene']:r for r in banks['records']}
 for r in manifest['records']:
  start=time.time();sid=r['scene'];path=a.predictions/r['file'];assert sha(path)==r['sha256'];pred=np.load(path);labels=np.load(a.data/sid/'labels.npz');assert np.array_equal(pred['query_ids'],labels['query_ids']);bp=a.banks/sid/f'grid_{a.grid_cm}cm.npz';assert sha(bp)==bank_records[sid]['files'][bp.name];bank=np.load(bp);rows=[]
  dirs=FOV if a.head=='V' else np.arange(36)
  for i,qid in enumerate(pred['query_ids']):
   mu=pred[a.head+'_mu'][i];b=pred[a.head+'_scale'][i];field=laplace_field(bank['world_rays'],mu,b,dirs);ids=proposals(field,bank['xy']);row=diagnostic(field,bank['xy'],labels['truth'][i],candidate_ids=ids)
   err=abs(mu-labels['target'][i,dirs]);inside=np.isin(dirs,FOV);row.update(query_id=int(qid),scene=sid,ray_mae=float(err.mean()),in_fov_mae=float(err[inside].mean()),out_fov_mae=float(err[~inside].mean()) if (~inside).any() else None,ray_nll=float((err/b+np.log(2*b)).mean()),candidate_ids=ids.tolist())
   # Explicit null for unrepresentable regions; aggregate counts retain the limitation.
   if not row['gt_region_exists']:row['gt_region_nll']=None
   rows.append(row)
  (a.out/f'{sid}.json').write_text(json.dumps(rows,indent=2)+'\n');allrows.extend(rows);scenes.append(dict(scene=sid,queries=len(rows),joint=float(np.mean([v['joint'] for v in rows])),ray_mae=float(np.mean([v['ray_mae'] for v in rows])),seconds=time.time()-start));print('SCENE',sid,len(rows),scenes[-1]['joint'],flush=True)
 metrics=['r1','joint','fine','top5_nms','top20_nms','top5_raw','top20_raw','ray_mae','ray_nll','in_fov_mae','entropy_normalized','gt_region_mass','nearest_grid_nll']
 result=dict(complete=True,scope='reused Val19 development; not independent validation',queries=len(allrows),head=a.head,grid_cm=a.grid_cm,temperature=1,pose_prior='uniform discrete grid; product of conditional ray Laplace densities',metric_means={k:float(np.mean([v[k] for v in allrows])) for k in metrics},scene_macro_joint=float(np.mean([v['joint'] for v in scenes])),unrepresentable_gt_regions=sum(not v['gt_region_exists'] for v in allrows),joint_event_calibration=calibration([v['joint_confidence'] for v in allrows],[v['joint'] for v in allrows]),scenes=scenes,prediction_manifest_sha256=sha(a.predictions/'manifest.json'),bank_manifest_sha256=sha(a.banks/'manifest.json'),source_sha256={n:sha(Path(__file__).with_name(n)) for n in ['matching.py','evaluate_fields.py']},test_accessed=False)
 assert len(allrows)==9965;(a.out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()

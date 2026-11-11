"""Temperature diagnostic for completed AVLoc predictions, with native replay checks.
Uses the same temperatures and family folds as cached Track A baselines. No fitting
of network parameters. Native full-field evaluation must already be complete.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from matching import laplace_field,FOV
from temperature_calibration import curve,crossfit,TEMPERATURES
from predict_hash import sha


def check_native(values,meta,row):
 """Independent native evaluation must agree before accepting any curve."""
 idx=int(np.flatnonzero(TEMPERATURES==1)[0])
 assert meta['top1_id']==row['top1_id']
 assert meta['joint']==row['joint']
 keys=['nearest_grid_nll','joint_confidence','entropy_normalized','gt_region_mass']
 np.testing.assert_allclose(values[idx,[0,2,3,4]],[row[k] for k in keys],rtol=1e-9,atol=1e-9)


def main():
 p=argparse.ArgumentParser()
 for name in ['predictions','data','banks','native','out']:p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--grid-cm',type=int,choices=[10,20],required=True)
 p.add_argument('--head',choices=['V','A','AV','F'],required=True)
 a=p.parse_args()
 manifest=json.loads((a.predictions/'manifest.json').read_text());banks=json.loads((a.banks/'manifest.json').read_text());native=json.loads((a.native/'summary.json').read_text())
 assert manifest['complete'] and manifest['split']=='val' and manifest['queries']==9965
 assert banks['complete'] and native['complete'] and native['queries']==9965
 assert native['head']==a.head and native['grid_cm']==a.grid_cm and native['temperature']==1
 assert native['prediction_manifest_sha256']==sha(a.predictions/'manifest.json')
 assert native['bank_manifest_sha256']==sha(a.banks/'manifest.json')
 assert manifest['input_manifest_sha256']==sha(a.data/'manifest.json')
 data=json.loads((a.data/'manifest.json').read_text());expected={r['scene'] for r in data['records'] if r['split']=='val'}
 assert len(expected)==19 and {r['scene'] for r in manifest['records']}==expected
 assert len(manifest['records'])==19 and {r['scene'] for r in banks['records']}==expected
 bank_records={r['scene']:r for r in banks['records']}
 a.out.mkdir(parents=True,exist_ok=False);scenes=[];values=[];correct=[];records=[]
 dirs=FOV if a.head=='V' else np.arange(36)
 for r in manifest['records']:
  started=time.time();sid=r['scene'];pp=a.predictions/r['file'];assert sha(pp)==r['sha256']
  pred=np.load(pp);labels=np.load(a.data/sid/'labels.npz');np.testing.assert_array_equal(pred['query_ids'],labels['query_ids'])
  bp=a.banks/sid/f'grid_{a.grid_cm}cm.npz';assert sha(bp)==bank_records[sid]['files'][bp.name];bank=np.load(bp)
  native_path=a.native/f'{sid}.json';rows=json.loads(native_path.read_text());assert len(rows)==len(pred['query_ids'])
  assert [(v['scene'],v['query_id']) for v in rows]==[(sid,int(q)) for q in pred['query_ids']]
  local=[];joints=[];tops=[]
  for i,row in enumerate(rows):
   field=laplace_field(bank['world_rays'],pred[a.head+'_mu'][i],pred[a.head+'_scale'][i],dirs)
   cc,meta=curve(field,bank['xy'],labels['truth'][i]);check_native(cc,meta,row)
   local.append(cc);joints.append(meta['joint']);tops.append(meta['top1_id'])
   if i%100==0:print('RAY CURVES',a.head,a.grid_cm,sid,i,len(rows),flush=True)
  path=a.out/f'{sid}.npz';np.savez_compressed(path,query_ids=pred['query_ids'],ray_curve=np.asarray(local),ray_joint=np.asarray(joints),ray_top1=np.asarray(tops))
  scenes.extend([sid]*len(rows));values.extend(local);correct.extend(joints)
  records.append(dict(scene=sid,queries=len(rows),file=path.name,sha256=sha(path),native_rows_sha256=sha(native_path),seconds=time.time()-started))
 assert len(scenes)==9965
 result=dict(complete=True,scope='Reused Val19 development; checkpoint and upstream exposure persists',seed=manifest['seed'],head=a.head,grid_cm=a.grid_cm,queries=len(scenes),temperatures=TEMPERATURES.tolist(),methods={'ray':crossfit(scenes,values,correct)},records=records,native_checks=len(scenes),prediction_manifest_sha256=sha(a.predictions/'manifest.json'),bank_manifest_sha256=sha(a.banks/'manifest.json'),native_summary_sha256=sha(a.native/'summary.json'),source_sha256={n:sha(Path(__file__).with_name(n)) for n in ['calibrate_rays.py','predict_hash.py','matching.py','temperature_calibration.py']},test_accessed=False)
 (a.out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()

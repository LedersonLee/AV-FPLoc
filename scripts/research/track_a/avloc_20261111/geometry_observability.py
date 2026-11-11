"""Post-hoc geometry ambiguity diagnostics; GT profiles NEVER feed a trained model.
This equal-scale perfect-profile experiment is not an upper bound for learned uncertainty.
"""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from matching import laplace_field,circular_error,pose_values,FOV

def diagnose(world,xy,target,truth,directions=None):
 directions=np.arange(36) if directions is None else np.asarray(directions);target=np.asarray(target);truth=np.asarray(truth);xy=np.asarray(xy)
 # Algebraic recovery of mean absolute ray discrepancy from the tested Eq2 kernel.
 score=laplace_field(world,target[directions],np.ones(len(directions)),directions);error=-score/len(directions)-np.log(2)
 dist=np.linalg.norm(xy-truth[:2],axis=1);yaw=circular_error(np.arange(36)*10,truth[2]);near=((dist[:,None]<=1)&(yaw[None]<=30)).ravel();position_near=np.repeat(dist<=1,36);best=int(error.argmin());pose=pose_values(xy,[best])[0]
 correct=float(error[near].min()) if near.any() else None;wrong=float(error[~near].min()) if (~near).any() else None
 pos_correct=float(error[position_near].min()) if position_near.any() else None;pos_wrong=float(error[~position_near].min()) if (~position_near).any() else None
 return dict(perfect_profile_top1_id=best,perfect_profile_position_error_m=float(np.linalg.norm(pose[:2]-truth[:2])),perfect_profile_yaw_error_deg=float(circular_error(pose[2],truth[2])),perfect_profile_joint=bool(near[best]),perfect_profile_fine=bool(np.linalg.norm(pose[:2]-truth[:2])<=.1),minimum_profile_mae_m=float(error[best]),best_joint_correct_profile_mae_m=correct,best_joint_incorrect_profile_mae_m=wrong,joint_geometry_margin_m=wrong-correct if correct is not None and wrong is not None else None,position_geometry_margin_m=pos_wrong-pos_correct if pos_correct is not None and pos_wrong is not None else None,incorrect_profiles_within_5cm_of_best_correct=int(((error<=correct+.05)&~near).sum()) if correct is not None else None,correct_region_exists=bool(near.any()),ray_count=len(directions))

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--grid-cm',type=int,choices=[10,20],required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();data=json.loads((a.study/'data_v1/manifest.json').read_text());banks=json.loads((a.study/'banks_v1/manifest.json').read_text());assert data['ready'] and banks['complete'];br={r['scene']:r for r in banks['records']};records=[r for r in data['records'] if r['split']=='val'];a.out.mkdir(parents=True,exist_ok=False);out=[];receipts=[]
 for r in (records[:1] if a.smoke else records):
  start=time.time();sid=r['scene'];lp=a.study/'data_v1'/sid/'labels.npz';assert sha(lp)==r['files']['labels.npz'];labels=np.load(lp);bp=a.study/'banks_v1'/sid/f'grid_{a.grid_cm}cm.npz';assert sha(bp)==br[sid]['files'][bp.name];bank=np.load(bp);rows=[];n=min(4,r['queries']) if a.smoke else r['queries']
  for i in range(n):
   row=dict(scene=sid,query_id=int(labels['query_ids'][i]),omnidirectional=diagnose(bank['world_rays'],bank['xy'],labels['target'][i],labels['truth'][i]),visual_fov=diagnose(bank['world_rays'],bank['xy'],labels['target'][i],labels['truth'][i],FOV));rows.append(row)
   if i%100==0:print('GEOMETRY',a.grid_cm,sid,i,n,flush=True)
  path=a.out/f'{sid}.json';path.write_text(json.dumps(rows)+'\n');out.extend(rows);receipts.append(dict(scene=sid,queries=n,file=path.name,sha256=sha(path),bank_sha256=sha(bp),labels_sha256=sha(lp),seconds=time.time()-start))
 assert a.smoke or len(out)==9965
 summary={}
 for name in ['omnidirectional','visual_fov']:
  rows=[v[name] for v in out];margins=[v['joint_geometry_margin_m'] for v in rows if v['joint_geometry_margin_m'] is not None];summary[name]=dict(perfect_profile_joint=float(np.mean([v['perfect_profile_joint'] for v in rows])),perfect_profile_fine=float(np.mean([v['perfect_profile_fine'] for v in rows])),median_geometry_margin_m=float(np.median(margins)),nonpositive_geometry_margin_fraction=float(np.mean(np.array(margins)<=1e-12)),missing_correct_regions=sum(not v['correct_region_exists'] for v in rows))
 (a.out/'summary.json').write_text(json.dumps(dict(complete=True,scope='4query diagnostic code check' if a.smoke else 'all Val19 post-hoc geometric ambiguity; no learned predictions',queries=len(out),grid_cm=a.grid_cm,results=summary,records=receipts,warning='GT-derived profiles only in diagnostic; not model input, not pose proposals, not learned upper bound.5cm profile allowance is a predeclared descriptive threshold, not localization success.',source_sha256={n:sha(Path(__file__).with_name(n)) for n in ['geometry_observability.py','matching.py']},test_accessed=False),indent=2)+'\n')
if __name__=='__main__':main()

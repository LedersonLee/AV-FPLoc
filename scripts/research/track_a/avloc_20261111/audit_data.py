"""Full Train/Val RGB + observed-RIR integrity audit. No Test observations or FP-RIR reads."""
import argparse,hashlib,json,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def inspect_scene(job):
 root,out,part,sid=job;start=time.time();sp=root/f'RIR/manifests/positive_pairs_v1/schedules/{part}/{sid}.json';row=dict(scene=sid,split=part,issues=[],queries=0,valid_queries=0)
 if not sp.exists():row['issues'].append('missing schedule');return row
 sch=read(sp);mp=root/f'2DFP/{sid}/map.png';meta=read(mp.with_name('floorplan_metadata.json'));free=np.array(Image.open(mp).convert('L'))==255;origin=np.array([meta['world_bounds_m'][k] for k in ['x_min','z_min']]);spacing=meta['map_spacing_m'];capfile=root/f'UnLoc/RGB/{sid}/pilot_manifest.json';cap=read(capfile);caps={q['sample_index']:q for q in cap['records'] if q['accepted']}
 replacement=root/f'UnLoc/furnished_depth_qa_v2_nohit_only/replacements/{sid}/replacement_manifest.json'
 if replacement.exists():caps.update({q['sample_index']:q for q in read(replacement)['records'] if q.get('accepted')})
 rows=[];rir_cache={};shapes={};lengths=[]
 for q in sch['queries']:
  row['queries']+=1;issues=[];pose=q['centre_xyz_m'];i=q['sample_index'];rgb=root/q['image_path'];rp=root/q['observed_rir_path'];assert q['observed_rir_path'].startswith('RIR/Observed/');assert '/test/' not in q['observed_rir_path']
  try:
   with Image.open(rgb) as im:im.load();size=im.size;assert size==(640,480) and im.mode in ['RGB','RGBA']
   image_hash=sha(rgb)
  except Exception as e:issues.append('RGB:'+type(e).__name__);image_hash=None
  if str(rp) not in rir_cache:
   err=[];info={}
   try:
    x=np.load(rp,allow_pickle=False);side=read(rp.with_suffix('.json'));assert x.ndim==2 and x.shape[0]==6 and x.shape[1]>0 and np.isfinite(x).all();assert (x*x).sum()>0;h=sha(rp);assert h==side['rir_sha256'];info=dict(sha256=h,shape=list(x.shape),centre=side['centre_xyz_m'],peak_sample=np.argmax(abs(x),axis=1).tolist());lengths.append(x.shape[1]);shapes[str(tuple(x.shape))]=shapes.get(str(tuple(x.shape)),0)+1
   except Exception as e:err.append('RIR:'+type(e).__name__)
   rir_cache[str(rp)]=(err,info)
  err,ri=rir_cache[str(rp)];issues.extend(err)
  if ri and not np.allclose(ri['centre'],pose,atol=1e-6):issues.append('RIR centre mismatch')
  c=caps.get(i)
  if c is None or not np.allclose(c['pose_xyz_m'],pose,atol=1e-6) or not np.isclose(c['logical_yaw_deg'],q['world_yaw_deg']):issues.append('RGB pose mismatch')
  if c:
   a=np.deg2rad(c['habitat_yaw_deg']);w=np.deg2rad(q['world_yaw_deg'])
   if not np.allclose([-np.sin(a),-np.cos(a)],[np.cos(w),np.sin(w)],atol=1e-6):issues.append('camera yaw mismatch')
  a=(-q['world_yaw_deg'])%360;turns=(a-q['base_acoustic_yaw_deg'])/60
  if not np.isclose(turns,round(turns)) or q['channel_roll']!=-int(round(turns)):issues.append('channel roll mismatch')
  px=np.rint((np.array(pose)[[0,2]]-origin)/spacing).astype(int)
  if not(0<=px[0]<free.shape[1] and 0<=px[1]<free.shape[0] and free[px[1],px[0]]):issues.append('outside map free support')
  if issues:row['issues'].append(dict(query=i,reasons=issues))
  else:row['valid_queries']+=1
  rows.append(dict(query_id=i,rgb=q['image_path'],rir=q['observed_rir_path'],pose=[pose[0],pose[2],q['world_yaw_deg']],channel_roll=q['channel_roll'],base_acoustic_yaw=q['base_acoustic_yaw_deg'],rgb_sha256=image_hash,rir_info=ri,issues=issues))
 row.update(unique_rirs=len(rir_cache),rir_shapes=shapes,rir_samples_min=min(lengths) if lengths else None,rir_samples_max=max(lengths) if lengths else None,camera=cap['camera_contract'],acoustics=sch.get('acoustics'),map_sha256=sha(mp),metadata_sha256=sha(mp.with_name('floorplan_metadata.json')),schedule_sha256=sha(sp),seconds=time.time()-start)
 (out/f'{sid}.json').write_text(json.dumps(dict(summary=row,queries=rows),indent=2)+'\n');print('AUDIT',sid,row['valid_queries'],'/',row['queries'],'issues',len(row['issues']),flush=True);return row

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);split=read(a.root/'manifests/scene_splits_v1.json');family=lambda s:s.split('_')[0];tf={family(s) for s in split['test']};vf={family(s) for s in split['val']};excluded_test=[s for s in split['train'] if family(s) in tf];excluded_val=[s for s in split['train'] if family(s) in vf and s not in excluded_test];ids={'train':[s for s in split['train'] if s not in excluded_test+excluded_val],'val':split['val']};assert not {family(s) for s in ids['val']}&tf
 protocol=dict(official_counts={k:len(v) for k,v in split.items()},excluded_test_family=excluded_test,excluded_val_family=excluded_val,requested=ids,split_sha256=sha(a.root/'manifests/scene_splits_v1.json'),test_observations_accessed=False,source_family='prefix before underscore; conservative proxy, not a full near-duplicate geometry audit');(a.out/'split.json').write_text(json.dumps(protocol,indent=2)+'\n')
 with ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(inspect_scene,[(a.root,a.out,part,sid) for part in ids for sid in ids[part]]))
 summary=dict(**protocol,scenes=rows,total_queries=sum(r['queries'] for r in rows),valid_queries=sum(r['valid_queries'] for r in rows),eligible_scenes={p:[r['scene'] for r in rows if r['split']==p and r['valid_queries']>0] for p in ids},code_sha256=sha(Path(__file__)),complete=True)
 (a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print('DONE',summary['total_queries'],summary['valid_queries'],flush=True)
if __name__=='__main__':main()

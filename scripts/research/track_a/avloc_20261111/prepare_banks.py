"""Val19 geometry-only banks, primary XY verified against historical bank metadata.
No raw FP-RIR arrays/embeddings or Test observations are loaded.
"""
import argparse,json,hashlib,time
from pathlib import Path
import numpy as np
from PIL import Image
from geometry import xy_grid,cast_rays

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);split=json.loads((a.root/'manifests/scene_splits_v1.json').read_text());metadata=a.root/'RIR/manifests/dense_banks_v1/bank_index.json';entries={e['scene_id']:e for e in json.loads(metadata.read_text())['scenes'] if e['split']=='val'};assert set(entries)==set(split['val']);records=[]
 for sid in split['val']:
  dest=a.out/sid;dest.mkdir(exist_ok=True);receipt=dest/'receipt.json'
  if receipt.exists():records.append(json.loads(receipt.read_text()));continue
  start=time.time();mapfile=a.root/f'2DFP/{sid}/map.png';meta=json.loads(mapfile.with_name('floorplan_metadata.json').read_text());free=np.asarray(Image.open(mapfile).convert('L'))==255;origin=[meta['world_bounds_m'][k] for k in ['x_min','z_min']];sp=a.root/entries[sid]['schedule_path'];assert sha(sp)==entries[sid]['schedule_sha256'];schedule=json.loads(sp.read_text());reference=np.array([[q['centre_xyz_m'][0],q['centre_xyz_m'][2]] for q in schedule['positions']]);files={};sizes={}
  for step in [.2,.1]:
   xy=xy_grid(free,meta['map_spacing_m'],origin,step)
   if step==.2:np.testing.assert_allclose(xy,reference,atol=1e-7,rtol=0)
   pieces=[]
   for first in range(0,len(xy),1024):
    p=np.column_stack([xy[first:first+1024],np.zeros(min(1024,len(xy)-first))]);r,known=cast_rays(free,meta['map_spacing_m'],origin,p);assert known.all();pieces.append(r)
   name=f'grid_{int(step*100)}cm.npz';np.savez_compressed(dest/name,xy=xy,world_rays=np.concatenate(pieces));files[name]=sha(dest/name);sizes[str(step)]=dict(xy=len(xy),poses=len(xy)*36,max_distance=float(np.max(np.concatenate(pieces))))
  rec=dict(scene=sid,grid=sizes,files=files,map_sha256=sha(mapfile),historical_grid_schedule_sha256=sha(sp),primary_xy_verified=True,clearance_m=.06,ray_policy='exact first blocked pixel-cell boundary, uncapped, yaw rotation by index permutation',seconds=time.time()-start,test_accessed=False);receipt.write_text(json.dumps(rec,indent=2)+'\n');records.append(rec);print('BANK',sid,sizes,round(time.time()-start,1),flush=True)
 (a.out/'manifest.json').write_text(json.dumps(dict(records=records,complete=len(records)==19,code_sha256=sha(Path(__file__))),indent=2)+'\n')
if __name__=='__main__':main()

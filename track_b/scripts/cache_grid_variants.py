"""Map-only candidate resolution/mask sensitivity; no proposal or GT filtering."""
import argparse,json,hashlib,math,time
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
import torch
from track_b.data.geometry import candidate_bank
from track_b.data.rays import boundary_rays

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--reference-cache',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
 variants=[('baseline',.25,36,0.),('xy020',.2,36,0.),('xy050',.5,36,0.),('yaw12',.25,12,0.),('yaw72',.25,72,0.),('eroded010',.25,36,.1)];data=read(a.reference_cache/'protocol.json');original={x['scene']:x for x in read(a.reference_cache/'manifest.json')};rows=[]
 protocol=dict(variants=variants,scenes=data['scenes']['val'],reference_protocol_sha256=sha(a.reference_cache/'protocol.json'),construction='Same native footprint; map-origin anchored grid; optional clearance>=0.1m validity only; ray geometry uses unmodified map',ray_bins=16,max_range_m=20,ray_step_m=.1,observations='none used to construct candidates; original cache loaded only for baseline equality assertion',test_accessed=False)
 (a.out/'protocol.json').write_text(json.dumps(protocol,indent=2))
 for sid in protocol['scenes']:
  p=a.root/f'2DFP/{sid}/map.png';assert sha(p)==original[sid]['map_sha256'];free=np.array(Image.open(p))==255;meta=read(p.parent/'floorplan_metadata.json');spacing=meta['map_spacing_m'];origin=[meta['world_bounds_m'][k] for k in ['x_min','z_min']];reference=torch.load(a.reference_cache/f'{sid}.pt',weights_only=True,map_location='cpu')['candidates']
  clearance=distance_transform_edt(np.pad(free,1,constant_values=False))[1:-1,1:-1]*spacing
  for name,step,yaws,erosion in variants:
   start=time.perf_counter();mask=free if not erosion else free&(clearance>=erosion);poses,valid=candidate_bank(mask,spacing,origin,step,yaws);ranges,hits=boundary_rays(free,spacing,origin,poses[::yaws,:2],yaws);candidates=torch.tensor(np.column_stack([poses,np.log1p(ranges)/math.log(21),hits]).astype(np.float32))
   if name=='baseline':assert torch.equal(candidates,reference),'Baseline differs from frozen candidate bank'
   dest=a.out/f'{sid}_{name}.pt';torch.save(dict(scene=sid,variant=name,candidates=candidates,valid=torch.tensor(valid),grid_m=step,yaw_bins=yaws,erosion_m=erosion),dest);rows.append(dict(scene=sid,variant=name,candidates=len(candidates),valid_xy=int(valid.sum()),grid_m=step,yaw_bins=yaws,erosion_m=erosion,map_sha256=sha(p),sha256=sha(dest),bytes=dest.stat().st_size,construction_seconds=time.perf_counter()-start,baseline_exact_match=name=='baseline'));(a.out/'manifest.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows[-1]),flush=True)
 print('COMPLETE',flush=True)
if __name__=='__main__':main()

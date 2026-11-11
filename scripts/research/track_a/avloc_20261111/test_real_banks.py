"""Numerical self-consistency check on all prepared Val banks; no learned accuracy."""
import argparse,json,time
from pathlib import Path
import numpy as np
from matching import laplace_field,proposals,pose_values,diagnostic,FOV
p=argparse.ArgumentParser();p.add_argument('--banks',type=Path,required=True);a=p.parse_args();manifest=json.loads((a.banks/'manifest.json').read_text());assert manifest['complete'];rows=[]
for record in manifest['records']:
 for grid in [20,10]:
  bank=np.load(a.banks/record['scene']/f'grid_{grid}cm.npz');world=bank['world_rays'];xy=bank['xy'];idx=len(xy)//2;yaw=35;target_id=idx*36+yaw;start=time.time()
  for dirs in [FOV,np.arange(36)]:
   mu=world[idx,(yaw+dirs)%36];scale=np.full(len(dirs),.1);scores=laplace_field(world,mu,scale,dirs);assert abs(scores[target_id]+np.log(.2)*len(dirs))<1e-10;assert scores.max()<=scores[target_id]+1e-10
   ids=proposals(scores,xy);truth=pose_values(xy,[target_id])[0];d=diagnostic(scores,xy,truth,candidate_ids=ids);assert np.isfinite(d['nearest_grid_nll']) and 0<=d['entropy_normalized']<=1+1e-10
  rows.append(dict(scene=record['scene'],grid_cm=grid,poses=len(scores),seconds=time.time()-start))
print(json.dumps(dict(status='pass',scope='known bank profile likelihood maximum; tied geometries allowed, not model localization performance',scenes=19,bank_checks=len(rows),rows=rows),indent=2))

"""Validate that geometric clearance checks have a functioning collision backend."""
import argparse,json
from pathlib import Path
import numpy as np
from run_acoustic import make,ray_distance,write
ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--mode');a=ap.parse_args();c=json.loads((a.out/'config.json').read_text());sim=make(c,a.out,physics=True);pairs=json.loads((a.out/'acoustic_pairs.json').read_text());rows=[]
for p in pairs:
 center=np.array(p['center']);source=np.array(p['source']);d=source-center;rows.append(dict(label=p['label'],down_ray=ray_distance(sim,center,[0,-1,0]),source_down=ray_distance(sim,source,[0,-1,0]),center_source_ray=ray_distance(sim,center,d/np.linalg.norm(d)),source_distance=float(np.linalg.norm(d))))
manager=sim.get_rigid_object_manager();counts={}
for handle in manager.get_object_handles():
 obj=manager.get_object_by_handle(handle);typ=str(obj.motion_type);counts[typ]=counts.get(typ,0)+1
write(a.out/'collision_backend_physics_check.json',dict(rays=rows,motion_types=counts));print(rows);print(counts);sim.close()

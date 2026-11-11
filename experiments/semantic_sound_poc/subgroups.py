"""Additional reporting only; no model or sample selection changes."""
import argparse,json
from pathlib import Path
import numpy as np
from matplotlib.path import Path as Polygon
ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);o=ap.parse_args().out;c=json.loads((o/'config.json').read_text());gt=np.load(o/'evaluation_truth.npz')['truth'];z=np.load(o/'oracle_predictions.npz');regs=json.loads((Path(c['root'])/'source/hssd-hab/semantics/scenes'/(c['scene']+'.semantic_config.json')).read_text())['region_annotations'];room=np.full(len(gt),-1)
for i,r in enumerate(regs):room[Polygon(np.array(r['poly_loop'])[:,[0,2]]).contains_points(gt[:,:2])]=i
result={}
for i in np.unique(room):
 mask=room==i;result[str(i)]=dict(label=regs[i]['label'] if i>=0 else 'unassigned',unique_queries=int(mask.sum()),joint_percent={str(name):float(z['correct'][:,j,mask].mean()*100) for j,name in enumerate(z['names'])})
# Full paired C1/C2 harms/repairs vs B3, not only vs B0.
comparisons={}
for j in [4,5]:
 d=z['correct'][:,j].astype(int)-z['correct'][:,3];comparisons[str(z['names'][j])+'_vs_B3']=dict(improved=int((d>0).sum()),degraded=int((d<0).sum()),unchanged=int((d==0).sum()))
(o/'subgroup_details.json').write_text(json.dumps(dict(robot_room=result,negative_controls=comparisons),indent=2)+'\n')

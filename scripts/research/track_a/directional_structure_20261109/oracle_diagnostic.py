from pathlib import Path
import json
import numpy as np
from scipy.spatial import cKDTree
from fullbank import field_geometry
from prepare_candidates import nms
from discrimination import coarse
P=Path('/output');W=Path('/workspace');OLD=W/'experiments/directional_tribranch_20261109';PR=json.loads((P/'PROTOCOL.json').read_text());records=[];saved=P/'oracle';saved.mkdir(exist_ok=True)
for sid in PR['training_scenes']:
 o=np.load(OLD/'data'/sid/'observations.npz');b=np.load(P/'candidates'/sid/'bank_geometry.npz');xy=b['xy'];g=field_geometry(b['depth']);near=cKDTree(xy).query_ball_tree(cKDTree(xy),1+1e-7);gt=np.log1p(coarse(o['target']));indices=[];labels=[];qerr=[]
 for i in range(128):
  score=-abs(g-gt[i]).mean(-1);idx=nms(score,near);d=np.linalg.norm(xy-o['truth'][i,:2],axis=-1);yaw=abs((np.arange(36)*10-o['truth'][i,2]+180)%360-180);lab=np.stack([d[idx//36]<=1,(d[idx//36]<=1)&(yaw[idx%36]<=30),d[idx//36]<=.1],-1);indices.append(idx);labels.append(lab);nearest=int(d.argmin())*36+int(yaw.argmin());qerr.append(abs(g[nearest]-gt[i]));records.append(dict(scene=sid,query_id=int(o['query_ids'][i]),joint=int(lab[0,1]),fine=int(lab[0,2]),top5=bool(lab[:5,1].any()),top20=bool(lab[:,1].any()),nearest_grid_log_geometry_error=float(abs(g[nearest]-gt[i]).mean())))
 np.savez_compressed(saved/(sid+'.npz'),indices=indices,labels=labels,nearest_grid_direction_log_error=qerr,query_ids=o['query_ids'])
(P/'oracle_results.json').write_text(json.dumps(dict(records=records,scope='NONDEPLOYABLE: uses true query pose-derived geometry only to measure representation/quantization ceiling. Never used in method selection/features. Scores use4coarse extents, not fullgeometry.',test_accessed=False),indent=2));print('ORACLE COMPLETE')

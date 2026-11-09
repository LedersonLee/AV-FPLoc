"""Reuse actual prior repeated renders; no new renderer or checkpoint changes."""
from pathlib import Path
import json,hashlib,importlib.util,itertools,joblib
import numpy as np
from PIL import Image
P=Path('/output');W=Path('/workspace');OLD=W/'experiments/directional_tribranch_20261109';REPEAT=W/'experiments/pairwise_acoustic_evidence_20261107_2100';R=Path('/file2/jeongeon/AV-FPLoc/datasets/HSSD')
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
old=module('old_a',OLD/'stage_a.py');ray=module('old_prepare',OLD/'prepare.py');plan=json.loads((REPEAT/'repeat_plan.json').read_text());m=joblib.load(OLD/'stage_a/repeat_probe_embedding_ridge.joblib');results=[]
for scene in plan['scenes']:
 sid=scene['scene'];f=REPEAT/'repeat_features'/(sid+'.npz');receipt=json.loads(f.with_suffix('.json').read_text());assert hashlib.sha256(f.read_bytes()).hexdigest()==receipt['sha256'];z=np.load(f);pred=old.predict(m,dict(features=np.empty((len(z['embedding']),0)),embedding=z['embedding']));lookup={(r['key'],r['repeat']):i for i,r in enumerate(receipt['records'])};meta=json.loads((R/'2DFP'/sid/'floorplan_metadata.json').read_text());grid=np.array(Image.open(R/'2DFP'/sid/'map.png'))
 for pose in scene['poses']:
  rm=json.loads((R/pose['path']).with_suffix('.json').read_text());yaw=-(rm['base_acoustic_yaw_deg']-60*pose['roll'])%360;target,known=ray.rays(np.array([[rm['centre_xyz_m'][0],rm['centre_xyz_m'][2],yaw]]),grid,meta);gt=old.coarse(target)[0];ids=[lookup[(pose['key'],i)] for i in range(4)];v=np.clip(np.expm1(pred[ids]),0,10);pair=list(itertools.combinations(range(4),2));results.append(dict(scene=sid,key=pose['key'],role=pose['role'],GT_distance=gt.tolist(),repeat_predictions_m=v.tolist(),known=known.all().item(),mae_m=float(abs(v-gt).mean()),direction_std_m=v.std(0).tolist(),pair_distance_difference_m=float(np.mean([abs(v[i]-v[j]).mean() for i,j in pair])),near_label_agreement=float(np.mean([((v[i]<3)==(v[j]<3)).mean() for i,j in pair])),scope='Four cachedfreshrepeats; priorprobe trainednew12 excludes these4scenes, historicalacousticencoder exposedthem.'))
(P/'repeat_results.json').write_text(json.dumps(dict(records=results,source_plan_sha256=hashlib.sha256((REPEAT/'repeat_plan.json').read_bytes()).hexdigest(),model_sha256=hashlib.sha256((OLD/'stage_a/repeat_probe_embedding_ridge.joblib').read_bytes()).hexdigest(),scope='Only8Observed poses vs32FP,4scenes. Descriptive stability not real sensor reproducibility, independence or precise reflection GT.',test_accessed=False),indent=2));print('REPEAT',len(results))

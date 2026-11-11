"""Prepare only declared development candidates; query truth never enters paths."""
import argparse,json,sys,time,hashlib
from pathlib import Path
import numpy as np
from geometry import merge_segments,paths

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();sys.path.insert(0,str(a.out/'deps'));import trimesh
 w=Path('/home/jeongeonlee/track1');root=Path('/file2/jeongeon/AV-FPLoc/datasets/HSSD');old=w/'experiments/directional_structure_20261109';arch=w/'experiments/acoustic_geometry_20261110/targets_v3';scenes=json.loads((old/'PROTOCOL.json').read_text())['training_scenes'];dst=a.out/'hssd';dst.mkdir(exist_ok=True)
 split=json.loads((root/'manifests/scene_splits_v1.json').read_text());assert not set(scenes)&set(split['test']);assert not {s.split('_')[0] for s in scenes}&{s.split('_')[0] for s in split['test']}
 for sid in scenes:
  save=dst/(sid+'.json')
  if save.exists():continue
  tick=time.perf_counter();candidate_path=old/'candidates'/sid/'pure_visual.npz';c=np.load(candidate_path);audit=json.loads((old/'signals'/sid/'receipt.json').read_text());records={r['query_id']:r for r in audit['raw_records']};stage=np.load(arch/(sid+'.npz'));architecture=merge_segments(stage['segments_1.5'])
  mesh_path=root/'RIR/Geometry'/sid/'floorplan.glb';mesh=trimesh.load(mesh_path,force='mesh',process=False);segments=merge_segments(trimesh.intersections.mesh_plane(mesh,[0,1,0],[0,1.5,0])[:,:,[0,2]])
  schedule_path=root/f'RIR/manifests/positive_pairs_v1/schedules/train/{sid}.json';schedule=json.loads(schedule_path.read_text());queries={q['sample_index']:q for q in schedule['queries']};selected=np.linspace(0,len(c['query_ids'])-1,32).round().astype(int);output=[]
  for row in selected:
   qid=int(c['query_ids'][row]);rec=records[qid];q=queries[qid];assert q['observed_rir_path']==str(Path(rec['path']).relative_to(root));meta=json.loads(Path(rec['path']).with_suffix('.json').read_text());assert meta['acoustics']['sample_rate_hz']==48000;assert meta['centre_xyz_m'][1]==1.5
   hypotheses=[]
   for k,pose in enumerate(c['pred'][row]):hypotheses.append(dict(rank=k,pose=pose.tolist(),paths=paths(pose,segments,architecture)))
   # Ground-truth XY/yaw deliberately absent from prepared inference file.
   output.append(dict(row=int(row),query_id=qid,raw_path=rec['path'],raw_sha256=rec['sha256'],raw_shape=rec['raw_shape'],roll=q['channel_roll'],hypotheses=hypotheses))
  data=dict(scene=sid,queries=output,segments=segments.tolist(),architecture=architecture.tolist(),candidate_sha256=sha(candidate_path),fp_sha256=sha(mesh_path),architecture_source_sha256=sha(arch/(sid+'.npz')),schedule_sha256=sha(schedule_path),seconds=time.perf_counter()-tick,test_accessed=False)
  save.write_text(json.dumps(data));print(sid,len(segments),sum(len(h['paths']) for q in output for h in q['hypotheses']),round(data['seconds'],2),flush=True)
if __name__=='__main__':main()

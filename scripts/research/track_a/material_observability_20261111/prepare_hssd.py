"""Native-stage-only feasibility; vertical-face proxy, not semantic wall GT."""
import argparse,json,sys,hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0,'/home/jeongeonlee/track1/experiments/reflection_observability_20261111/deps')
import trimesh

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(exist_ok=False,parents=True);root=Path('/file2/jeongeon/AV-FPLoc/datasets/HSSD');old=Path('/home/jeongeonlee/track1/experiments/reflection_observability_20261111/hssd');split=json.loads((root/'manifests/scene_splits_v1.json').read_text());scenes=['102344250','102816615'];assert set(scenes)<=set(split['train']);assert not set(scenes)&set(split['test']);result=[]
 for sid in scenes:
  source=root/f'source/hssd-hab/stages/{sid}.glb';mesh=trimesh.load(source,force='mesh',process=False);target=abs(mesh.face_normals[:,1])<.25
  dest=a.out/(sid+'.mesh.txt')
  with dest.open('w') as f:
   f.write(f'{len(mesh.vertices)} {len(mesh.faces)}\n');np.savetxt(f,mesh.vertices,fmt='%.9g');np.savetxt(f,np.c_[mesh.faces,(~target).astype(int)],fmt='%d')
  previous=json.loads((old/(sid+'.json')).read_text());queries=[]
  for q in previous['queries'][:2]:
   meta_path=Path(q['raw_path']).with_suffix('.json');meta=json.loads(meta_path.read_text());queries.append(dict(query_id=q['query_id'],source=meta['centre_xyz_m'],channel_roll=q['roll'],hypotheses=q['hypotheses'],pose_metadata_sha256=sha(meta_path)))
  instance=root/f'source/hssd-hab/scenes/{sid}.scene_instance.json';nobjects=len(json.loads(instance.read_text())['object_instances']);result.append(dict(scene=sid,split='train',source=str(source),source_sha256=sha(source),mesh_file=dest.name,mesh_sha256=sha(dest),vertices=len(mesh.vertices),faces=len(mesh.faces),vertical_proxy_faces=int(target.sum()),vertical_proxy_area_m2=float(mesh.area_faces[target].sum()),total_area_m2=float(mesh.area),bounds=mesh.bounds.tolist(),furniture_object_instances=nobjects,furniture_included=False,assignment='Native stage vertical-face |normal_y|<.25 proxy; not architectural-wall identity; nonvertical stage remains fixed default',fp_sha256=previous['fp_sha256'],prepared_candidates_sha256=sha(old/(sid+'.json')),queries=queries))
 (a.out/'manifest.json').write_text(json.dumps(result,indent=2));print([(r['scene'],r['faces'],r['vertical_proxy_faces']) for r in result])
if __name__=='__main__':main()

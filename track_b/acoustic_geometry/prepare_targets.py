"""Read only existing non-Test schedules; derive vertical architectural stage rays."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import trimesh
from .targets import intersect_segments


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--cohort', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--code-sha', required=True)
    a=p.parse_args(); a.out.mkdir(parents=True, exist_ok=False)
    original=read(a.cohort)['scenes']
    # Selection solely by scene ID, before new target metrics/model results.
    order=sorted(original['train'], key=lambda s:hashlib.sha256(('acoustic-geometry-v1:'+s).encode()).hexdigest())
    splits={'train':order[16:], 'val':order[:16], 'development_test':original['val']}
    families=[{s.split('_')[0] for s in v} for v in splits.values()]
    assert all(not families[i]&families[j] for i in range(3) for j in range(i))
    protocol=dict(scenes=splits, code_sha=a.code_sha, cohort_sha256=sha(a.cohort),
        directions=36, max_range_m=20., queries_per_scene=64, normal_y_abs_max=.15,
        supervision='first vertical architectural stage intersection at receiver height; no furniture; no semantic outer closure',
        limitations='Architectural stage may contain fixed built-ins; vertical-surface proxy, not semantic wall classification. Door/window holes remain openings. Beyond-range/no-hit masked.',
        official_test_accessed=False, evidence='Exploratory; development_test is reused original Val19; inner Val from historically exposed Train112',
        selection='fixed final checkpoints; inner val reserved for implementation diagnosis; development_test stays unopened until model freeze')
    (a.out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    records=[]
    for part,scenes in splits.items():
      for scene in scenes:
        t=time.perf_counter()
        source_part='val' if part=='development_test' else 'train'
        schedule_path=a.root/f'RIR/manifests/positive_pairs_v1/schedules/{source_part}/{scene}.json'
        schedule=read(schedule_path)
        queries=schedule['queries']
        selected=[queries[i] for i in np.linspace(0,len(queries)-1,min(64,len(queries))).round().astype(int)]
        mesh_path=a.root/f'source/hssd-hab/stages/{scene}.glb'
        mesh=trimesh.load(mesh_path,force='mesh',process=False)
        keep=np.abs(mesh.face_normals[:,1])<=.15
        vertical=mesh.submesh([np.flatnonzero(keep)],append=True,repair=False)
        poses=np.array([[q['centre_xyz_m'][0],q['centre_xyz_m'][2],np.deg2rad(q['world_yaw_deg'])] for q in selected])
        heights=np.array([q['centre_xyz_m'][1] for q in selected])
        target=np.zeros((len(selected),36),np.float32);mask=np.zeros_like(target,bool)
        segments_by_height={}
        for height in np.unique(heights):
            segments=trimesh.intersections.mesh_plane(vertical,[0,1,0],[0,height,0])[:,:,[0,2]]
            segments_by_height[str(height)]=segments
            ids=heights==height
            target[ids],mask[ids]=intersect_segments(poses[ids],segments)
        # Manifest only: no held-out RIR/RGB payload is read here.
        refs=[dict(query_id=q['sample_index'],rir_path=q['observed_rir_path'],rgb_path=q['image_path'],
                   centre_xyz_m=q['centre_xyz_m'],world_yaw_deg=q['world_yaw_deg'],base_acoustic_yaw_deg=q['base_acoustic_yaw_deg']) for q in selected]
        np.savez_compressed(a.out/f'{scene}.npz',depth=target,mask=mask,poses=poses,heights=heights,
                            **{'segments_'+h:s for h,s in segments_by_height.items()})
        (a.out/f'{scene}_queries.json').write_text(json.dumps(refs,indent=2))
        row=dict(scene=scene,split=part,queries=len(selected),stage_sha256=sha(mesh_path),schedule_sha256=sha(schedule_path),
                 target_sha256=sha(a.out/f'{scene}.npz'),vertical_faces=int(keep.sum()),seconds=time.perf_counter()-t)
        # Do not inspect/report held-out target statistics during preparation.
        if part!='development_test': row['valid_fraction']=float(mask.mean())
        records.append(row)
        (a.out/'manifest.json').write_text(json.dumps(records,indent=2))
        print(json.dumps(row),flush=True)
    (a.out/'COMPLETE').write_text(a.code_sha+'\n')


if __name__=='__main__':main()

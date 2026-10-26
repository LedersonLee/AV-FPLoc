"""Pilot test-only independent 0.2-m/36-yaw bank and matched observations.

Grid FP IDs start at 1,000,000 to avoid the captured-query p00000 namespace.
No GT locations or visual scores are used to select grid candidates.
"""
import argparse
import copy
import fcntl
import json
from pathlib import Path
import time
import traceback

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

from .contracts import sha, immutable_json, atomic_json, recipe, PROTOCOL, RING
from .data import read, splits, safe_path
from .features import extract
from .render_adapter import positive_tools, supervisor_tools

REL = 'AcousticContrastive/pilots/scene_subset_v1/evaluation'


def grid_cells(free, meta, spacing=.2):
    step = int(round(spacing/meta['map_spacing_m']))
    if not np.isclose(step*meta['map_spacing_m'],spacing):
        raise ValueError('Grid must align to canonical map pixels')
    clearance = distance_transform_edt(free)*meta['map_spacing_m']
    rows,cols = np.meshgrid(np.arange(0,free.shape[0],step),np.arange(0,free.shape[1],step),indexing='ij')
    keep = free[rows,cols] & (clearance[rows,cols] >= .06)
    rc = np.column_stack([rows[keep],cols[keep]])
    bounds = meta['world_bounds_m']
    xy = np.column_stack([bounds['x_min']+rc[:,1]*meta['map_spacing_m'],
                          bounds['z_min']+rc[:,0]*meta['map_spacing_m']])
    if not len(xy):raise ValueError('Empty test bank')
    return rc,xy


def prepare(root,config):
    tools=positive_tools();selected=splits(root,config)['test']
    full=read(root/'UnLoc/frame_adapter_v4_furnished_qa/manifests/test_scenes.json')
    entries={e['scene_id']:e for e in full['scenes']}
    scenes=[]
    for scene in selected:
        entry=tools.prepare_scene(root,root/'source/hssd-hab','test',entries[scene],config['split_sha256'],'area_weighted_mean')
        positive=read(root/entry['schedule_path'])
        meta=read(root/'2DFP'/scene/'floorplan_metadata.json')
        free=np.asarray(Image.open(root/'2DFP'/scene/'map.png').convert('L'))==255
        rc,xy=grid_cells(free,meta)
        schedule=copy.deepcopy(positive)
        height=positive['height_contract']['floor_y_m']+1.5
        schedule.update(version='pilot_dense_grid_v1',queries=[],query_count=0,excluded=[],
            grid_spacing_m=.2,grid_rc=rc.tolist(),
            positions=[dict(pose_id=1000000+i,centre_xyz_m=[float(x),height,float(z)],base_headings=list(range(0,60,10)))
                       for i,(x,z) in enumerate(xy)],
            record_directory='records',pose_prefix='p',position_count=len(xy),physical_records_per_role=len(xy)*6,
            namespace='p1000000+ is test grid, not a GT positive; no GT injection')
        path=root/REL/'candidate_schedules'/f'{scene}.json'
        immutable_json(path,schedule)
        record=dict(scene_id=scene,split='test',schedule_path=str(path.relative_to(root)),schedule_sha256=sha(path),
            observation_schedule_path=entry['schedule_path'],observation_schedule_sha256=sha(root/entry['schedule_path']),
            queries=len(positive['queries']),excluded_queries=len(positive['excluded']),cells=len(xy),
            fp_arrays=len(xy)*6,observed_arrays=entry['physical_records_per_role'],candidate_xz=xy.tolist())
        scenes.append(record)
        print(json.dumps({k:v for k,v in record.items() if k!='candidate_xz'}),flush=True)
    index=dict(version='pilot_dense_test_bank_v1',split='test',split_sha256=config['split_sha256'],
        subset_sha256=config['scene_subset_sha256'],config_sha256=sha(root/'AcousticContrastive/pilots/scene_subset_v1/config.json'),
        grid_spacing_m=.2,yaw_step_deg=10,scenes=scenes,
        fp_arrays=sum(e['fp_arrays'] for e in scenes),observed_arrays=sum(e['observed_arrays'] for e in scenes))
    immutable_json(root/REL/'bank_index.json',index)
    return index


def tasks_for(root,renderer,index):
    tasks=[]
    for entry in index['scenes']:
        for role,key in [('Observed','observation_schedule_path'),('FP','schedule_path')]:
            path=entry[key];schedule=read(root/path)
            expected=entry['observation_schedule_sha256'] if role=='Observed' else entry['schedule_sha256']
            if sha(root/path)!=expected:raise ValueError('Test schedule changed')
            ids=[p['pose_id'] for p in schedule['positions']]
            for offset in range(0,len(ids),12):
                e=dict(scene_id=entry['scene_id'],split='test',schedule_path=path)
                tasks.append((str(root),str(root/'source/hssd-hab'),str(renderer),e,role,ids[offset:offset+12],2))
    return tasks


def extract_scene(root,entry,config,workers):
    from concurrent.futures import ThreadPoolExecutor
    from numpy.lib.format import open_memmap
    scene=entry['scene_id'];dest=root/REL/'features'/scene;dest.mkdir(parents=True,exist_ok=True)
    schedules={role:read(root/entry[key]) for role,key in [('FP','schedule_path'),('Observed','observation_schedule_path')]}
    receipts={}
    for role,schedule in schedules.items():
        paths=[root/f'RIR/{role}/test/{scene}/records/p{p["pose_id"]:05d}/base_acoustic_yaw_{base:03d}.npy'
               for p in schedule['positions'] for base in p['base_headings']]
        out=dest/role;out.mkdir(exist_ok=True)
        committed=out/'receipt.json'
        if committed.exists():
            receipt=read(committed)
            if receipt['features']!=config['features'] or receipt['paths']!=[str(p.relative_to(root)) for p in paths]:
                raise ValueError('Stale test feature cache')
            if config.get('corrected_extrusion') and receipt.get('frame_version')!='corrected_yup_v2':
                raise ValueError('Legacy test features cannot be reused for corrected evaluation')
            for name,digest in receipt['files'].items():
                if sha(out/name)!=digest:raise ValueError('Corrupt test features')
            receipts[role]=dict(path=str(committed.relative_to(root)),sha256=sha(committed));continue
        def work(path):
            meta=read(path.with_suffix('.json'))
            if meta['split']!='test' or meta['role']!=role or meta['scene_id']!=scene or meta['acoustics']!=PROTOCOL:
                raise ValueError('Test raw contract changed')
            if sha(path)!=meta['rir_sha256']:raise ValueError('Test raw checksum failed')
            expected_schedule=entry['schedule_sha256'] if role=='FP' else entry['observation_schedule_sha256']
            if meta.get('schedule_sha256')!=expected_schedule:
                raise ValueError('Test raw schedule mismatch')
            if role=='FP' and config.get('corrected_extrusion'):
                frame=meta.get('loaded_frame') or {}
                geometry=read(root/'RIR/Geometry'/scene/'floorplan.json')
                if (not frame.get('verified') or frame.get('up')!=[0,1,0]
                    or frame.get('front')!=[0,0,-1]
                    or meta.get('geometry_sha256')!=geometry['glb_sha256']):
                    raise ValueError('Test FP frame is not validated')
                np.testing.assert_allclose(frame.get('loaded_bounds'),geometry['mesh']['bounds_xyz_m'],atol=1e-4,rtol=0)
            a=np.load(path,allow_pickle=False)
            e,c=extract(a,config['features'])
            return e,c,meta['rir_sha256']
        arrays=None;raw=[]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for i,(e,c,digest) in enumerate(pool.map(work,paths)):
                if arrays is None:
                    arrays=[open_memmap(out/'energy.tmp.npy',mode='w+',dtype='float16',shape=(len(paths),*e.shape)),
                            open_memmap(out/'coherence.tmp.npy',mode='w+',dtype='float16',shape=(len(paths),*c.shape))]
                arrays[0][i]=e;arrays[1][i]=c;raw.append(digest)
                if i%256==0:atomic_json(dest/'progress.json',dict(role=role,complete=i,total=len(paths),updated_unix=time.time()))
        files={}
        for a,name in zip(arrays,('energy','coherence')):
            a.flush();a._mmap.close();(out/f'{name}.tmp.npy').replace(out/f'{name}.npy');files[name+'.npy']=sha(out/f'{name}.npy')
        receipt=dict(features=config['features'],paths=[str(p.relative_to(root)) for p in paths],raw_sha256=raw,files=files,
            frame_version='corrected_yup_v2' if config.get('corrected_extrusion') else 'legacy_v1')
        atomic_json(committed,receipt);receipts[role]=dict(path=str(committed.relative_to(root)),sha256=sha(committed))
    return receipts


def run(args):
    from .preprocess import cpu_guard
    cpu_guard(args.feature_workers,args.workers)
    root=args.root;config=read(args.config);out=root/REL;out.mkdir(parents=True,exist_ok=True)
    lock=(out/'queue.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def status(stage,**extra):atomic_json(out/'status.json',dict(status='running',stage=stage,updated_unix=time.time(),**extra))
    try:
        status('prepare_test_schedules');index=prepare(root,config)
        while True:
            progress=read(root/'AcousticContrastive/pilots/scene_subset_v1/preprocessing/status.json')
            if progress['status']=='failed':raise RuntimeError('Pilot preprocessing failed; sharing node2 safely requires inspection')
            if progress['status']=='complete':break
            status('waiting_feature_workers_to_release_node2');time.sleep(30)
        supervisor=supervisor_tools();tasks=tasks_for(root,args.renderer_dir,index)
        status('render_test_rirs',fp_arrays=index['fp_arrays'],observed_arrays=index['observed_arrays'])
        supervisor.supervise_tasks(tasks,out/'render',args.workers,expected_records=index['fp_arrays']+index['observed_arrays'])
        status('verify_test_rirs')
        for task in tasks:
            if supervisor.verify_paths(supervisor.record_paths(task),task[4]):raise RuntimeError('Uncommitted test RIR')
        status('extract_test_features');features={}
        for e in index['scenes']:features[e['scene_id']]=extract_scene(root,e,config,args.feature_workers)
        atomic_json(out/'ready.json',dict(status='complete',bank_sha256=sha(out/'bank_index.json'),features=features,
            config_sha256=sha(args.config),updated_unix=time.time()))
        atomic_json(out/'status.json',dict(status='complete',stage='test_bank_ready',updated_unix=time.time()))
    except Exception:
        atomic_json(out/'status.json',dict(status='failed',error=traceback.format_exc(),updated_unix=time.time()));raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--renderer-dir',type=Path,required=True)
    p.add_argument('--workers',type=int,default=38);p.add_argument('--feature-workers',type=int,default=64)
    run(p.parse_args())


if __name__=='__main__':main()

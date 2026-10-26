"""Corrected full-train FP positives/negatives, independent of subset training.

Uses frozen116-scene schedules, canonical raw paths and native CPU renderer.
Never mines new pairs, opens test data, changes raw sidecars, or starts training.
"""
import argparse
import fcntl
from functools import lru_cache
import os
from pathlib import Path
import time
import traceback

import numpy as np

from .contracts import PROTOCOL, atomic_json, immutable_json, sha
from .data import read
from .render_adapter import supervisor_tools

REL = 'RIR/manifests/corrected_train_v1'


def schedule_tasks(root, renderer, entries, chunk=24):
    tasks = []
    count = 0
    for entry in entries:
        schedule = read(root/entry['schedule_path'])
        if sha(root/entry['schedule_path']) != entry['schedule_sha256']:
            raise ValueError('Frozen train schedule changed')
        positions = schedule['positions']
        if len({p['pose_id'] for p in positions}) != len(positions):
            raise ValueError('Duplicate physical pose ID')
        expected = sum(len(p['base_headings']) for p in positions)
        if expected != entry['fp_arrays']:
            raise ValueError('Physical array count differs from schedule')
        count += expected
        ids = [p['pose_id'] for p in positions]
        for first in range(0,len(ids),chunk):
            tasks.append((str(root),str(root/'source/hssd-hab'),str(renderer),
                          entry,'FP',ids[first:first+chunk],2))
    return tasks,count


def validate_loaded_frame(metadata, expected_bounds):
    frame = metadata.get('loaded_frame') or {}
    bounds = np.asarray(frame.get('loaded_bounds',[]))
    if (not frame.get('verified') or frame.get('up') != [0,1,0]
        or frame.get('front') != [0,0,-1] or bounds.shape != (2,3)
        or not np.allclose(bounds,expected_bounds,rtol=0,atol=1e-4)
        or metadata.get('acoustics') != PROTOCOL):
        raise ValueError('Invalid FP frame/protocol: preserve legacy data; do not reuse or overwrite it')


def prepare(root):
    root = Path(root)
    out = root/REL
    split_path = root/'manifests/scene_splits_v1.json'
    membership = read(split_path)
    allowed = set(membership['train'])
    if len(allowed) != 116 or allowed & (set(membership['val']) | set(membership['test'])):
        raise ValueError('Frozen parent scene split changed')
    sources = {'manifests/scene_splits_v1.json':sha(split_path)}
    positive_path = root/'RIR/manifests/positive_pairs_v1/index.json'
    negative_path = root/'RIR/manifests/hard_negatives_epoch020_v1/index.json'
    positive,negative = read(positive_path),read(negative_path)
    entries = []
    for index_path,index in ((positive_path,positive),(negative_path,negative)):
        if index['split_sha256'] != sha(split_path):
            raise ValueError('Mining/positive split hash mismatch')
        sources[str(index_path.relative_to(root))] = sha(index_path)
        if {e['scene_id'] for e in index['scenes']} != allowed:
            raise ValueError('Incomplete full-train scene coverage')
        for entry in index['scenes']:
            path = root/entry['schedule_path']
            schedule = read(path)
            digest = sha(path)
            if entry.get('schedule_sha256',digest) != digest:
                raise ValueError('Source index and schedule differ')
            if (schedule['scene_id'] != entry['scene_id'] or schedule['split'] != 'train'
                or schedule['split_sha256'] != sha(split_path) or schedule['acoustics'] != PROTOCOL):
                raise ValueError('Cross-split/signal-protocol mismatch')
            mesh = root/schedule['geometry_path']
            geometry = read(mesh.with_suffix('.json'))
            if sha(mesh) != schedule['geometry_sha256'] or sha(mesh) != geometry['glb_sha256']:
                raise ValueError('Geometry hash mismatch')
            immutable_json(mesh.with_name(mesh.stem+'.y_up.stage_config.json'),
                dict(render_asset=mesh.name,up=[0,1,0],front=[0,0,-1],units_to_meters=1.))
            namespace = schedule.get('record_directory','records')
            prefix = schedule.get('pose_prefix','p')
            expected_namespace = ('records','p') if index_path == positive_path else ('negative_records','n')
            if (namespace,prefix) != expected_namespace:
                raise ValueError('Raw output namespace changed')
            entries.append(dict(scene_id=entry['scene_id'],split='train',schedule_path=entry['schedule_path'],
                schedule_sha256=digest,fp_arrays=schedule['physical_records_per_role'],
                kind='positive' if namespace=='records' else 'hard_negative',
                geometry_sha256=schedule['geometry_sha256']))
            sources[entry['schedule_path']] = digest
    # Interleave positive and negative schedules by scene, preserving original IDs.
    entries.sort(key=lambda e:(e['scene_id'],e['kind'] != 'positive'))
    plan = dict(version='corrected_full_train_v1',split_sha256=sha(split_path),
        scenes=116,entries=entries,fp_arrays=sum(e['fp_arrays'] for e in entries),
        positive_arrays=sum(e['fp_arrays'] for e in entries if e['kind']=='positive'),
        hard_negative_arrays=sum(e['fp_arrays'] for e in entries if e['kind']=='hard_negative'),
        sources=sources,acoustics=PROTOCOL,chunk_poses=24,test_accessed=False,
        raw_directory='RIR/FP/train',policy='Reuse only checksum-valid native-Y-up arrays; refuse legacy frame; no resplit, no new mining')
    immutable_json(out/'plan.json',plan)
    print({k:v for k,v in plan.items() if k not in ('entries','sources')},flush=True)
    return plan


def run(root, renderer, workers=6):
    root = Path(root);out = root/REL
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='-1' or os.environ.get('NVIDIA_VISIBLE_DEVICES')!='void':
        raise ValueError('Full-train rendering must remain CPU-only')
    if workers < 1 or workers*2 > len(os.sched_getaffinity(0)):
        raise ValueError('Need two CPU slots per native worker')
    lock = (out/'run.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan = read(out/'plan.json')
    def status(stage,**extra):
        atomic_json(out/'status.json',{'stage':stage,'updated_unix':time.time(),'test_accessed':False,**extra})
    try:
        for relative,digest in plan['sources'].items():
            if sha(root/relative) != digest:raise ValueError('Pinned train input changed: '+relative)
        tasks,total = schedule_tasks(root,renderer,plan['entries'],plan['chunk_poses'])
        if total != plan['fp_arrays']:raise ValueError('Full task coverage mismatch')
        supervisor = supervisor_tools()
        original = supervisor.verify_paths
        @lru_cache(maxsize=128)
        def bounds(scene):
            return read(root/f'RIR/Geometry/{scene}/floorplan.json')['mesh']['bounds_xyz_m']
        def strict_verify(paths,role):
            missing = original(paths,role)
            for path,*_ in paths:
                if str(path) in missing:continue
                scene = path.relative_to(root).parts[3]
                validate_loaded_frame(read(path.with_suffix('.json')),bounds(scene))
            return missing
        supervisor.verify_paths = strict_verify
        status('rendering',status='running',scenes=116,workers=workers,cpu_cap=workers*2,fp_arrays=total)
        results = supervisor.supervise_tasks(tasks,out/'render',workers,expected_records=total,
                                             task_lock_dir=out/'task_locks')
        receipt = dict(status='complete',plan_sha256=sha(out/'plan.json'),split_sha256=plan['split_sha256'],
            fp_arrays=total,records_written=sum(r['written'] for r in results),
            verified_existing=sum(r['skipped'] for r in results),tasks=len(results),
            loaded_frame='explicit native Y-up and post-load bounds verified',test_accessed=False,
            training_started=False,feature_cache_built=False,updated_unix=time.time())
        atomic_json(out/'render_complete.json',receipt)
        status('raw_FP_train_verified',**receipt)
    except Exception:
        status('failed',status='failed',error=traceback.format_exc());raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run'])
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--renderer',type=Path)
    parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args()
    if args.action=='prepare':prepare(args.root)
    else:run(args.root,args.renderer,args.workers)

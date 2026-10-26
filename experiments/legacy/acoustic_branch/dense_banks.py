"""Shared, split-preserving val/test FP banks, independent of experiments.

Schedules reuse the already committed pilot banks without changing their IDs
or fingerprints. New scenes use p3000000+. Node shards are disjoint, and shared
POSIX locks also reject accidental duplicate launches on the NFS mount.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import fcntl
import os
from pathlib import Path
import time
import traceback

import numpy as np
from PIL import Image

from .contracts import PROTOCOL, atomic_json, immutable_json, sha
from .data import read
from .render_adapter import positive_tools, supervisor_tools
from .test_rirs import grid_cells

REL = 'RIR/manifests/dense_banks_v1'
PILOT = 'ReliabilityFusion/pilots/v1'
OLD_TEST = 'AcousticContrastive/pilots/scene_subset_v1/evaluation'


def partition(count, node):
    if node not in ('node1', 'node2'):
        raise ValueError('Unknown CPU node')
    return list(range(0 if node == 'node1' else 1, count, 2))


def priority_allows_render(pilot, overlap_feature_extraction=False):
    """Never overlap native rendering; optionally share CPUs with feature I/O."""
    if pilot['status'] == 'failed':
        raise RuntimeError('Priority pilot job failed; inspect before taking its CPUs')
    return pilot['status'] == 'complete' or (
        overlap_feature_extraction and pilot['status'] == 'running'
        and pilot.get('stage') == 'verify_and_extract_features')


def balanced_remaining(tasks, completed):
    """Split unfinished task IDs by physical record count, preserving task IDs."""
    completed = set(completed)
    if not completed <= set(range(len(tasks))):
        raise ValueError('Completed task ID outside immutable task list')
    shards = {node: dict(task_ids=[], physical_rirs=0) for node in ('node1', 'node2')}
    for task_id, task in enumerate(tasks):
        if task_id in completed:
            continue
        node = min(shards, key=lambda n: (shards[n]['physical_rirs'], n))
        shards[node]['task_ids'].append(task_id)
        shards[node]['physical_rirs'] += len(task[5])*6
    return shards


def assignment_ids(assignment, tasks, node, bank_sha256):
    if assignment['bank_sha256'] != bank_sha256 or assignment['task_count'] != len(tasks):
        raise ValueError('Rebalanced assignment belongs to a different bank')
    completed = assignment['completed_task_ids']
    a, b = [assignment['shards'][n]['task_ids'] for n in ('node1', 'node2')]
    groups = [completed, a, b]
    if (any(len(g) != len(set(g)) for g in groups) or
        any(set(groups[i]) & set(groups[j]) for i, j in ((0, 1), (0, 2), (1, 2))) or
        set(completed+a+b) != set(range(len(tasks)))):
        raise ValueError('Rebalanced task IDs overlap or omit work')
    for n in ('node1', 'node2'):
        if sum(len(tasks[i][5])*6 for i in assignment['shards'][n]['task_ids']) != assignment['shards'][n]['physical_rirs']:
            raise ValueError('Rebalanced physical count differs')
    return assignment['shards'][node]['task_ids']


def prepare(root, workers=4):
    root = Path(root)
    full = read(root/'manifests/scene_splits_v1.json')
    split_hash = sha(root/'manifests/scene_splits_v1.json')
    all_ids = [s for k in ('train', 'val', 'test') for s in full[k]]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError('Original split overlap')
    reused = {}
    for catalog in (root/PILOT/'bank_index.json', root/OLD_TEST/'bank_index.json'):
        for entry in read(catalog)['scenes']:
            if entry['split'] in ('val', 'test'):
                key = (entry['split'], entry['scene_id'])
                if key in reused:
                    raise ValueError('Duplicate reusable scene')
                reused[key] = entry
    adapters = {}
    for split in ('val', 'test'):
        manifest = read(root/f'UnLoc/frame_adapter_v4_furnished_qa/manifests/{split}_scenes.json')
        if {e['scene_id'] for e in manifest['scenes']} != set(full[split]):
            raise ValueError('Adapter and original split differ')
        adapters[split] = {e['scene_id']: e for e in manifest['scenes']}
    tools = positive_tools()

    def scene(item):
        split, name = item
        meta = read(root/'2DFP'/name/'floorplan_metadata.json')
        free = np.asarray(Image.open(root/'2DFP'/name/'map.png').convert('L')) == 255
        rc, xy = grid_cells(free, meta)
        if (split, name) in reused:
            original = reused[(split, name)]
            path = root/original['schedule_path']
            schedule = read(path)
            if sha(path) != original['schedule_sha256']:
                raise ValueError('Existing schedule changed')
            np.testing.assert_allclose(
                [[p['centre_xyz_m'][0], p['centre_xyz_m'][2]] for p in schedule['positions']], xy,
                atol=1e-7, rtol=0)
            owner = 'pilot_val' if split == 'val' else 'completed_test'
        else:
            positive = root/f'RIR/manifests/positive_pairs_v1/schedules/{split}/{name}.json'
            if not positive.exists():
                tools.prepare_scene(root, root/'source/hssd-hab', split, adapters[split][name],
                                    split_hash, 'area_weighted_mean')
            schedule = copy.deepcopy(read(positive))
            height = schedule['height_contract']['floor_y_m'] + 1.5
            schedule.update(version='dense_fp_bank_v1', queries=[], query_count=0, excluded=[],
                grid_spacing_m=.2, grid_rc=rc.tolist(), record_directory='records', pose_prefix='p',
                positions=[dict(pose_id=3000000+i, centre_xyz_m=[float(x), height, float(z)],
                                base_headings=list(range(0, 60, 10))) for i, (x, z) in enumerate(xy)],
                position_count=len(xy), physical_records_per_role=6*len(xy),
                namespace='p3000000+ global evaluation grid; no GT injection')
            path = root/REL/'schedules'/split/(name+'.json')
            immutable_json(path, schedule)
            owner = 'distributed'
        if schedule['split'] != split or schedule['scene_id'] != name or schedule['split_sha256'] != split_hash:
            raise ValueError('Schedule split mismatch')
        if schedule['acoustics'] != PROTOCOL:
            raise ValueError('RIR protocol differs')
        geometry = root/schedule['geometry_path']
        contract = schedule['height_contract']
        if (sha(geometry) != schedule['geometry_sha256'] or
            contract['map_sha256'] != sha(root/'2DFP'/name/'map.png')):
            raise ValueError('Stale geometry or canonical map')
        immutable_json(geometry.with_name('floorplan.y_up.stage_config.json'),
            dict(render_asset='floorplan.glb', up=[0, 1, 0], front=[0, 0, -1], units_to_meters=1.))
        if (schedule['physical_records_per_role'] != 6*len(xy) or
            any(p['base_headings'] != list(range(0, 60, 10)) for p in schedule['positions'])):
            raise ValueError('Incomplete directional grid')
        entry = dict(scene_id=name, split=split, cells=len(xy), fp_arrays=6*len(xy),
            effective_pose_candidates=36*len(xy), schedule_path=str(path.relative_to(root)),
            schedule_sha256=sha(path), generation_owner=owner,
            raw_directory=f'RIR/FP/{split}/{name}/records')
        print(dict(scene=name, split=split, cells=len(xy), owner=owner), flush=True)
        return entry

    work = [(k, s) for k in ('val', 'test') for s in sorted(full[k])]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        entries = list(pool.map(scene, work))
    index = dict(version='dense_fp_bank_v1', split_sha256=split_hash,
        grid_spacing_m=.2, grid_cell_area_m2=.04, yaw_step_deg=10,
        physical_heading_bases_deg=list(range(0, 60, 10)),
        heading_contract='6 physical rotations x 6 cyclic microphone rolls = 36 headings; see schedule heading_contract',
        grid_rule='Canonical indoor-free grid, array radius 0.05m, centre clearance >=0.06m; no query GT injection',
        geometry='Approved 2DFP extruded with floor and ceiling; corrected native Y-up',
        scenes=entries, fp_arrays=sum(e['fp_arrays'] for e in entries),
        new_fp_arrays=sum(e['fp_arrays'] for e in entries if e['generation_owner']=='distributed'),
        counts={k:sum(e['split']==k for e in entries) for k in ('val', 'test')},
        training_or_selection=False)
    immutable_json(root/REL/'bank_index.json', index)
    return index


def tasks_for(root, renderer, index):
    tasks = []
    for entry in index['scenes']:
        if entry['generation_owner'] != 'distributed':
            continue
        path = root/entry['schedule_path']
        if sha(path) != entry['schedule_sha256']:
            raise ValueError('Schedule changed after planning')
        ids = [p['pose_id'] for p in read(path)['positions']]
        for first in range(0, len(ids), 12):
            tasks.append((str(root), str(root/'source/hssd-hab'), str(renderer),
                          entry, 'FP', ids[first:first+12], 2))
    return tasks


def verify_scene(root, entry, workers=16):
    """Validate every raw byte, six-channel signal, pose and native loaded frame."""
    supervisor = supervisor_tools()
    schedule = read(root/entry['schedule_path'])
    if sha(root/entry['schedule_path']) != entry['schedule_sha256']:
        raise ValueError('Schedule changed before final verification')
    ids = [p['pose_id'] for p in schedule['positions']]
    task = (str(root), str(root/'source/hssd-hab'), '', entry, 'FP', ids, 2)
    paths = supervisor.record_paths(task)

    def check(record):
        if supervisor.verify_paths([record], 'FP'):
            raise ValueError('Missing committed RIR: '+str(record[0]))
        metadata = read(record[0].with_suffix('.json'))
        frame = metadata.get('loaded_frame') or {}
        if (not frame.get('verified') or frame.get('up') != [0, 1, 0] or
            frame.get('front') != [0, 0, -1] or metadata['acoustics'] != PROTOCOL or
            metadata['split'] != entry['split'] or metadata['scene_id'] != entry['scene_id']):
            raise ValueError('RIR frame/protocol/split mismatch')
        np.testing.assert_allclose(frame['loaded_bounds'],
            schedule['height_contract']['mesh']['bounds_xyz_m'], atol=1e-4, rtol=0)
        return dict(path=str(record[0].relative_to(root)), sha256=metadata['rir_sha256'],
                    bytes=record[0].stat().st_size)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        records = list(pool.map(check, paths))
    receipt = dict(status='complete', scene_id=entry['scene_id'], split=entry['split'],
        schedule_sha256=entry['schedule_sha256'], fp_arrays=len(records),
        raw_bytes=sum(r['bytes'] for r in records), records=records,
        verified_unix=time.time(), loaded_frame='corrected native Y-up')
    atomic_json(root/REL/'verified'/entry['split']/(entry['scene_id']+'.json'), receipt)
    return receipt


def run(args):
    root = args.root
    out = root/REL
    lock = (out/(args.node+'.lock')).open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '-1' or os.environ.get('NVIDIA_VISIBLE_DEVICES') != 'void':
        raise ValueError('CPU-only job; GPU must be inaccessible')
    index = read(out/'bank_index.json')
    if sha(root/'manifests/scene_splits_v1.json') != index['split_sha256']:
        raise ValueError('Global scene split changed')
    def status(stage, **extra):
        atomic_json(out/'nodes'/(args.node+'.json'),
            dict(status='running', stage=stage, updated_unix=time.time(), **extra))
    try:
        if args.node == 'node2':
            while True:
                pilot = read(root/PILOT/'preprocess_status.json')
                if priority_allows_render(pilot, args.overlap_feature_extraction):
                    break
                status('waiting_priority_pilot_cpu_release')
                time.sleep(30)
        tasks = tasks_for(root, args.renderer, index)
        assignment_path = getattr(args, 'assignment', None)
        if assignment_path:
            assignment = read(out/assignment_path)
            ids = assignment_ids(assignment, tasks, args.node, sha(out/'bank_index.json'))
            round_id = sha(out/assignment_path)[:16]
        else:
            ids = partition(len(tasks), args.node)
            round_id = None
        render_out = out/'render'/('balanced_'+round_id)/args.node if round_id else out/'render'/args.node
        def marker(node):
            return out/'nodes'/(node+('_'+round_id if round_id else '')+'_render_complete.json')
        own = [tasks[i] for i in ids]
        count = sum(len(t[5])*6 for t in own)
        status('render_dense_fp_bank', task_count=len(own), expected_arrays=count, workers=args.workers,
               overlap_feature_extraction=args.overlap_feature_extraction,
               assignment=assignment_path, render_directory=str(render_out.relative_to(out)))
        supervisor_tools().supervise_tasks(own, render_out, args.workers,
            expected_records=count, task_ids=ids, task_lock_dir=out/'task_locks')
        # Both shards can finish different scenes last: wait for *both* success
        # markers before auditing complete scenes. Never mistake partial data for ready.
        atomic_json(marker(args.node),
            dict(status='complete', bank_sha256=sha(out/'bank_index.json'), arrays=count))
        while True:
            markers = [marker(n) for n in ('node1','node2')]
            priority = read(root/PILOT/'preprocess_status.json')
            if all(p.exists() for p in markers) and priority['status']=='complete':
                for p in markers:
                    if read(p)['bank_sha256'] != sha(out/'bank_index.json'):
                        raise ValueError('Other node used a different bank manifest')
                break
            status('waiting_other_shard_for_full_verification')
            time.sleep(30)
        status('verify_all_bytes_and_loaded_frames')
        own_scenes = [index['scenes'][i] for i in partition(len(index['scenes']), args.node)]
        for entry in own_scenes:
            status('verify_all_bytes_and_loaded_frames', scene=entry['scene_id'])
            verify_scene(root, entry)
        atomic_json(out/'nodes'/(args.node+'.json'), dict(status='complete', stage='shard_verified', updated_unix=time.time()))
        receipts = [out/'verified'/e['split']/(e['scene_id']+'.json') for e in index['scenes']]
        if all(p.exists() for p in receipts):
            values = [read(p) for p in receipts]
            if sum(v['fp_arrays'] for v in values) != index['fp_arrays']:
                raise ValueError('Final record count mismatch')
            atomic_json(out/'ready.json', dict(status='complete', bank_sha256=sha(out/'bank_index.json'),
                fp_arrays=index['fp_arrays'], counts=index['counts'],
                raw_bytes=sum(v['raw_bytes'] for v in values), updated_unix=time.time()))
    except Exception:
        atomic_json(out/'nodes'/(args.node+'.json'),
            dict(status='failed', error=traceback.format_exc(), updated_unix=time.time()))
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'run'])
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--renderer', type=Path)
    p.add_argument('--node', choices=['node1', 'node2'])
    p.add_argument('--workers', type=int, default=38)
    p.add_argument('--assignment', help='Relative rebalanced assignment path under the bank manifest directory')
    p.add_argument('--overlap-feature-extraction', action='store_true',
                   help='Node2 only: start after priority rendering, during feature extraction; reserve CPUs externally')
    a = p.parse_args()
    if a.assignment and (Path(a.assignment).is_absolute() or '..' in Path(a.assignment).parts):
        p.error('Assignment path must remain under the bank manifest directory')
    if a.overlap_feature_extraction and (a.node != 'node2' or a.workers > 32):
        p.error('Feature overlap requires node2 and at most32 two-thread workers')
    if a.action == 'prepare':
        index = prepare(a.root)
        print({k:v for k,v in index.items() if k != 'scenes'})
    else:
        if not a.node or not a.renderer:
            p.error('run requires --node and --renderer')
        run(a)


if __name__ == '__main__':
    main()

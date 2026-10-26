"""Frozen full-scene preparation and one-GPU training queue.

The native CPU val renderer, CPU feature job and GPU training queue have separate
containers/locks. GPU training requires the exhaustive distributed train receipt,
all 19 captured-val scenes, raw SHA/frame checks, and all three feature streams.
No test arrays or test metrics are opened for training or checkpoint selection.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import fcntl
from pathlib import Path
import subprocess
import time
import traceback

import numpy as np

from .contracts import atomic_json, immutable_json, sha, PROTOCOL
from .data import (read, splits, load_manifest, unique_refs, checked_rir,
                   build_cache, preflight, DEST)
from .full_train_rirs import REL as TRAIN_RAW, validate_loaded_frame

REL = 'AcousticContrastive/full_scene_v1'
GPU = 'GPU-ea0fc3b1-6762-928c-f83d-0843a0c7a1db'


def update(root, name, stage, **fields):
    atomic_json(Path(root)/REL/f'{name}_status.json', dict(stage=stage,
        updated_unix=time.time(), test_accessed=False, **fields))


def prepare(root):
    root = Path(root)
    out = root/REL
    config = read(root/'AcousticContrastive/pilots/scene_subset_v1/corrected_yup_v2/config.json')
    config = copy.deepcopy(config)
    for key in ('scene_subset_path', 'scene_subset_sha256', 'subset_counts', 'corrected_extrusion'):
        config.pop(key, None)
    membership = splits(root, config)
    if {k:len(membership[k]) for k in ('train', 'val', 'test')} != dict(train=116, val=19, test=17):
        raise ValueError('Frozen full split required')
    sources = {f'{TRAIN_RAW}/plan.json': sha(root/TRAIN_RAW/'plan.json')}
    raw = read(root/TRAIN_RAW/'plan.json')
    if raw['split_sha256'] != config['split_sha256'] or raw['fp_arrays'] != 357548:
        raise ValueError('Frozen corrected-train render plan changed')
    manifest_counts = {}
    for split in ('train', 'val'):
        queries, digest = load_manifest(root, split, config)
        sources[f'{DEST}/{split}.json'] = digest
        manifest_counts[split] = dict(queries=len(queries), arrays=len(unique_refs(queries)))
    stages = {}
    for scene in sorted(set(membership['train']) | set(membership['val'])):
        mesh = root/f'RIR/Geometry/{scene}/floorplan.glb'
        metadata = read(mesh.with_suffix('.json'))
        if sha(mesh) != metadata['glb_sha256']:
            raise ValueError('FP geometry changed: '+scene)
        immutable_json(mesh.with_name(mesh.stem+'.y_up.stage_config.json'),
            dict(render_asset=mesh.name, up=[0,1,0], front=[0,0,-1], units_to_meters=1.))
        sources[str(mesh.relative_to(root))] = metadata['glb_sha256']
        stages[scene] = dict(mesh_sha256=metadata['glb_sha256'], expected_bounds=metadata['mesh']['bounds_xyz_m'])
    val_index = root/DEST/'val_rir_index.json'
    sources[str(val_index.relative_to(root))] = sha(val_index)
    entries = read(val_index)['scenes']
    if {e['scene_id'] for e in entries} != set(membership['val']):
        raise ValueError('Full held-out val render coverage required')
    for entry in entries:
        schedule = root/entry['schedule_path']
        if sha(schedule) != entry['schedule_sha256'] or entry['split'] != 'val':
            raise ValueError('Val schedule changed')
        sources[entry['schedule_path']] = sha(schedule)
    plan = dict(version='full_scene_native_yup_v1', sources=sources, stages=stages,
        split_sha256=config['split_sha256'], counts=manifest_counts,
        scenes=dict(train=116, val=19), train_raw_plan_sha256=sha(root/TRAIN_RAW/'plan.json'),
        train_fp_arrays=raw['fp_arrays'], acoustics=PROTOCOL, test_accessed=False,
        policy='Same frozen RGB poses/pairs; native Y-up extrusion; no test selection; no resplit')
    immutable_json(out/'plan.json', plan)
    config['native_frame'] = dict(plan_path=REL+'/plan.json', plan_sha256=sha(out/'plan.json'),
        verification_directory=REL+'/preprocessing')
    config['model']['features'] = 'all'
    config['cache_envelope'] = True
    config['training'].update(epochs=40, early_stop_patience=8, batch_size=64,
        workers=4, cpu_threads=4, device='cuda')
    config['wandb'].update(mode='online', name='hssd-full116-energy-coherence-envelope-seed42',
        group='hssd-full-scene-v1')
    immutable_json(out/'config.json', config)
    return config


def plan_check(root, config):
    path = root/config['native_frame']['plan_path']
    if sha(path) != config['native_frame']['plan_sha256']:
        raise ValueError('Full native plan changed')
    plan = read(path)
    for relative, digest in plan['sources'].items():
        if sha(root/relative) != digest:
            raise ValueError('Pinned full-data source changed: '+relative)
    splits(root, config)
    return plan


def train_ready(root, plan):
    folder = root/TRAIN_RAW/'distributed_v1'
    path = folder/'ready.json'
    if not path.exists():
        for node in ('node1', 'node2', 'local'):
            status = folder/'render'/node/'status.json'
            if status.exists() and read(status).get('status') == 'failed':
                raise RuntimeError('Upstream corrected train render failed on '+node)
        return False
    report = read(path)
    if (report.get('status') != 'complete' or report.get('plan_sha256') != plan['train_raw_plan_sha256']
        or report.get('fp_arrays') != plan['train_fp_arrays'] or report.get('test_accessed') is not False
        or report.get('assignment_sha256') != sha(folder/'assignment.json')
        or report.get('task_proofs_sha256') != sha(folder/'task_proofs.json')):
        raise ValueError('Exhaustive distributed corrected-train receipt failed')
    return True


def render_val(root, workers=4):
    from .render_adapter import supervisor_tools
    root = Path(root)
    config = read(root/REL/'config.json')
    plan = plan_check(root, config)
    lock = (root/REL/'val_render.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    supervisor = supervisor_tools()
    original = supervisor.verify_paths
    def strict(paths, role):
        missing = original(paths, role)
        for path, *_ in paths:
            if str(path) not in missing and role == 'FP':
                meta = read(path.with_suffix('.json'))
                stage = plan['stages'][path.relative_to(root).parts[3]]
                validate_loaded_frame(meta, stage['expected_bounds'])
                if meta.get('geometry_sha256') != stage['mesh_sha256']:
                    raise ValueError('Val FP geometry differs from frozen plan')
        return missing
    supervisor.verify_paths = strict
    index = read(root/DEST/'val_rir_index.json')
    tasks = []
    # Captured positives only. Never duplicate a dense-bank render or a train ID.
    for entry in index['scenes']:
        schedule = read(root/entry['schedule_path'])
        ids = [p['pose_id'] for p in schedule['positions']]
        for role in ('FP', 'Observed'):
            for start in range(0, len(ids), 12):
                tasks.append((str(root), str(root/'source/hssd-hab'),
                    str(root/'RIR/tools/hard_negatives_epoch020_v1'), entry, role, ids[start:start+12], 2))
    expected = sum(2*e['physical_records_per_role'] for e in index['scenes'])
    update(root, 'val_render', 'rendering', status='running', workers=workers, arrays=expected)
    supervisor.supervise_tasks(tasks, root/REL/'val_render', workers, expected_records=expected,
        task_lock_dir=root/REL/'val_task_locks')
    atomic_json(root/REL/'val_render_ready.json', dict(status='complete', arrays=expected,
        plan_sha256=config['native_frame']['plan_sha256'], test_accessed=False))
    update(root, 'val_render', 'complete', status='complete')


def verify_split(root, config, split, workers):
    out = root/REL/'preprocessing'
    queries, digest = load_manifest(root, split, config)
    refs = unique_refs(queries)
    report_path = out/f'{split}_verification.json'
    if report_path.exists():
        prior = read(report_path)
        if (prior.get('manifest_sha256') != digest or prior.get('arrays') != len(refs)
            or prior.get('plan_sha256') != config['native_frame']['plan_sha256']
            or prior.get('status') != 'complete' or set(prior.get('records', {})) != set(refs)):
            raise ValueError('Committed full raw verification changed')
        return
    def check(item):
        p, ref = item
        return p, checked_rir(root, p, ref, config)[1]
    records = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for p, receipt in pool.map(check, sorted(refs.items())):
            records[p] = receipt
            if len(records) % 1000 == 0:
                update(root, 'features', 'verify_'+split, status='running',
                    arrays=len(records), expected=len(refs), workers=workers)
    immutable_json(report_path, dict(status='complete', split=split, manifest_sha256=digest,
        plan_sha256=config['native_frame']['plan_sha256'], split_sha256=config['split_sha256'],
        arrays=len(refs), queries=len(queries), records=records, test_accessed=False))


def features(root, workers=64):
    root = Path(root)
    lock = (root/REL/'features.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = read(root/REL/'config.json')
    plan = plan_check(root, config)
    while True:
        raw_ok = train_ready(root, plan)
        val_path = root/REL/'val_render_ready.json'
        val_ok = val_path.exists()
        if val_ok:
            marker = read(val_path)
            if marker.get('status') != 'complete' or marker.get('plan_sha256') != config['native_frame']['plan_sha256']:
                raise ValueError('Val completion proof changed')
        val_status = root/REL/'val_render_status.json'
        if val_status.exists() and read(val_status).get('status') == 'failed':
            raise RuntimeError('Full val render failed; training remains blocked')
        if raw_ok and val_ok:
            break
        update(root, 'features', 'waiting_corrected_raw', status='waiting', train_ready=raw_ok, val_ready=val_ok)
        time.sleep(30)
    for split in ('train', 'val'):
        update(root, 'features', 'verify_'+split, status='running', workers=workers)
        verify_split(root, config, split, workers)
        update(root, 'features', 'extract_three_inputs_'+split, status='running', workers=workers)
        build_cache(root, config, split, workers=workers)
    from .envelope_cache import check as envelope_check
    report = preflight(root, config)
    if report['status'] != 'ready':
        raise RuntimeError(str(report['blockers']))
    envelope = {split:envelope_check(root, config, split)[1] for split in ('train', 'val')}
    immutable_json(root/REL/'ready.json', dict(status='complete', data=report, envelope=envelope,
        config_sha256=sha(root/REL/'config.json'), plan_sha256=config['native_frame']['plan_sha256'], test_accessed=False))
    update(root, 'features', 'ready_for_training', status='complete', data=report)


def gpu_idle(output, expected_uuid=GPU):
    rows = [r.strip().split(',') for r in output.strip().splitlines() if r.strip()]
    if len(rows) != 1 or len(rows[0]) != 3 or rows[0][0].strip() != expected_uuid:
        raise ValueError('Exactly the authorized physical GPU must be queried')
    return float(rows[0][1]) < 100 and float(rows[0][2]) == 0


def train_queue(root):
    root = Path(root)
    out = root/REL
    lock = (out/'gpu_queue.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = read(out/'config.json')
    while not (out/'ready.json').exists():
        status = read(out/'features_status.json') if (out/'features_status.json').exists() else {}
        if status.get('status') == 'failed':
            raise RuntimeError('Full preprocessing failed; GPU queue refuses to start')
        update(root, 'training', 'waiting_verified_three_input_cache', status='waiting',
            preprocessing_stage=status.get('stage'), gpu=GPU, gpu_count=1)
        time.sleep(30)
    ready = read(out/'ready.json')
    if (ready.get('status') != 'complete' or ready.get('config_sha256') != sha(out/'config.json')
        or ready.get('plan_sha256') != config['native_frame']['plan_sha256']):
        raise ValueError('Training-ready contract mismatch')
    # Mounting a GPU while waiting does not allocate CUDA memory. Never evict a
    # process that another user may start on the previously idle device.
    while True:
        output = subprocess.check_output(['nvidia-smi', '--id='+GPU,
            '--query-gpu=uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
        if gpu_idle(output):
            break
        update(root, 'training', 'waiting_authorized_gpu_idle', status='waiting', gpu=GPU, gpu_count=1)
        time.sleep(30)
    import torch
    from .train import train
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Queue must expose exactly one authorized CUDA GPU')
    plan_check(root, config)
    run = out/'runs/energy_coherence_envelope_seed42'
    state = read(run/'status.json') if (run/'status.json').exists() else {}
    if state.get('status') != 'complete':
        update(root, 'training', 'training', status='running', gpu=GPU, gpu_count=1,
            device=torch.cuda.get_device_name(0), fresh=not (run/'last.pt').exists())
        train(root, config, run, run/'last.pt' if (run/'last.pt').exists() else None)
    update(root, 'training', 'complete', status='complete', gpu=GPU, gpu_count=1,
        run=str(run.relative_to(root)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'render-val', 'features', 'train-queue'])
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--workers', type=int, default=64)
    a = parser.parse_args()
    name = {'render-val':'val_render', 'features':'features', 'train-queue':'training'}.get(a.action)
    try:
        if a.action == 'prepare': prepare(a.root)
        elif a.action == 'render-val': render_val(a.root, a.workers)
        elif a.action == 'features': features(a.root, a.workers)
        else: train_queue(a.root)
    except Exception:
        if name:
            update(a.root, name, 'failed', status='failed', error=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()

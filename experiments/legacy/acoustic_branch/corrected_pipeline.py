"""Versioned corrected-frame pilot RIRs -> verification -> cache -> fresh train.

Frozen train/val queries only; test RIRs are neither read nor generated here.
The legacy audit remains blocked for all old configurations.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np

from .contracts import atomic_json, immutable_json, sha, PROTOCOL, RING, recipe
from .data import read, load_manifest, unique_refs, checked_rir, build_cache, preflight, TRAIN_SOURCE

REL = 'AcousticContrastive/pilots/scene_subset_v1/corrected_yup_v2'
PREFIX = 'RIR/FP'


def schedule(root, path):
    p = Path(path).parts
    if p[4] == 'negative_records':
        return root/TRAIN_SOURCE/'schedules/train'/f'{p[3]}.json'
    return root/'RIR/manifests/positive_pairs_v1/schedules'/p[2]/f'{p[3]}.json'


def prepare(root):
    out = root/REL
    config = read(root/'AcousticContrastive/pilots/scene_subset_v1/config.json')
    refs, manifests, stages, tasks = {}, {}, {}, []
    for split in ('train', 'val'):
        queries, digest = load_manifest(root, split, config)
        manifests[split] = dict(sha256=digest, queries=len(queries))
        refs.update({p:r for p,r in unique_refs(queries).items() if p.startswith('RIR/FP/')})
    for scene in sorted({Path(p).parts[3] for p in refs}):
        mesh = root/f'RIR/Geometry/{scene}/floorplan.glb'
        metadata = read(mesh.with_suffix('.json'))
        if sha(mesh) != metadata['glb_sha256']:
            raise ValueError('Extrusion mesh SHA mismatch')
        stage = out/'stages'/f'{scene}.stage_config.json'
        # Dataset-relative relative asset path remains portable across mounts.
        asset = os.path.relpath(mesh, stage.parent)
        immutable_json(stage, dict(render_asset=asset, up=[0,1,0], front=[0,0,-1], units_to_meters=1.))
        stages[scene] = dict(config_sha256=sha(stage), mesh_sha256=sha(mesh),
                             expected_bounds=metadata['mesh']['bounds_xyz_m'])
        paths = sorted(p for p in refs if Path(p).parts[3] == scene)
        for offset in range(0, len(paths), 12):
            tasks.append(dict(scene=scene, paths=paths[offset:offset+12]))
    schedules = {}
    # Hash each immutable schedule once, not per physical RIR.
    for path in refs:
        source = schedule(root, path)
        relative = str(source.relative_to(root))
        if relative not in schedules:
            schedules[relative] = sha(source)
    plan = dict(version='corrected_yup_v2', refs=refs, tasks=tasks, stages=stages,
                manifests=manifests, schedules=schedules, split_sha256=config['split_sha256'],
                test_accessed=False, prefix=PREFIX, policy='same queries/candidates; explicit Y-up; post-load bounds checked')
    if (out/'plan.json').exists():
        # Preserve the immutable historical plan/config identity after a storage
        # move. The storage-layout manifest resolves its old logical prefix.
        plan['prefix']=read(out/'plan.json')['prefix']
    immutable_json(out/'plan.json', plan)
    config['corrected_extrusion'] = dict(prefix=plan['prefix'], plan_path=REL+'/plan.json', plan_sha256=sha(out/'plan.json'))
    immutable_json(out/'config.json', config)
    print(dict(tasks=len(tasks), FP_arrays=len(refs), manifests=manifests), flush=True)


def target(root, path):
    return root/PREFIX/Path(path).relative_to('RIR/FP')


def child(a):
    sys.path[:0] = [str(a.helpers), str(a.renderer)]
    from hssd_positive_pairs import make_geometry_simulator
    from render_hssd_paired_rirs import _add_audio_sensor
    from render_hssd_exp4_ring_rirs import _render_ring, _atomic_npy
    from hssd_cpu_rir_contract import require_cpu_device
    require_cpu_device(-1)
    out = a.root/REL
    config = read(out/'config.json')
    plan = read(out/'plan.json')
    digest = sha(out/'plan.json')
    if digest != config['corrected_extrusion']['plan_sha256']:
        raise ValueError('Plan changed')
    task = plan['tasks'][a.task]
    scene = task['scene']
    stage = out/'stages'/f'{scene}.stage_config.json'
    if sha(stage) != plan['stages'][scene]['config_sha256']:
        raise ValueError('Stage config changed')
    mesh = a.root/f'RIR/Geometry/{scene}/floorplan.glb'
    if sha(mesh) != plan['stages'][scene]['mesh_sha256']:
        raise ValueError('Mesh changed')
    sim = make_geometry_simulator(stage, None)
    box = sim.get_active_scene_graph().get_root_node().cumulative_bb
    bounds = np.array([np.asarray(box.min), np.asarray(box.max)])
    np.testing.assert_allclose(bounds, plan['stages'][scene]['expected_bounds'], atol=1e-4, rtol=0)
    sensor = _add_audio_sensor(sim, SimpleNamespace(gpu_device_id=-1, sample_rate=48000,
        indirect_rays=20000, ray_depth=50, source_rays=512, diffraction=True,
        max_diffraction_order=10, acoustic_threads=2))
    for path in task['paths']:
        r = plan['refs'][path]
        dest = target(a.root, path)
        if dest.exists() and dest.with_suffix('.json').exists():
            checked_rir(a.root, path, r, config)
            continue
        base = recipe(r['yaw'])[0]
        xyz = np.array(r['xyz'])
        angles = RING+np.deg2rad(base)
        offsets = np.column_stack([.05*np.cos(angles), np.zeros(6), -.05*np.sin(angles)])
        points = np.vstack([xyz, xyz+offsets])
        if not np.all(points > bounds[0]) or not np.all(points < bounds[1]):
            raise ValueError('Source or microphone outside post-load geometry bounds')
        # Corrected VAL diagnostics are already physically verified; reuse raw
        # bytes only after exact plan/reference/SHA/bounds checks.
        prior = a.root/'AcousticContrastive/diagnostics/extrusion_frame_fix_v1'
        candidate = prior/'corrected_fp'/Path(path).relative_to('RIR/FP/val') if Path(path).parts[2]=='val' else None
        if candidate is not None and candidate.exists():
            old = read(candidate.with_suffix('.json'))
            if (old['reference']['xyz'] != r['xyz'] or sha(candidate) != old['sha256']
                or old['plan_sha256'] != sha(prior/'plan.json')):
                raise ValueError('Corrected VAL reuse failed lineage check')
            np.testing.assert_allclose(old['loaded_bounds'], bounds, rtol=0, atol=1e-4)
            rir = np.load(candidate, allow_pickle=False)
        else:
            rir = _render_ring(sim, sensor, xyz.astype(np.float32), base, .05)
        if rir.ndim!=2 or rir.shape[0]!=6 or not np.isfinite(rir).all() or not np.all(np.any(rir!=0,axis=1)):
            raise ValueError('Invalid corrected six-channel RIR')
        dest.parent.mkdir(parents=True, exist_ok=True)
        _atomic_npy(dest, rir)
        source = str(schedule(a.root, path).relative_to(a.root))
        atomic_json(dest.with_suffix('.json'), dict(version='corrected_yup_v2',role='FP',
            geometry_kind='canonical_map_extrusion', split=Path(path).parts[2], scene_id=scene,
            centre_xyz_m=r['xyz'], source_xyz_m=r['xyz'], base_acoustic_yaw_deg=base,
            microphone_angles_rad=angles.tolist(), acoustics=PROTOCOL, rir_shape=list(rir.shape),
            rir_sha256=sha(dest), schedule_sha256=plan['schedules'][source],
            geometry_sha256=plan['stages'][scene]['mesh_sha256'], corrected_plan_sha256=digest,
            loaded_frame=dict(verified=True,up=[0,1,0],front=[0,0,-1],loaded_bounds=bounds.tolist(),stage_config_sha256=sha(stage)),
            cpu_only=True,gpu_device_id=-1,acoustic_threads=2))
        print('corrected', path, flush=True)
    sys.stdout.flush()
    os._exit(0)


def pipeline(a):
    out = a.root/REL
    lock = (out/'pipeline.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = read(out/'config.json')
    plan = read(out/'plan.json')
    status = dict(status='running', stage='render', completed_tasks=0, total_tasks=len(plan['tasks']), test_accessed=False)
    def publish():
        atomic_json(out/'status.json', dict(status, updated_unix=time.time()))
    publish()
    def task(i):
        cmd = [sys.executable,'-u','-m','acoustic_branch.corrected_pipeline','child',
               '--root',str(a.root),'--helpers',str(a.helpers),'--renderer',str(a.renderer),'--task',str(i)]
        path = out/'logs'/f'{i:05d}.log'
        path.parent.mkdir(exist_ok=True)
        with path.open('a') as f:
            for attempt in range(3):
                try:
                    subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=300, check=True)
                    return
                except (subprocess.TimeoutExpired,subprocess.CalledProcessError):
                    if attempt==2:
                        raise
    try:
        with ThreadPoolExecutor(max_workers=38) as pool:
            for future in as_completed([pool.submit(task,i) for i in range(len(plan['tasks']))]):
                future.result()
                status['completed_tasks'] += 1
                publish()
        for split in ('train','val'):
            status.update(stage='verify_'+split)
            publish()
            queries, digest = load_manifest(a.root, split, config)
            refs = unique_refs(queries)
            def check(item):
                p,r = item
                return p, checked_rir(a.root,p,r,config)[1]
            with ThreadPoolExecutor(max_workers=32) as pool:
                receipts = dict(pool.map(check, refs.items()))
            atomic_json(out/f'{split}_verification.json',dict(status='complete',split=split,
                manifest_sha256=digest,plan_sha256=sha(out/'plan.json'),split_sha256=config['split_sha256'],
                arrays=len(refs),queries=len(queries),records=receipts,test_accessed=False))
            status.update(stage='cache_'+split)
            publish()
            build_cache(a.root,config,split,workers=48)
        report = preflight(a.root,config)
        atomic_json(out/'preflight.json',report)
        if report['status']!='ready':
            raise RuntimeError(str(report['blockers']))
        status.update(status='complete',stage='ready_for_fresh_training')
        publish()
    except Exception:
        status.update(status='failed',error=traceback.format_exc())
        publish()
        raise


def train_queue(a):
    out = a.root/REL
    run = out/'runs/acoustic_fresh_v1'
    run.mkdir(parents=True,exist_ok=True)
    lock = (run/'queue.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        while True:
            status = read(out/'status.json') if (out/'status.json').exists() else {}
            if status.get('status')=='failed':
                raise RuntimeError('Corrected preprocessing failed: '+str(status.get('error')))
            if status.get('status')=='complete':
                break
            atomic_json(run/'queue_status.json',dict(status='waiting_preprocessing',stage=status.get('stage'),updated_unix=time.time()))
            time.sleep(30)
        atomic_json(run/'queue_status.json',dict(status='training',fresh=True,updated_unix=time.time()))
        # Never resume a legacy checkpoint. Recovery may resume only this exact
        # corrected lineage, validated again by the trainer's full contract.
        cmd=[sys.executable,'-u','-m','acoustic_branch.train','--root',str(a.root),
             '--config',str(out/'config.json'),'--run-dir',str(run)]
        if (run/'last.pt').exists():
            cmd+=['--resume',str(run/'last.pt')]
        subprocess.run(cmd,check=True)
        atomic_json(run/'queue_status.json',dict(status='complete',test_accessed=False,updated_unix=time.time()))
    except Exception:
        atomic_json(run/'queue_status.json',dict(status='failed',error=traceback.format_exc(),updated_unix=time.time()))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['prepare','child','pipeline','train_queue'])
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--helpers',type=Path)
    p.add_argument('--renderer',type=Path)
    p.add_argument('--task',type=int)
    a=p.parse_args()
    if a.mode=='prepare': prepare(a.root)
    else: globals()[a.mode](a)

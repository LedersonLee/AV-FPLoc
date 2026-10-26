#!/usr/bin/env python3
"""Immutable disjoint CPU shards + full final RIR/pair verification.

10/48 task residues belong to local, 38/48 to node2. A per-shard supervisor
lock and per-task NFS locks add protection beyond deterministic ownership.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import json
import os
from pathlib import Path
import time
import threading

import numpy as np
from .hssd_positive_pairs import atomic_json, immutable_json, sha, recipe, RING, PROTOCOL
from .hssd_rir_supervisor import IntegrityError, record_paths, verify_paths, supervise_tasks

VERSION='hard_negatives_epoch020_v1'
IMAGE='sha256:71d6114777a23b5eac0d9bb29d755bc81516cda270515009128c04f7d9951d75'


def owner(task_id):return 'local' if task_id%48<10 else 'node2'


def manifest_root(root):return Path(root)/'RIR/manifests'/VERSION


def shared_stage_lock(path):
    # NFS implements flock with POSIX byte locks: SH needs a readable FD.
    lock=Path(path).open('a+')
    fcntl.flock(lock,fcntl.LOCK_SH|fcntl.LOCK_NB)
    return lock


def validate_plan(plan,index):
    ids=[t['task_id'] for t in plan['tasks']]
    if ids!=list(range(len(ids))) or len(ids)!=plan['total_tasks']:raise IntegrityError('Invalid task coverage')
    if any(t['shard']!=owner(t['task_id']) for t in plan['tasks']):raise IntegrityError('Invalid task ownership')
    if sum(t['records'] for t in plan['tasks'])!=index['new_fp_records']:raise IntegrityError('Physical record coverage mismatch')
    if plan['split_sha256']!=index['split_sha256']:raise IntegrityError('Split identity changed')
    for shard in ('local','node2'):
        selected=[t for t in plan['tasks'] if t['shard']==shard]
        if plan['shards'][shard]['tasks']!=len(selected) or plan['shards'][shard]['records']!=sum(t['records'] for t in selected):
            raise IntegrityError('Shard summary mismatch')


def prepare(root):
    from .hssd_negative_rirs import check_ready
    root=Path(root);index=check_ready(root);dest=manifest_root(root);tasks=[];paths=set()
    for entry in index['scenes']:
        path=root/entry['schedule_path']
        if sha(path)!=entry['schedule_sha256']:raise IntegrityError('Schedule hash mismatch')
        schedule=json.loads(path.read_text())
        if schedule['split']!='train' or schedule['record_directory']!='negative_records' or schedule['pose_prefix']!='n':
            raise IntegrityError('Unsafe schedule namespace/split')
        if sha(root/schedule['pair_path'])!=schedule['pair_sha256']:raise IntegrityError('Pair hash mismatch')
        positive=root/'RIR/manifests/positive_pairs_v1/schedules/train'/f'{entry["scene_id"]}.json'
        if sha(positive)!=schedule['positive_schedule_sha256']:raise IntegrityError('Positive schedule changed')
        positions=schedule['positions']
        for offset in range(0,len(positions),12):
            selected=positions[offset:offset+12];count=0
            for p in selected:
                for base in p['base_headings']:
                    identity=(entry['scene_id'],p['pose_id'],base)
                    if identity in paths:raise IntegrityError('Duplicate output in task plan')
                    paths.add(identity);count+=1
            task_id=len(tasks)
            tasks.append(dict(task_id=task_id,scene_id=entry['scene_id'],schedule_path=entry['schedule_path'],
                schedule_sha256=entry['schedule_sha256'],pose_ids=[p['pose_id'] for p in selected],records=count,shard=owner(task_id)))
    plan=dict(version='static_48_v1',index_sha256=sha(dest/'index.json'),split_sha256=index['split_sha256'],
        image_id=IMAGE,total_tasks=len(tasks),total_records=len(paths),tasks=tasks,
        ownership='global task_id % 48 < 10 -> local; otherwise node2',
        shards={s:dict(tasks=sum(t['shard']==s for t in tasks),records=sum(t['records'] for t in tasks if t['shard']==s),
            workers=10 if s=='local' else 38,cpu_cap=20 if s=='local' else 76) for s in ('local','node2')})
    validate_plan(plan,index)
    immutable_json(dest/'distributed/plan.json',plan)
    print(json.dumps({k:plan[k] for k in ('total_tasks','total_records','shards')}),flush=True)
    return plan,index


def expanded_task(root,renderer,item):
    return (str(root),str(Path(root)/'source/hssd-hab'),str(renderer),
        dict(scene_id=item['scene_id'],split='train',schedule_path=item['schedule_path']), 'FP',item['pose_ids'],2)


def aggregate(root,plan,state='running',**extra):
    dest=manifest_root(root);stages={}
    for shard in ('node2','local'):
        path=dest/'runs/negative_shards'/shard/'status.json'
        stages[shard]=json.loads(path.read_text()) if path.exists() else dict(status='waiting',completed_tasks=0,records_written=0,records_verified_existing=0)
    value=dict(status=state,distributed=True,total_records=plan['total_records'],tasks=plan['total_tasks'],
        completed_tasks=sum(v.get('completed_tasks',0) for v in stages.values()),
        failed_tasks=sum(v.get('failed_tasks',0) for v in stages.values()),
        records_written=sum(v.get('records_written',0) for v in stages.values()),
        records_verified_existing=sum(v.get('records_verified_existing',0) for v in stages.values()),
        shards=stages,updated_unix=time.time(),last_progress_unix=max(v.get('last_progress_unix',0) for v in stages.values()) or time.time(),**extra)
    atomic_json(dest/'runs/node2_negative/status.json',value)
    return stages


def verify_final(root,renderer,plan,index):
    """Read-only full SHA/pose/ray-contract validation, including positive links."""
    root=Path(root);dest=manifest_root(root);report_path=dest/'distributed/verification.json'
    if sha(dest/'index.json')!=plan['index_sha256']:raise IntegrityError('Final index changed since partitioning')
    if report_path.exists():
        saved=json.loads(report_path.read_text())
        if saved.get('status')=='complete' and saved.get('plan_sha256')==sha(dest/'distributed/plan.json') and saved.get('index_sha256')==sha(dest/'index.json'):
            return saved
    negative_tasks=[expanded_task(root,renderer,t) for t in plan['tasks']]
    positive_index=json.loads((root/'RIR/manifests/positive_pairs_v1/index.json').read_text())
    positive_tasks=[];expected_new=set();expected_positive=set()
    schedule_cache={}
    for item,task in zip(plan['tasks'],negative_tasks):
        if item['schedule_path'] not in schedule_cache:schedule_cache[item['schedule_path']]=json.loads((root/item['schedule_path']).read_text())
        schedule=schedule_cache[item['schedule_path']]
        for p in schedule['positions']:
            if p['pose_id'] not in item['pose_ids']:continue
            for base in p['base_headings']:
                path=f'RIR/FP/train/{item["scene_id"]}/negative_records/n{p["pose_id"]:05d}/base_acoustic_yaw_{base:03d}.npy'
                if path in expected_new:raise IntegrityError('Duplicate negative output')
                expected_new.add(path)
    for entry in positive_index['scenes']:
        if entry['split']!='train':continue
        schedule=json.loads((root/entry['schedule_path']).read_text());ids=[p['pose_id'] for p in schedule['positions']]
        for role in ('FP','Observed'):
            for offset in range(0,len(ids),12):
                positive_tasks.append((str(root),str(root/'source/hssd-hab'),str(renderer),entry,role,ids[offset:offset+12],2))
            for p in schedule['positions']:
                for base in p['base_headings']:
                    expected_positive.add(f'RIR/{role}/train/{entry["scene_id"]}/records/p{p["pose_id"]:05d}/base_acoustic_yaw_{base:03d}.npy')
    if len(expected_new)!=index['new_fp_records']:raise IntegrityError('Negative unique count mismatch')
    all_tasks=negative_tasks+positive_tasks;checked=0;start=time.monotonic()
    def check(task):
        paths=record_paths(task);missing=verify_paths(paths,task[4])
        if missing:raise IntegrityError(f'Missing committed RIRs: {missing[:3]}')
        for path,*_ in paths:
            meta=json.loads(path.with_suffix('.json').read_text())
            if meta.get('acoustics')!=PROTOCOL or meta.get('source_xyz_m')!=meta['centre_xyz_m']:
                raise IntegrityError('Acoustics/source mismatch')
        return len(paths)
    with ThreadPoolExecutor(max_workers=24) as pool:
        futures=[pool.submit(check,t) for t in all_tasks]
        for number,future in enumerate(as_completed(futures),1):
            checked+=future.result()
            if number%25==0:
                atomic_json(dest/'distributed/verification_progress.json',dict(status='verifying',checked_records=checked,
                    expected_records=len(expected_new)+len(expected_positive),updated_unix=time.time()))
                aggregate(root,plan,'verifying',verification_checked_records=checked,last_verification_progress_unix=time.time())
    # Also reject unexpected committed negatives rather than hiding stale outputs.
    actual={str(p.relative_to(root)) for scene in index['scenes'] for p in (root/'RIR/FP/train'/scene['scene_id']/'negative_records').glob('n*/base_acoustic_yaw_*.npy')}
    if actual!=expected_new:raise IntegrityError(f'Unexpected/missing negative files: extra={len(actual-expected_new)}, missing={len(expected_new-actual)}')
    pairs=0;queries=set();pair_keys=set();positive_cache={}
    for entry in index['scenes']:
        scene=entry['scene_id'];schedule=json.loads((root/entry['schedule_path']).read_text())
        positive=json.loads((root/'RIR/manifests/positive_pairs_v1/schedules/train'/f'{scene}.json').read_text())
        qmap={q['sample_index']:q for q in positive['queries']}
        source_pairs=json.loads((root/schedule['pair_path']).read_text())['pairs']
        for p in source_pairs:
            q=qmap[p['sample_index']]
            if p['scene_id']!=scene or p['split']!='train' or p['observed_rir_path']!=q['observed_rir_path'] or p['positive_fp_rir_path']!=q['fp_rir_path']:
                raise IntegrityError('Cross-scene/split or positive pair mismatch')
            if p['gt_centre_xyz_m']!=q['centre_xyz_m'] or p['gt_world_yaw_deg']!=q['world_yaw_deg']:
                raise IntegrityError('GT pair mismatch')
            path=p['negative_fp_rir_path']
            if not path.startswith(f'RIR/FP/train/{scene}/') or (path not in expected_new and path not in expected_positive):raise IntegrityError('Unscheduled negative reference')
            if path not in positive_cache:positive_cache[path]=json.loads((root/path).with_suffix('.json').read_text())
            meta=positive_cache[path];base,roll=recipe(p['negative_world_yaw_deg'])
            if meta['centre_xyz_m']!=p['negative_centre_xyz_m'] or meta['base_acoustic_yaw_deg']!=base or p['negative_channel_roll']!=roll:
                raise IntegrityError('Negative pose/channel-roll mismatch')
            if p['observed_channel_roll']!=q['channel_roll'] or p['positive_channel_roll']!=q['channel_roll']:
                raise IntegrityError('Positive channel-roll mismatch')
            key=(scene,p['sample_index'],tuple(p['negative_centre_xyz_m']),p['negative_world_yaw_deg'])
            if key in pair_keys:raise IntegrityError('Duplicate query/negative pair')
            pair_keys.add(key)
            queries.add((scene,p['sample_index']));pairs+=1
            if pairs%10000==0:aggregate(root,plan,'verifying',verification_checked_records=checked,verified_pair_links=pairs,last_verification_progress_unix=time.time())
        positive_cache.clear()
    if pairs!=index['pairs'] or len(queries)!=index['queries']:raise IntegrityError('Pair/query coverage mismatch')
    report=dict(status='complete',plan_sha256=sha(dest/'distributed/plan.json'),index_sha256=sha(dest/'index.json'),
        split_sha256=index['split_sha256'],checkpoint_sha256=index['checkpoint_sha256'],train_scenes=len(index['scenes']),
        verified_negative_arrays=len(expected_new),verified_positive_arrays=len(expected_positive),
        verified_arrays=checked,verified_pair_links=pairs,queries=len(queries),duplicates=0,missing=0,
        checksum_failures=0,elapsed_seconds=time.monotonic()-start,completed_unix=time.time(),
        checks=['full SHA256','finite nonzero six-channel arrays','exact pose/base/source','acoustic protocol',
            'disjoint task/output ownership','complete query/pair coverage','train-only split','channel rolls'])
    equivalence=dest/'distributed/renderer_equivalence.json'
    if equivalence.exists():report['renderer_equivalence_sha256']=sha(equivalence)
    atomic_json(report_path,report);return report


def render(args):
    for key in ('CUDA_VISIBLE_DEVICES','NVIDIA_VISIBLE_DEVICES'):
        if os.environ.get(key) not in ('-1','void','none',''):raise IntegrityError('GPU must be disabled')
    root=args.root;dest=manifest_root(root)
    # Negative coordinators share this stage lock; positives still require EX.
    stage=shared_stage_lock(root/'RIR/manifests/positive_pairs_v1/render.lock')
    plan,index=prepare(root)
    if args.workers>plan['shards'][args.shard]['workers']:raise IntegrityError('Worker cap exceeded')
    lock=(dest/'distributed'/f'{args.shard}.supervisor.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    selected=[t for t in plan['tasks'] if t['shard']==args.shard]
    tasks=[expanded_task(root,args.renderer_dir,t) for t in selected]
    shard_out=dest/'runs/negative_shards'/args.shard
    stopped=threading.Event()
    def heartbeat():
        while not stopped.wait(10):aggregate(root,plan,'running')
    thread=threading.Thread(target=heartbeat,daemon=True) if args.shard=='node2' else None
    if thread:thread.start()
    try:
        supervise_tasks(tasks,shard_out,args.workers,expected_records=plan['shards'][args.shard]['records'],
            task_ids=[t['task_id'] for t in selected],task_lock_dir=dest/'distributed/task_locks')
        stopped.set()
        if thread:thread.join()
        if args.shard=='local':return
        while True:
            stages=aggregate(root,plan,'waiting_local')
            if all(v['status']=='complete' for v in stages.values()):break
            time.sleep(10)
        aggregate(root,plan,'verifying')
        report=verify_final(root,args.renderer_dir,plan,index)
        aggregate(root,plan,'complete',verification=report)
    except Exception as exc:
        stopped.set()
        if thread:thread.join()
        if args.shard=='node2':aggregate(root,plan,'failed',error=str(exc))
        atomic_json(shard_out/'fatal_error.json',dict(error=str(exc),updated_unix=time.time()))
        raise

#!/usr/bin/env python3
"""CPU-only bounded subprocess RIR scheduler, integrity-first resumable recovery.

Each native simulator runs in a fresh process/session (no reused fork pool).
Completed arrays are SHA-validated. A stuck native shutdown cannot lose valid
records: after validation, only that child process group is terminated.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import fcntl
import json
import os
import queue
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

import numpy as np
from .hssd_positive_pairs import atomic_json, sha, render_task

SUPPLEMENT={'104348028_171512877','105515235_173104215','106878867_174886977',
            '103997562_171030642','104348037_171512898'}


class IntegrityError(RuntimeError):
    pass


def record_paths(task):
    root,source,renderer,entry,role,ids,threads=task
    root=Path(root);schedule_path=root/entry['schedule_path']
    schedule=json.loads(schedule_path.read_text());fingerprint=sha(schedule_path)
    if sha(root/'manifests/scene_splits_v1.json')!=schedule['split_sha256']:
        raise IntegrityError('Split hash mismatch')
    geometry=root/schedule['geometry_path'] if role=='FP' else Path(source)/schedule['scene_path']
    expected=schedule['geometry_sha256'] if role=='FP' else schedule['scene_sha256']
    if sha(geometry)!=expected:raise IntegrityError('Geometry hash mismatch')
    paths=[]
    for p in schedule['positions']:
        if p['pose_id'] not in ids:continue
        for base in p['base_headings']:
            directory=schedule.get('record_directory','records');prefix=schedule.get('pose_prefix','p')
            if directory not in ('records','negative_records') or prefix not in ('p','n'):
                raise IntegrityError('Unsafe output namespace')
            path=root/'RIR'/role/schedule['split']/schedule['scene_id']/directory/f'{prefix}{p["pose_id"]:05d}'/f'base_acoustic_yaw_{base:03d}.npy'
            paths.append((path,fingerprint,expected,p['centre_xyz_m'],base))
    return paths


def verify_paths(paths,role):
    missing=[]
    for path,fingerprint,geometry,xyz,base in paths:
        metadata=path.with_suffix('.json')
        if not path.exists() or not metadata.exists():missing.append(str(path));continue
        saved=json.loads(metadata.read_text())
        if saved.get('schedule_sha256')!=fingerprint or saved.get('geometry_sha256')!=geometry or saved.get('role')!=role:
            raise IntegrityError(f'Record contract mismatch: {path}')
        if saved.get('centre_xyz_m')!=xyz or saved.get('base_acoustic_yaw_deg')!=base:
            raise IntegrityError(f'Record pose mismatch: {path}')
        if saved.get('rir_sha256')!=sha(path):raise IntegrityError(f'RIR checksum mismatch: {path}')
        array=np.load(path,mmap_mode='r')
        if array.ndim!=2 or array.shape[0]!=6 or array.shape[1]<1 or not np.isfinite(array).all() or not np.all(np.any(array!=0,axis=1)):
            raise IntegrityError(f'Invalid RIR array: {path}')
        del array
    return missing


def stop_child(child):
    if child.poll() is not None:return
    try:os.killpg(child.pid,signal.SIGTERM)
    except ProcessLookupError:
        child.wait(timeout=10);return
    try:child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)


def isolated_task(task_id,task,out,stall_seconds=600,hard_seconds=1800,cpus=None,progress=None):
    out=Path(out);paths=record_paths(task);role=task[4];initial=verify_paths(paths,role)
    split=json.loads((Path(task[0])/task[3]['schedule_path']).read_text())['split']
    def result(written,skipped,**extra):
        return dict(scene_id=task[3]['scene_id'],split=split,role=role,pose_ids=task[5],
            written=written,skipped=skipped,verified_records=len(paths),**extra)
    if not initial:return result(0,len(paths),cache_verified=True)
    payload=out/'inputs'/f'task_{task_id:06d}.json';atomic_json(payload,list(task))
    for attempt in range(1,4):
        logfile=out/'logs'/f'task_{task_id:06d}_attempt_{attempt}.log';logfile.parent.mkdir(parents=True,exist_ok=True)
        answer=out/'results'/f'task_{task_id:06d}_attempt_{attempt}.json'
        start=time.monotonic();last_progress=start;last_mtime=None
        with logfile.open('a') as log:
            command=[sys.executable,'-u','-m','avfploc.preprocessing.hssd.hssd_rir_supervisor','worker','--task',str(payload),'--result',str(answer)]
            if cpus is not None:command=['taskset','-c',','.join(map(str,cpus))]+command
            child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,
                stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
            try:
                while child.poll() is None:
                    time.sleep(5)
                    now=time.monotonic();mtime=logfile.stat().st_mtime
                    if mtime!=last_mtime:
                        last_mtime=mtime;last_progress=now
                        if progress:progress()
                    # A final native sim.close()/executor shutdown may deadlock.
                    # Validate every array before treating this as recoverable.
                    if now-last_progress>=30 and all(p.exists() and p.with_suffix('.json').exists() for p,*_ in paths):
                        if not verify_paths(paths,role):
                            stop_child(child)
                            atomic_json(out/'recovery'/f'task_{task_id:06d}_attempt_{attempt}.json',
                                dict(reason='all_records_valid_native_shutdown_stalled',pid=child.pid,verified_records=len(paths),updated_unix=time.time()))
                            return result(len(initial),len(paths)-len(initial),shutdown_recovered=True,attempt=attempt)
                    if now-last_progress>stall_seconds or now-start>hard_seconds:
                        stop_child(child)
                        atomic_json(out/'recovery'/f'task_{task_id:06d}_attempt_{attempt}.json',
                            dict(reason='isolated_worker_timeout',pid=child.pid,updated_unix=time.time()))
                        break
            finally:
                if child.poll() is None:stop_child(child)
        missing=verify_paths(paths,role)
        if not missing:
            return result(len(initial),len(paths)-len(initial),attempt=attempt,worker_exit=child.returncode)
        if answer.exists():
            saved=json.loads(answer.read_text())
            error=saved.get('error','')
            if any(token in error for token in ('checksum','hash mismatch','Geometry changed','Split hash','Input/output error','No space left')):
                raise IntegrityError(error)
    raise RuntimeError(f'Task {task_id} exhausted 3 isolated attempts; {len(missing)} records missing')


def worker_cpu_slots(available,workers,oversubscribe=False):
    if len(available)<2 or (len(available)<workers*2 and not oversubscribe):
        raise ValueError('Need two CPU slots per acoustic worker (or explicit bounded oversubscription)')
    return [[available[(2*i)%len(available)],available[(2*i+1)%len(available)]] for i in range(workers)]


def supervise_tasks(tasks,out,workers,expected_records=None,limited_smoke=False,task_ids=None,task_lock_dir=None,oversubscribe=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    results=[];errors=[];start=time.monotonic();activity=[time.time()]
    def progress():activity[0]=time.time()
    def status(state):
        atomic_json(out/'status.json',dict(status=state,tasks=len(tasks),completed_tasks=len(results),failed_tasks=len(errors),
            records_written=sum(r['written'] for r in results),records_verified_existing=sum(r['skipped'] for r in results),
            elapsed_seconds=time.monotonic()-start,workers=workers,acoustic_threads=2,cpu_only=True,
            scheduler='isolated_subprocess_v2',updated_unix=time.time(),last_progress_unix=activity[0],
            total_records=expected_records,limited_smoke=limited_smoke,oversubscribe=oversubscribe))
    status('running')
    slots=queue.Queue();available=sorted(os.sched_getaffinity(0))
    for pair in worker_cpu_slots(available,workers,oversubscribe):slots.put(pair)
    if task_ids is None:task_ids=list(range(len(tasks)))
    if len(task_ids)!=len(tasks) or len(set(task_ids))!=len(task_ids):raise IntegrityError('Invalid global task IDs')
    if task_lock_dir is not None:Path(task_lock_dir).mkdir(parents=True,exist_ok=True)
    def run_slot(i,task):
        cpus=slots.get()
        lock=None
        try:
            if task_lock_dir is not None:
                lock=(Path(task_lock_dir)/f'task_{i:06d}.lock').open('a')
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:raise IntegrityError(f'Task {i} already owned; refusing duplicate generation')
            return isolated_task(i,task,out,cpus=cpus,progress=progress)
        finally:
            if lock is not None:lock.close()
            slots.put(cpus)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        items=list(zip(task_ids,tasks))
        order=sorted(items,key=lambda item:
            (out/'verified'/f'task_{item[0]:06d}.json').exists() or (out/f'task_{item[0]:05d}.json').exists())
        pending={pool.submit(run_slot,i,task):i for i,task in order}
        while pending:
            done,_=wait(pending,timeout=10,return_when=FIRST_COMPLETED)
            for future in done:
                i=pending.pop(future)
                try:
                    value=future.result();results.append(value);atomic_json(out/'verified'/f'task_{i:06d}.json',value);progress()
                except Exception:
                    error=dict(task=i,error=traceback.format_exc());errors.append(error);atomic_json(out/'failures'/f'task_{i:06d}.json',error)
                    print(json.dumps(error),flush=True)
            status('running')
    if not errors and expected_records is not None and sum(r['written']+r['skipped'] for r in results)!=expected_records:
        errors.append(dict(error='Verified record count differs from the immutable schedule'))
    status('failed' if errors else 'complete')
    if errors:raise RuntimeError(f'{len(errors)} tasks failed; integrity failures require inspection')
    return results


def positive_render(args):
    lock=(args.root/'RIR/manifests/positive_pairs_v1/render.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX if args.wait_lock else fcntl.LOCK_EX|fcntl.LOCK_NB)
    for key in ('CUDA_VISIBLE_DEVICES','NVIDIA_VISIBLE_DEVICES'):
        if os.environ.get(key) not in ('-1','void','none',''):raise ValueError('GPU visibility must be disabled')
    index=json.loads((args.root/'RIR/manifests/positive_pairs_v1/index.json').read_text())
    entries=[e for e in index['scenes'] if e['split'] in args.splits and (not args.scene or e['scene_id']==args.scene)
        and (not args.only_scenes or e['scene_id'] in args.only_scenes)]
    # Main was launched before five mean-height schedules were appended to index.
    # Preserve its original 111-scene scope so completion counts never double-count.
    if args.run_name=='node2_train' and not args.only_scenes:entries=[e for e in entries if e['scene_id'] not in SUPPLEMENT]
    tasks=[]
    for e in entries:
        schedule=json.loads((args.root/e['schedule_path']).read_text());ids=[p['pose_id'] for p in schedule['positions']]
        if args.limit:ids=ids[:args.limit]
        for role in ('FP','Observed'):
            for start in range(0,len(ids),args.chunk):
                tasks.append((str(args.root),str(args.source),str(args.renderer_dir),e,role,ids[start:start+args.chunk],args.acoustic_threads))
    expected=None if args.limit else 2*sum(e['physical_records_per_role'] for e in entries)
    supervise_tasks(tasks,args.root/'RIR/manifests/positive_pairs_v1/runs'/args.run_name,args.workers,
        expected_records=expected,limited_smoke=bool(args.limit))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['worker']);parser.add_argument('--task',type=Path,required=True);parser.add_argument('--result',type=Path,required=True)
    args=parser.parse_args()
    try:
        result=render_task(tuple(json.loads(args.task.read_text())));atomic_json(args.result,dict(status='complete',result=result))
    except Exception:
        atomic_json(args.result,dict(status='failed',error=traceback.format_exc()));raise

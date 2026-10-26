"""Disjoint continuation of the frozen full-train RIR job on three hosts.

Only launch receipts owned by this workflow may be stopped. Global task IDs,
shared advisory locks and immutable assignments prevent duplicate generation.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time

from .contracts import atomic_json, immutable_json, sha
from .data import read
from .full_train_rirs import REL, schedule_tasks, validate_loaded_frame
from .render_adapter import supervisor_tools

ROUND = 'distributed_v1'
HOSTS = dict(node1='n1.unist.info', node2='n2.unist.info')
RESOURCES = dict(node1=dict(cpus=84,workers=48,memory=96), node2=dict(cpus=78,workers=48,memory=80),
                 local=dict(cpus=12,workers=6,memory=16))
ROOT = Path('/file2/jeongeon/AV-FPLoc/datasets/HSSD')


def execute(node, command):
    if node == 'local':
        return subprocess.check_output(command,text=True)
    return subprocess.check_output(['ssh','-i','/home/jeongeonlee/.ssh/id_ed25519_lgsdf','-p','55555',
        '-o','BatchMode=yes','-o','ConnectTimeout=15','jeongeon@'+HOSTS[node],shlex.join(command)],text=True,timeout=180)


def counts(root, tasks):
    schedules = {}
    result = []
    for task in tasks:
        name = task[3]['schedule_path']
        if name not in schedules:
            schedules[name] = {p['pose_id']:len(p['base_headings']) for p in read(root/name)['positions']}
        result.append(sum(schedules[name][i] for i in task[5]))
    return result


def check_receipt(task, count, receipt):
    if (receipt.get('scene_id') != task[3]['scene_id'] or receipt.get('split') != 'train'
        or receipt.get('role') != 'FP' or receipt.get('pose_ids') != task[5]
        or receipt.get('verified_records') != count or receipt['written']+receipt['skipped'] != count):
        raise ValueError('Task receipt no longer matches frozen global task')


def partition(task_counts, completed, resources=RESOURCES):
    if any(i < 0 or i >= len(task_counts) for i in completed):
        raise ValueError('Completion ID outside immutable task list')
    shards = {node:[] for node in resources}
    assigned = {node:0 for node in resources}
    for i in sorted(set(range(len(task_counts)))-set(completed),key=lambda i:(-task_counts[i],i)):
        node = min(resources,key=lambda n:(assigned[n]/resources[n]['cpus'],n))
        shards[node].append(i);assigned[node] += task_counts[i]
    for ids in shards.values():
        ids.sort()
    validate_partition(len(task_counts),completed,shards)
    return shards, assigned


def validate_partition(total, completed, shards):
    ids = list(completed)+[i for values in shards.values() for i in values]
    if len(ids) != len(set(ids)) or set(ids) != set(range(total)):
        raise ValueError('Distributed assignment must be disjoint and exhaustive')


def prepare_launch():
    out = ROOT/REL;folder = out/ROUND
    folder.mkdir(exist_ok=True)
    controller = (folder/'launch.lock').open('a+')
    fcntl.flock(controller,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # Existing local renderer only, identified by its immutable launch receipt.
    receipt = read(out/'launch_local.json')
    state = json.loads(execute('local',['docker','inspect',receipt['name']]))[0]
    if state['Id'] != receipt['container_id']:
        raise ValueError('Refusing to stop an unowned local container')
    if state['State']['Running']:
        execute('local',['docker','update','--restart=no',receipt['name']])
        execute('local',['docker','stop','--time','30',receipt['name']])
    lock = (out/'run.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan = read(out/'plan.json')
    tasks,total = schedule_tasks(ROOT,ROOT/'RIR/tools/hard_negatives_epoch020_v1',plan['entries'],plan['chunk_poses'])
    task_counts = counts(ROOT,tasks)
    if total != plan['fp_arrays'] or sum(task_counts) != total:
        raise ValueError('Physical counts changed')
    assignment_path = folder/'assignment.json'
    if assignment_path.exists():
        assignment = read(assignment_path)
    else:
        completed, proofs = [], {}
        for path in (out/'render/verified').glob('task_*.json'):
            i = int(path.stem.split('_')[1])
            check_receipt(tasks[i],task_counts[i],read(path))
            completed.append(i);proofs[str(i)] = sha(path)
        shards, arrays = partition(task_counts,completed)
        assignment = dict(version=ROUND,plan_sha256=sha(out/'plan.json'),task_count=len(tasks),
            completed_task_ids=sorted(completed),completed_receipt_sha256=proofs,
            completed_arrays=sum(task_counts[i] for i in completed),shards=shards,arrays=arrays,
            resources=RESOURCES,created_unix=time.time(),
            policy='Re-use committed SHA/frame-verified tasks; resume partial tasks; never resplit scenes or mine new negatives')
        immutable_json(assignment_path,assignment)
    validate_partition(len(tasks),assignment['completed_task_ids'],assignment['shards'])
    if assignment['plan_sha256'] != sha(out/'plan.json'):
        raise ValueError('Plan changed after partition')
    lock.close()
    source = Path(__file__).resolve().parents[1]
    files = list((source/'acoustic_branch').glob('*.py'))
    files += [source/'experiments/exp4_acoustic_xy_observability/scripts'/n for n in ('hssd_positive_pairs.py','hssd_rir_supervisor.py')]
    hashes = {str(p.relative_to(source)):sha(p) for p in files}
    digest = hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:16]
    snapshot = ROOT/'RIR/tools/distributed_train'/digest
    for p in files:
        dest = snapshot/p.relative_to(source);dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():
            if sha(dest) != sha(p):raise ValueError('Immutable source snapshot changed')
        else:shutil.copy2(p,dest)
    immutable_json(snapshot/'snapshot.json',hashes)
    def launch(node):
        name = 'hssd-full-train-rir-'+node+'-distributed-v1'
        saved = folder/'launches'/(node+'.json')
        if saved.exists():
            old = read(saved)
            current = json.loads(execute(node,['docker','inspect',name]))[0]
            if current['Id'] != old['container_id']:raise ValueError('Container ownership changed')
            return dict(node=node,status=current['State']['Status'])
        resource = RESOURCES[node]
        image = read(out/'renderer_image_identity.json')['loaded_image_id'] if node=='local' else (
            'sha256:71d6114777a23b5eac0d9bb29d755bc81516cda270515009128c04f7d9951d75')
        owner = ROOT.stat()
        command = ['docker','run','-d','--init','--name',name,'--runtime=runc','--entrypoint','/usr/bin/nice',
            '--cpus',str(resource['cpus']),'--cpu-shares','256','--memory',str(resource['memory'])+'g',
            '--pids-limit','4096','--restart','on-failure:2','--user',f'{owner.st_uid}:{owner.st_gid}']
        env = dict(NVIDIA_VISIBLE_DEVICES='void',CUDA_VISIBLE_DEVICES='-1',EGL_PLATFORM='surfaceless',
            LIBGL_ALWAYS_SOFTWARE='1',LD_PRELOAD='/usr/lib/x86_64-linux-gnu/libstdc++.so.6',
            OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',LP_NUM_THREADS='1',
            MESA_GLTHREAD='false',HABITAT_SIM_LOG='quiet',MAGNUM_LOG='quiet',
            PYTHONDONTWRITEBYTECODE='1',MESA_SHADER_CACHE_DISABLE='true')
        for key,value in env.items():command += ['-e',key+'='+value]
        command += ['-v',f'{ROOT}:/datasets/HSSD:ro','-v',f'{out}:/datasets/HSSD/{REL}:rw',
            '-v',f'{ROOT}/RIR/FP/train:/datasets/HSSD/RIR/FP/train:rw',
            '-v',f'{snapshot}:/code:ro','-w','/code',image,'-n','19',
            '/opt/conda/envs/soundspaces/bin/python3','-u','-m','acoustic_branch.distributed_train_rirs',
            'run','--root','/datasets/HSSD','--node',node]
        cid = execute(node,command).strip()
        current = json.loads(execute(node,['docker','inspect',name]))[0]
        if current['HostConfig']['Runtime'] != 'runc' or current['HostConfig'].get('DeviceRequests'):
            execute(node,['docker','stop',name]);raise ValueError('CPU-only isolation failed')
        immutable_json(saved,dict(node=node,container_id=cid,name=name,command=command,image=image,
            assignment_sha256=sha(assignment_path),snapshot_sha256=sha(snapshot/'snapshot.json'),
            resources=resource,gpu_forbidden=True,created_unix=time.time()))
        return dict(node=node,status='launched',assigned_arrays=assignment['arrays'][node],**resource)
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(launch,RESOURCES))
    atomic_json(folder/'launch_complete.json',dict(status='complete',results=results,assignment_sha256=sha(assignment_path)))
    print(json.dumps(results),flush=True)


def finalize(root, assignment, tasks, task_counts):
    out = root/REL;folder = out/ROUND
    if not all((folder/'nodes'/(node+'.json')).exists() for node in assignment['shards']):return
    for node in assignment['shards']:
        marker = read(folder/'nodes'/(node+'.json'))
        if marker['assignment_sha256'] != sha(folder/'assignment.json') or marker['status'] != 'complete':
            raise ValueError('Node completion belongs to a different assignment')
    lock = (folder/'finalize.lock').open('a+')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:return
    if (folder/'ready.json').exists():return
    proofs = {}
    for i in assignment['completed_task_ids']:
        path = out/'render/verified'/f'task_{i:06d}.json'
        if sha(path) != assignment['completed_receipt_sha256'][str(i)]:raise ValueError('Prior completion receipt changed')
        check_receipt(tasks[i],task_counts[i],read(path));proofs[str(i)] = sha(path)
    for node,ids in assignment['shards'].items():
        for i in ids:
            path = folder/'render'/node/'verified'/f'task_{i:06d}.json'
            check_receipt(tasks[i],task_counts[i],read(path));proofs[str(i)] = sha(path)
    if len(proofs) != len(tasks):raise ValueError('Incomplete global coverage')
    atomic_json(folder/'task_proofs.json',proofs)
    result = dict(status='complete',plan_sha256=assignment['plan_sha256'],
        assignment_sha256=sha(folder/'assignment.json'),tasks=len(tasks),fp_arrays=sum(task_counts),
        task_proofs_sha256=sha(folder/'task_proofs.json'),test_accessed=False,
        stage='raw_FP_train_verified',feature_cache_built=False,training_started=False,updated_unix=time.time())
    atomic_json(folder/'ready.json',result);atomic_json(out/'render_complete.json',result);atomic_json(out/'status.json',result)


def run(root,node,workers=None):
    root = Path(root);out = root/REL;folder = out/ROUND
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '-1' or os.environ.get('NVIDIA_VISIBLE_DEVICES') != 'void':
        raise ValueError('CPU only')
    assignment = read(folder/'assignment.json');plan = read(out/'plan.json')
    if sha(out/'plan.json') != assignment['plan_sha256']:raise ValueError('Plan changed')
    lock = (folder/(node+'.lock')).open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    run_lease = (out/'run.lock').open('a+')
    fcntl.flock(run_lease,fcntl.LOCK_SH|fcntl.LOCK_NB)
    for relative,digest in plan['sources'].items():
        if sha(root/relative) != digest:raise ValueError('Pinned train source changed')
    tasks,total = schedule_tasks(root,root/'RIR/tools/hard_negatives_epoch020_v1',plan['entries'],plan['chunk_poses'])
    validate_partition(len(tasks),assignment['completed_task_ids'],assignment['shards'])
    task_counts = counts(root,tasks);ids = assignment['shards'][node]
    if sum(task_counts[i] for i in ids) != assignment['arrays'][node] or total != plan['fp_arrays']:
        raise ValueError('Assignment physical count differs')
    supervisor = supervisor_tools();original = supervisor.verify_paths
    @lru_cache(maxsize=128)
    def bounds(scene):return read(root/f'RIR/Geometry/{scene}/floorplan.json')['mesh']['bounds_xyz_m']
    def strict(paths,role):
        missing = original(paths,role)
        for path,*_ in paths:
            if str(path) not in missing:
                validate_loaded_frame(read(path.with_suffix('.json')),bounds(path.relative_to(root).parts[3]))
        return missing
    supervisor.verify_paths = strict
    workers = workers or assignment['resources'][node]['workers']
    if workers<1 or workers>96:raise ValueError('Bounded worker range is 1..96')
    supervisor.supervise_tasks([tasks[i] for i in ids],folder/'render'/node,workers,
        expected_records=assignment['arrays'][node],task_ids=ids,task_lock_dir=out/'task_locks',
        oversubscribe=workers*2>len(os.sched_getaffinity(0)))
    atomic_json(folder/'nodes'/(node+'.json'),dict(status='complete',node=node,
        assignment_sha256=sha(folder/'assignment.json'),fp_arrays=assignment['arrays'][node],updated_unix=time.time()))
    finalize(root,assignment,tasks,task_counts)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['launch','run'])
    p.add_argument('--root',type=Path);p.add_argument('--node',choices=list(RESOURCES));p.add_argument('--workers',type=int);a=p.parse_args()
    if a.action=='launch':prepare_launch()
    else:run(a.root,a.node,a.workers)

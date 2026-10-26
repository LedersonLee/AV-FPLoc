"""Two-host disjoint joint caches with fused raw verification and extraction.

Keeps the frozen config, data paths, cache schema, GPU queue and training code.
Only one payload read supplies SHA validation and all three feature streams.
Each process owns complete immutable shards; threads overlap small-file reads.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import fcntl
import multiprocessing
import os
from pathlib import Path
import time
import traceback
import uuid

import numpy as np

from .contracts import atomic_json, immutable_json, sha
from .data import read, load_manifest, unique_refs, cache_directory, checked_rir, preflight
from .features import extract
from .envelope_cache import DEFINITION, extract as extract_envelope, check as envelope_check
from .full_scene_workflow import REL as FULL, train_ready, update

REL = FULL+'/distributed_cache_v1'
_STATE = {}


def raw_receipt(value):
    return {k:value[k] for k in ('raw_sha256','metadata_sha256','raw_samples')}


def task_layout(refs, shard_size=256):
    groups=defaultdict(list)
    for path in sorted(refs):groups['/'.join(Path(path).parts[1:4])].append(path)
    return [(group,offset//shard_size,paths[offset:offset+shard_size])
            for group,paths in sorted(groups.items()) for offset in range(0,len(paths),shard_size)]


def partition(tasks, completed):
    if any(i<0 or i>=len(tasks) for i in completed):raise ValueError('Completed ID out of bounds')
    result=dict(node1=[],node2=[]);counts=dict(node1=0,node2=0)
    for i in sorted(set(range(len(tasks)))-set(completed),key=lambda i:(-tasks[i]['arrays'],i)):
        node=min(counts,key=lambda n:(counts[n],n));result[node].append(i);counts[node]+=tasks[i]['arrays']
    ids=list(completed)+[i for rows in result.values() for i in rows]
    if len(ids)!=len(set(ids)) or set(ids)!=set(range(len(tasks))):
        raise ValueError('Assignment must be disjoint/exhaustive')
    for values in result.values():values.sort()
    return result,counts


def check_shard(directory, task, config, full=True):
    receipt=read(directory/'receipt.json')
    if (receipt.get('paths')!=task['paths'] or receipt.get('manifest_sha256')!=task['manifest_sha256']
        or receipt.get('features')!=config['features'] or receipt.get('envelope_definition')!=DEFINITION
        or len(receipt.get('records',[]))!=len(task['paths'])
        or set(receipt.get('files',{}))!={'energy.npy','coherence.npy','envelope.npy'}):
        raise ValueError('Committed shard identity changed')
    shapes=dict(energy=(6,19,95),coherence=(2,6,2,19),envelope=(6,252))
    for name,digest in receipt['files'].items():
        path=directory/name
        if full and sha(path)!=digest:raise ValueError('Committed shard checksum changed')
        array=np.load(path,mmap_mode='r',allow_pickle=False)
        if array.shape!=(len(task['paths']),*shapes[path.stem]) or array.dtype!=np.float16:
            raise ValueError('Committed feature shape/dtype changed')
        array._mmap.close()
    for path,record in zip(receipt['paths'],receipt['records']):
        expected=task.get('expected_raw',{}).get(path)
        if expected and raw_receipt(record)!=raw_receipt(expected):
            raise ValueError('Cache and existing raw verification differ')
    return receipt


def prepare(root):
    root=Path(root);folder=root/REL;folder.mkdir(parents=True,exist_ok=True)
    config=read(root/FULL/'config.json');native_plan=read(root/FULL/'plan.json')
    if not train_ready(root,native_plan) or not (root/FULL/'val_render_ready.json').exists():
        raise ValueError('Raw generation is not complete')
    if sha(root/FULL/'plan.json')!=config['native_frame']['plan_sha256']:
        raise ValueError('Frozen native frame changed')
    tasks=[];splits={}
    for split in ('train','val'):
        queries,digest=load_manifest(root,split,config);refs=unique_refs(queries)
        verification=root/FULL/'preprocessing'/f'{split}_verification.json'
        previous=read(verification) if verification.exists() else None
        if previous and (previous.get('status')!='complete' or previous.get('manifest_sha256')!=digest
            or previous.get('plan_sha256')!=config['native_frame']['plan_sha256']
            or set(previous.get('records',{}))!=set(refs)):
            raise ValueError('Prior raw verification does not match current queries')
        splits[split]=dict(manifest_sha256=digest,queries=len(queries),arrays=len(refs),
            prior_verification_sha256=sha(verification) if previous else None)
        dest=cache_directory(root,config,split)
        for group,index,paths in task_layout(refs):
            i=len(tasks)
            task=dict(task_id=i,split=split,paths=paths,refs={p:refs[p] for p in paths},
                directory=str((dest/group/f'{index:05d}').relative_to(root)),manifest_sha256=digest,
                config_sha256=sha(root/FULL/'config.json'),
                expected_raw={p:previous['records'][p] for p in paths} if previous else {})
            path=folder/'tasks'/f'{i:06d}.json';immutable_json(path,task)
            tasks.append(dict(task_id=i,path=str(path.relative_to(root)),sha256=sha(path),
                split=split,arrays=len(paths),directory=task['directory']))
        print(dict(prepared=split,arrays=len(refs),tasks=len(tasks)),flush=True)
    plan=dict(version='distributed_joint_cache_v1',tasks=tasks,splits=splits,
        config_sha256=sha(root/FULL/'config.json'),native_plan_sha256=config['native_frame']['plan_sha256'],
        total_arrays=sum(s['arrays'] for s in splits.values()),test_accessed=False,
        raw_policy='Single payload+metadata read; hash/frame/pose/protocol validated; extract all3 from same bytes')
    immutable_json(folder/'plan.json',plan)
    # Caller must stop the receipt-owned old feature job before this point.
    lease=(root/FULL/'features.lock').open('a+');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    locks=[]
    for split in ('train','val'):
        dest=cache_directory(root,config,split);dest.mkdir(parents=True,exist_ok=True)
        lock=(dest/'build.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(lock)
    completed=[];proofs={}
    for entry in tasks:
        directory=root/entry['directory']
        if (directory/'receipt.json').exists():
            task=read(root/entry['path']);check_shard(directory,task,config)
            completed.append(entry['task_id']);proofs[str(entry['task_id'])]=sha(directory/'receipt.json')
    assignments,counts=partition(tasks,completed)
    value=dict(plan_sha256=sha(folder/'plan.json'),completed_task_ids=completed,
        completed_receipt_sha256=proofs,shards=assignments,arrays=counts,
        reused_arrays=sum(tasks[i]['arrays'] for i in completed),test_accessed=False)
    immutable_json(folder/'assignment.json',value)
    update(root,'features','distributed_cache_ready',status='running',total_arrays=plan['total_arrays'],
        reused_arrays=value['reused_arrays'],assignment_arrays=counts)
    print(dict(assignment_arrays=counts,reused_arrays=value['reused_arrays']),flush=True)
    return plan,value


def init_worker(root,node,plan,assignment,threads):
    global _STATE
    root=Path(root);config=read(root/FULL/'config.json')
    _STATE=dict(root=root,node=node,config=config,plan=plan,allowed=set(assignment['shards'][node]),threads=threads)


def process_task(task_id):
    state=_STATE;root=state['root'];config=state['config']
    if task_id not in state['allowed']:raise ValueError('Worker requested another node task')
    entry=state['plan']['tasks'][task_id];path=root/entry['path']
    if sha(path)!=entry['sha256']:raise ValueError('Immutable task changed')
    task=read(path)
    if task['config_sha256']!=state['plan']['config_sha256']:raise ValueError('Task/config identity differs')
    lock_path=root/REL/'task_locks'/f'{task_id:06d}.lock';lock_path.parent.mkdir(exist_ok=True)
    lock=lock_path.open('a+');fcntl.flock(lock,fcntl.LOCK_EX)
    directory=root/task['directory'];directory.mkdir(parents=True,exist_ok=True)
    if (directory/'receipt.json').exists():
        check_shard(directory,task,config)
        return dict(task_id=task_id,arrays=len(task['paths']),reused=True)
    def one(path):
        array,receipt=checked_rir(root,path,task['refs'][path],config,single_read=True)
        expected=task['expected_raw'].get(path)
        if expected and receipt!=raw_receipt(expected):raise ValueError('Raw bytes/metadata changed after verification')
        energy,coherence=extract(array,config['features']);env=extract_envelope(array)
        receipt['zero_tail_padding']=bool(config['features']['version']=='energy_complex_zero_tail_v2'
            and array.shape[-1]<7+round(config['features']['end_ms']*48))
        values=[v.astype(np.float16) for v in (energy,coherence,env)]
        if not all(np.isfinite(v).all() for v in values):raise ValueError('Nonfinite feature')
        return values,receipt
    with ThreadPoolExecutor(max_workers=state['threads']) as pool:values=list(pool.map(one,task['paths']))
    files={}
    for index,name in enumerate(('energy.npy','coherence.npy','envelope.npy')):
        path=directory/name;pending=path.with_name(name+'.'+uuid.uuid4().hex+'.tmp')
        with pending.open('wb') as stream:np.save(stream,np.stack([v[0][index] for v in values]),allow_pickle=False)
        pending.replace(path);files[name]=sha(path)
    atomic_json(directory/'receipt.json',dict(paths=task['paths'],records=[v[1] for v in values],
        files=files,manifest_sha256=task['manifest_sha256'],features=config['features'],envelope_definition=DEFINITION))
    check_shard(directory,task,config,full=False)
    return dict(task_id=task_id,arrays=len(task['paths']),reused=False)


def commit_indexes(root,plan,assignment):
    config=read(root/FULL/'config.json');folder=root/REL
    for split in ('train','val'):
        details=plan['splits'][split];records={};shards=[]
        for entry in plan['tasks']:
            if entry['split']!=split:continue
            task=read(root/entry['path']);directory=root/entry['directory']
            receipt=check_shard(directory,task,config)
            if entry['task_id'] in assignment['completed_task_ids']:
                if sha(directory/'receipt.json')!=assignment['completed_receipt_sha256'][str(entry['task_id'])]:
                    raise ValueError('Reused receipt changed')
            for path,value in zip(receipt['paths'],receipt['records']):
                if path in records:raise ValueError('Duplicate physical RIR in shards')
                records[path]=raw_receipt(value)
            shards.append(dict(path=str((directory/'receipt.json').relative_to(root)),sha256=sha(directory/'receipt.json')))
        if len(records)!=details['arrays']:raise ValueError('Incomplete physical coverage')
        verification=root/FULL/'preprocessing'/f'{split}_verification.json'
        value=dict(status='complete',split=split,manifest_sha256=details['manifest_sha256'],
            plan_sha256=config['native_frame']['plan_sha256'],split_sha256=config['split_sha256'],
            arrays=len(records),queries=details['queries'],records=records,test_accessed=False)
        immutable_json(verification,value)
        if details['prior_verification_sha256'] and sha(verification)!=details['prior_verification_sha256']:
            raise ValueError('Existing verification receipt was replaced')
        dest=cache_directory(root,config,split)
        atomic_json(dest/'index.json',dict(status='complete',split=split,split_sha256=config['split_sha256'],
            manifest_sha256=details['manifest_sha256'],verification_sha256=sha(verification),
            features=config['features'],arrays=len(records),shards=shards))
        atomic_json(dest/'progress.json',dict(status='complete',completed_arrays=len(records),total_arrays=len(records),
            distributed=True,updated_unix=time.time()))
    report=preflight(root,config)
    if report['status']!='ready':raise RuntimeError(str(report['blockers']))
    envelope={split:envelope_check(root,config,split)[1] for split in ('train','val')}
    immutable_json(root/FULL/'ready.json',dict(status='complete',data=report,envelope=envelope,
        config_sha256=sha(root/FULL/'config.json'),plan_sha256=config['native_frame']['plan_sha256'],test_accessed=False))
    atomic_json(folder/'ready.json',dict(status='complete',total_arrays=plan['total_arrays'],test_accessed=False))
    update(root,'features','ready_for_training',status='complete',data=report,distributed=True)


def run(root,node,processes=12,threads=4):
    root=Path(root);folder=root/REL;config=read(root/FULL/'config.json')
    if os.environ.get('NVIDIA_VISIBLE_DEVICES')!='void' or os.environ.get('CUDA_VISIBLE_DEVICES')!='-1':
        raise ValueError('Both hosts are CPU-only; no GPU visibility allowed')
    plan=read(folder/'plan.json');assignment=read(folder/'assignment.json')
    if (sha(folder/'plan.json')!=assignment['plan_sha256'] or sha(root/FULL/'config.json')!=plan['config_sha256']
        or not 1<=processes<=16 or not 1<=threads<=4):raise ValueError('Invalid frozen plan/resources')
    lock=(folder/(node+'.lock')).open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    leases=[]
    if node=='node1':
        for path in [root/FULL/'features.lock',*[cache_directory(root,config,s)/'build.lock' for s in ('train','val')]]:
            lease=path.open('a+');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);leases.append(lease)
    started=time.monotonic();done=0;reused=0;tasks_done=0
    def publish(status,**extra):
        value=dict(status=status,node=node,completed_arrays=done,reused_arrays=reused,tasks_done=tasks_done,
            total_arrays=assignment['arrays'][node],processes=processes,threads_per_process=threads,
            elapsed_seconds=time.monotonic()-started,updated_unix=time.time(),test_accessed=False,**extra)
        atomic_json(folder/(node+'_status.json'),value)
        if node=='node1':
            other=read(folder/'node2_status.json') if (folder/'node2_status.json').exists() else {}
            update(root,'features','distributed_three_input_extraction',status='running',
                arrays=assignment['reused_arrays']+done+other.get('completed_arrays',0),expected=plan['total_arrays'],
                node1=value,node2=other,raw_reads_per_new_array=1)
    publish('running')
    try:
        with ProcessPoolExecutor(max_workers=processes,mp_context=multiprocessing.get_context('spawn'),
            initializer=init_worker,initargs=(str(root),node,plan,assignment,threads)) as pool:
            futures={pool.submit(process_task,i):i for i in assignment['shards'][node]}
            for future in as_completed(futures):
                result=future.result();done+=result['arrays'];tasks_done+=1
                if result['reused']:reused+=result['arrays']
                publish('running')
        if done!=assignment['arrays'][node]:raise ValueError('Node coverage mismatch')
        publish('complete')
        if node=='node1':
            while True:
                other=read(folder/'node2_status.json') if (folder/'node2_status.json').exists() else {}
                if other.get('status')=='failed':raise RuntimeError('Node2 feature extraction failed')
                if other.get('status')=='complete':break
                publish('complete');time.sleep(15)
            update(root,'features','merge_distributed_receipts',status='running',arrays=plan['total_arrays'],expected=plan['total_arrays'])
            commit_indexes(root,plan,assignment)
    except Exception:
        publish('failed',error=traceback.format_exc())
        if node=='node1':update(root,'features','failed',status='failed',error=traceback.format_exc())
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','run'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--node',choices=['node1','node2'])
    p.add_argument('--processes',type=int,default=12);p.add_argument('--threads',type=int,default=4);a=p.parse_args()
    if a.action=='prepare':prepare(a.root)
    else:run(a.root,a.node,a.processes,a.threads)

#!/usr/bin/env python3
"""Render only de-duplicated negative FP records after positive completion."""
import argparse
import fcntl
import json
import os
from pathlib import Path

from .hssd_positive_pairs import atomic_json, sha, render_task

VERSION='hard_negatives_epoch020_v1'


def check_ready(root):
    base=root/'RIR/manifests'
    total=0
    for run in ('node2_train','node2_height_supplement'):
        status=json.loads((base/'positive_pairs_v1/runs'/run/'status.json').read_text())
        if status['status']!='complete' or status.get('failed_tasks',0) or status.get('limited_smoke',False):
            raise ValueError(f'Positive run not safely complete: {run}')
        if status['completed_tasks']!=status['tasks']:raise ValueError('Incomplete positive task coverage')
        total+=status['records_written']+status['records_verified_existing']
    positive_index=json.loads((base/'positive_pairs_v1/index.json').read_text())
    expected=2*sum(e['physical_records_per_role'] for e in positive_index['scenes'] if e['split']=='train')
    if total!=expected:raise ValueError('Positive physical record coverage mismatch')
    ready=json.loads((base/VERSION/'ready.json').read_text())
    index_path=base/VERSION/'index.json'
    if ready['status']!='complete' or ready['index_sha256']!=sha(index_path):
        raise ValueError('Mining readiness/integrity failed')
    index=json.loads(index_path.read_text())
    splits=json.loads((root/'manifests/scene_splits_v1.json').read_text())
    if index['split']!='train' or index['split_sha256']!=sha(root/'manifests/scene_splits_v1.json'):
        raise ValueError('Negative split mismatch')
    if {e['scene_id'] for e in index['scenes']}!=set(splits['train']) or len(index['scenes'])!=116:
        raise ValueError('Negative scene coverage mismatch')
    if any(e['split']!='train' for e in index['scenes']):raise ValueError('Nontrain entry in mining index')
    return index


def _legacy_render(args):
    for name in ('CUDA_VISIBLE_DEVICES','NVIDIA_VISIBLE_DEVICES'):
        if os.environ.get(name) not in ('-1','void','none',''):
            raise ValueError('GPU visibility not disabled')
    # Same NFS lock as both positive jobs; no chance of heavy CPU-job overlap.
    lock=(args.root/'RIR/manifests/positive_pairs_v1/render.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    index=check_ready(args.root)
    tasks=[]
    for entry in index['scenes']:
        path=args.root/entry['schedule_path']
        if sha(path)!=entry['schedule_sha256']:raise ValueError('Schedule changed')
        schedule=json.loads(path.read_text())
        if schedule['record_directory']!='negative_records' or schedule['pose_prefix']!='n' or schedule['split']!='train':
            raise ValueError('Unsafe negative record namespace')
        if sha(args.root/schedule['pair_path'])!=schedule['pair_sha256']:
            raise ValueError('Pair links changed')
        positive=args.root/'RIR/manifests/positive_pairs_v1/schedules/train'/f'{entry["scene_id"]}.json'
        if sha(positive)!=schedule['positive_schedule_sha256']:raise ValueError('Positive schedule changed')
        ids=[p['pose_id'] for p in schedule['positions']]
        for start in range(0,len(ids),12):
            tasks.append((str(args.root),str(args.root/'source/hssd-hab'),str(args.renderer_dir),entry,'FP',ids[start:start+12],2))
    out=args.root/'RIR/manifests'/VERSION/'runs/node2_negative'
    from .hssd_rir_supervisor import supervise_tasks
    results=supervise_tasks(tasks,out,args.workers,expected_records=index['new_fp_records'])
    if sum(r['written']+r['skipped'] for r in results)!=index['new_fp_records']:
        atomic_json(out/'status.json',dict(status='failed',error='Negative record count mismatch'))
        raise ValueError('Negative physical record count mismatch')


def render(args):
    from .hssd_distributed_negatives import render as distributed_render
    return distributed_render(args)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--renderer-dir',type=Path,required=True);parser.add_argument('--workers',type=int,default=38)
    parser.add_argument('--phase',choices=['render','prepare'],default='render')
    parser.add_argument('--shard',choices=['node2','local'],default='node2')
    a=parser.parse_args()
    if not 1<=a.workers<=38:parser.error('Require 1..38 workers (2 threads each)')
    if a.phase=='prepare':
        from .hssd_distributed_negatives import prepare
        prepare(a.root)
    else:render(a)

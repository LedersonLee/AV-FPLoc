"""Durable node2 CPU-only train verification -> val RIR -> feature-cache queue.

Does not start training. Only current canonical train/val scenes are accessed.
Stage failures are explicit, not treated as completed data or skipped scenes.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import time
import traceback

from .contracts import atomic_json, immutable_json, sha
from .data import (read, splits, prepare_val, prepare_train, render_val,
                   train_verification, build_cache, preflight, DEST)


def cpu_guard(workers, render_workers):
    for key in ('CUDA_VISIBLE_DEVICES', 'NVIDIA_VISIBLE_DEVICES'):
        if os.environ.get(key) not in ('-1', 'void', 'none', ''):
            raise RuntimeError('Preprocessing must be GPU-free')
    cpus = len(os.sched_getaffinity(0))
    if workers < 1 or render_workers < 1 or workers > 76 or render_workers > 38 or render_workers*2 > cpus:
        raise ValueError('CPU worker cap/affinity violation')


def run(args):
    cpu_guard(args.feature_workers, args.render_workers)
    config = read(args.config)
    splits(args.root, config)
    output = args.root / 'AcousticContrastive/preprocessing/node2_v1'
    output.mkdir(parents=True, exist_ok=True)
    lock = (output/'queue.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    immutable_json(output/'contract.json', dict(config=config, config_sha256=sha(args.config),
        gpu_forbidden=True, feature_workers=args.feature_workers, render_workers=args.render_workers,
        source='official acoustic_branch', train_after_completion=False))
    def status(stage, **extra):
        value = dict(status='running', stage=stage, updated_unix=time.time(), gpu_forbidden=True, **extra)
        atomic_json(output/'status.json', value)
        print(json.dumps(value), flush=True)
    try:
        # Separate val schedules/index; never modify the live train RIR index.
        status('prepare_val_schedules')
        if not (args.root/DEST/'val.json').exists():
            result = prepare_val(args.root, config)
            print(json.dumps(dict(val_queries=result['queries'], val_scenes=len(result['scenes']))), flush=True)
        prepare_train(args.root, config)
        while not (args.root/'RIR/manifests/hard_negatives_epoch020_v1/distributed/verification.json').exists():
            progress_path = args.root/'RIR/manifests/hard_negatives_epoch020_v1/runs/node2_negative/status.json'
            progress = read(progress_path) if progress_path.exists() else {}
            if progress.get('status') in ('failed', 'attention_required'):
                raise RuntimeError('Upstream RIR generation needs attention; refusing downstream data')
            status('waiting_train_full_verification', upstream=progress.get('status', 'waiting'),
                   completed_tasks=progress.get('completed_tasks'), total_tasks=progress.get('tasks'))
            time.sleep(30)
        train_verification(args.root, config)
        status('render_and_verify_val_rirs')
        while True:
            try:
                render_val(args.root, config, args.renderer_dir, args.render_workers)
                break
            except BlockingIOError:
                status('waiting_train_renderer_lock_release')
                time.sleep(30)
        for split in ('train', 'val'):
            status('cache_'+split, feature_workers=args.feature_workers)
            build_cache(args.root, config, split, args.feature_workers)
        status('final_preflight')
        report = preflight(args.root, config)
        atomic_json(output/'preflight.json', report)
        if report['status'] != 'ready':
            raise RuntimeError('Final training gate failed: '+json.dumps(report['blockers']))
        atomic_json(output/'status.json', dict(status='complete', stage='training_ready',
            updated_unix=time.time(), gpu_forbidden=True, training_started=False))
    except Exception:
        atomic_json(output/'status.json', dict(status='failed', updated_unix=time.time(),
            error=traceback.format_exc(), gpu_forbidden=True, training_started=False))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--renderer-dir', type=Path, required=True)
    parser.add_argument('--feature-workers', type=int, default=64)
    parser.add_argument('--render-workers', type=int, default=38)
    run(parser.parse_args())


if __name__ == '__main__':
    main()

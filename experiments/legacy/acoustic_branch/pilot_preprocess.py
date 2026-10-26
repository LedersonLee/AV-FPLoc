"""Only selected train/val features; reuse verified full raw RIRs, no training."""
import argparse
import fcntl
import json
from pathlib import Path
import time
import traceback

from .contracts import atomic_json, immutable_json, sha
from .data import read, load_manifest, verification, build_cache, preflight
from .preprocess import cpu_guard


def run(args):
    cpu_guard(args.feature_workers, 1)
    config = read(args.config)
    if not config.get('scene_subset_path') or config['subset_counts'] != dict(train=12,val=3,test=3):
        raise ValueError('Explicit frozen pilot subset required')
    output = args.root/'AcousticContrastive/pilots/scene_subset_v1/preprocessing'
    output.mkdir(parents=True, exist_ok=True)
    lock = (output/'queue.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    immutable_json(output/'contract.json', dict(config_sha256=sha(args.config), feature_workers=args.feature_workers,
        gpu_forbidden=True, no_new_RIRs=True, training_started=False))
    try:
        for split in ('train', 'val'):
            atomic_json(output/'status.json', dict(status='running', stage='cache_'+split, updated_unix=time.time()))
            queries, digest = load_manifest(args.root, split, config)
            verification(args.root, split, config, digest)
            print(json.dumps(dict(split=split, queries=len(queries))), flush=True)
            build_cache(args.root, config, split, args.feature_workers)
        report = preflight(args.root, config)
        atomic_json(output/'preflight.json', report)
        if report['status'] != 'ready':
            raise RuntimeError(json.dumps(report['blockers']))
        atomic_json(output/'status.json', dict(status='complete', stage='pilot_training_ready',
            updated_unix=time.time(), training_started=False))
    except Exception:
        atomic_json(output/'status.json', dict(status='failed', error=traceback.format_exc(), updated_unix=time.time()))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--feature-workers', type=int, default=64)
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()

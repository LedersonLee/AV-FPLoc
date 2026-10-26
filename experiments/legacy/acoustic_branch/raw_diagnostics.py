"""Frozen train/val candidate comparisons using the colleague's exact featuriser."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import importlib.util
from pathlib import Path
import time

import numpy as np

from .contracts import atomic_json, sha
from .data import read, load_manifest


def run(root, config_path, source, output):
    spec = importlib.util.spec_from_file_location('frozen_grid_score', source)
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    base = module.GridScoreConfig(sample_rate_hz=48000, direct_guard_samples=96,
        usable_samples=6144, nfft=256, hop=64, bands=((0,500),(500,1500),(1500,4000)),
        normalise='sum', distance='symmetric', shape_feature=True, feature='stft_band')
    methods = dict(legacy_stft_l1=base, envelope_0p5ms=replace(base, feature='envelope', window_ms=.5),
        stft_early_frames12=replace(base,last_frame=12),
        stft_late_frames24plus=replace(base,first_frame=24),
        stft_500_1500=replace(base,bands=((500,1500),)),
        stft_500_1500_early=replace(base,bands=((500,1500),),last_frame=12),
        envelope_early20ms=replace(base,feature='envelope',window_ms=.5,last_frame=40))
    config = read(config_path)
    result = dict(test_accessed=False, source_sha256=sha(source), config_sha256=sha(config_path), splits={})
    for split in ('val', 'train'):
        queries, digest = load_manifest(root, split, config)
        # Train is a deterministic balanced per-scene diagnostic sample, not a fitted set.
        if split == 'train':
            scenes = sorted({q['scene_id'] for q in queries})
            queries = [q for s in scenes for q in [q for q in queries if q['scene_id']==s][::8][:32]]
        def task(q):
            refs = [q['observed']] + q['candidates']
            rirs = [np.roll(np.load(root/r['path'], allow_pickle=False), r['roll'], axis=0) for r in refs]
            row = dict(scene=q['scene_id'], sample_index=q['sample_index'], methods={})
            kinds = [r['kind'] for r in q['candidates']]
            pools = dict(all=list(range(8)), spatial=[0]+[i for i in range(1,8) if kinds[i]!='orientation_hard'],
                         orientation=[0]+[i for i in range(1,8) if kinds[i]=='orientation_hard'])
            for name, c in methods.items():
                features = [module.featurise(module.band_energy(r, c), c) for r in rirs]
                distances = [float(np.abs(f-features[0]).sum()) for f in features[1:]]
                scores = {}
                for pool, indices in pools.items():
                    d = np.array(distances)[indices]
                    winners = d == d.min()
                    scores[pool] = float(winners[0]/winners.sum())
                row['methods'][name] = dict(distances=distances, **scores)
            return row
        rows = []
        with ThreadPoolExecutor(max_workers=12) as pool:
            for row in pool.map(task, queries):
                rows.append(row)
                if len(rows)%64 == 0:
                    atomic_json(output/'raw_status.json', dict(status='running', split=split,
                        completed=len(rows), total=len(queries), updated_unix=time.time()))
        result['splits'][split] = dict(queries=len(rows), manifest_sha256=digest,
            metrics={name:{p:float(np.mean([r['methods'][name][p] for r in rows]))
                           for p in ('all','spatial','orientation')} for name in methods}, rows=rows)
        atomic_json(output/'raw_results.json', result)
    atomic_json(output/'raw_status.json', dict(status='complete', test_accessed=False, updated_unix=time.time()))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    try:
        run(a.root, a.config, a.source, a.output)
    except Exception as e:
        atomic_json(a.output/'raw_status.json', dict(status='failed', error=repr(e)))
        raise

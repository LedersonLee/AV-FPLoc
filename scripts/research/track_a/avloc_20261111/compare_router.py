"""Frozen A3 contrasts; paired development scenes, no model selection."""
import argparse
import json
from pathlib import Path
import numpy as np
from paired_summary import align, paired
from predict_hash import sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--grid-cm', type=int, choices=[20, 10], required=True)
    p.add_argument('--include-uniform', action='store_true')
    a = p.parse_args()
    manifest = json.loads((a.study/'predictions/A3_seed42/manifest.json').read_text())
    assert manifest['complete'] and manifest['queries'] == 9965
    expected = []
    for rec in manifest['records']:
        path = a.study/'predictions/A3_seed42'/rec['file']
        assert sha(path) == rec['sha256']
        with np.load(path) as z:
            expected.extend((rec['scene'], int(q)) for q in z['query_ids'])
    assert len(expected) == len(set(expected)) == 9965
    scenes = [s for s, q in expected]
    assert len(set(scenes)) == 19
    sources, methods = {}, {}
    tags = ['A3_seed42', 'A2_seed42', 'A0_seed42', 'A2_disabled_seed42']
    if a.include_uniform:
        tags.append('A3_seed42_uniform')
    for tag in tags:
        root = a.study/'evaluation'/f'{tag}_{a.grid_cm}cm'
        summary = json.loads((root/'summary.json').read_text())
        assert summary['complete'] and summary['queries'] == 9965
        assert summary['grid_cm'] == a.grid_cm
        rows, hashes = [], {}
        for sid in sorted(set(scenes)):
            path = root/f'{sid}.json'
            rows.extend(json.loads(path.read_text()))
            hashes[sid] = sha(path)
        methods[tag] = [align(rows, expected)]
        sources[tag] = dict(summary_sha256=sha(root/'summary.json'), rows_sha256=hashes)
    if a.grid_cm == 20:
        methods['B2_GEOQ'] = []
        sources['B2_GEOQ'] = {}
        for seed in [42, 43]:
            path = a.study/f'baselines/geoq_replay_v1/seed{seed}_queries.json'
            methods['B2_GEOQ'].append(align(json.loads(path.read_text()), expected))
            sources['B2_GEOQ'][str(seed)] = sha(path)
    contrasts = {}
    for name, rows in methods.items():
        if name == 'A3_seed42':
            continue
        contrasts[name] = {metric: paired(
            [[r[metric] for r in rs] for rs in methods['A3_seed42']],
            [[r[metric] for r in rs] for rs in rows], scenes)
            for metric in ['r1', 'joint', 'fine']}
    result = dict(complete=True, queries=9965, scenes=19, grid_cm=a.grid_cm,
        contrasts=contrasts, sources=sources,
        source_sha256={n: sha(Path(__file__).with_name(n)) for n in ['compare_router.py', 'paired_summary.py']},
        scope='Seed42; reused development, not independent validation. B2/GEOQ averages seeds42/43 within scenes. Uniform weights reuse A3 heads, not separately trained fixed routing. A3-vs-A2 changes training and architecture, not an isolated router intervention.',
        test_accessed=False)
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({n: {k: {x:v[x] for x in ['micro_delta','macro_ci95','repair_count_seed_mean','harm_count_seed_mean']} for k,v in rows.items()} for n,rows in contrasts.items()}, indent=2))


if __name__ == '__main__':
    main()

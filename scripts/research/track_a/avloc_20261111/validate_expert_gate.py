"""Validate completed seed-matched experts before A3 training, without a score threshold."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def validate(study, seed):
    data = read(study / 'data_v1/manifest.json')
    expected = {r['scene']: r for r in data['records'] if r['split'] == 'val'}
    assert len(expected) == 19 and sum(r['queries'] for r in expected.values()) == 9965
    gate = {'scope': 'Reused Val19, integrity/evaluation gate; not proof of accuracy or independent validation',
            'seed': seed, 'test_accessed': False, 'validator_sha256': sha(__file__)}
    for arm, head in [('A0', 'V'), ('A1', 'A'), ('A2', 'AV')]:
        run = study / f'runs/{arm}_seed{seed}'
        pred = study / f'predictions/{arm}_seed{seed}'
        protocol, completion, best = [read(run / f'{name}.json') for name in ['protocol', 'complete', 'best']]
        manifest = read(pred / 'manifest.json')
        assert completion['status'] == 'training_complete'
        assert protocol['arm'] == arm and protocol['seed'] == seed
        assert protocol['train_queries'] == 58894 and protocol['val_queries'] == 9965
        assert protocol['test_accessed'] is False
        assert manifest['complete'] and manifest['split'] == 'val' and manifest['queries'] == 9965
        assert manifest['arm'] == arm and manifest['seed'] == seed and manifest['test_accessed'] is False
        assert manifest['epoch'] == best['epoch']
        assert manifest['checkpoint_sha256'] == sha(run / best['path'])
        assert manifest['input_manifest_sha256'] == protocol['input_manifest_sha256'] == sha(study / 'data_v1/manifest.json')
        assert manifest['training_code_commit'] == protocol['code_commit']
        assert len(manifest['records']) == 19 and {r['scene'] for r in manifest['records']} == set(expected)
        evidence = {}
        for grid in [20, 10]:
            evdir = study / f'evaluation/{arm}_seed{seed}_{grid}cm'
            summary = read(evdir / 'summary.json')
            assert summary['complete'] and summary['queries'] == 9965 and summary['head'] == head
            assert summary['grid_cm'] == grid and summary['test_accessed'] is False
            assert summary['prediction_manifest_sha256'] == sha(pred / 'manifest.json')
            allrows, absolute_errors = [], []
            for rec in manifest['records']:
                sid = rec['scene']; pp = pred / rec['file']; lp = study / f'data_v1/{sid}/labels.npz'
                assert sha(pp) == rec['sha256'] and sha(lp) == expected[sid]['files']['labels.npz']
                with np.load(pp) as z, np.load(lp) as labels:
                    ids = z['query_ids']; assert np.array_equal(ids, labels['query_ids'])
                    assert len(ids) == rec['queries'] == expected[sid]['queries']
                    mu, scale = z[head+'_mu'].astype(float), z[head+'_scale'].astype(float)
                    dirs = np.array([31,32,33,34,35,0,1,2,3,4,5]) if head == 'V' else np.arange(36)
                    assert mu.shape == scale.shape == (len(ids), len(dirs))
                    assert np.isfinite(mu).all() and np.isfinite(scale).all() and (scale > 0).all()
                    err = abs(mu-labels['target'][:, dirs]).mean(axis=1)
                    rows = read(evdir / f'{sid}.json')
                    assert [r['query_id'] for r in rows] == ids.tolist()
                    assert all(r['scene'] == sid for r in rows)
                    np.testing.assert_allclose([r['ray_mae'] for r in rows], err, rtol=1e-6, atol=1e-6)
                    absolute_errors.extend(err.tolist()); allrows.extend(rows)
            assert len(allrows) == 9965
            for metric, value in summary['metric_means'].items():
                np.testing.assert_allclose(np.mean([r[metric] for r in allrows]), value, rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(np.mean(absolute_errors), best['ray_mae'], rtol=1e-5, atol=1e-6)
            evidence[str(grid)] = {'summary_sha256': sha(evdir/'summary.json'), 'metrics': summary['metric_means'], 'verified_queries': len(allrows)}
        gate[arm] = {'validated': True, 'checkpoint_sha256': manifest['checkpoint_sha256'],
                     'prediction_manifest_sha256': sha(pred/'manifest.json'), 'complete_sha256': sha(run/'complete.json'),
                     'training_code_commit': protocol['code_commit'], 'best_epoch': best['epoch'], 'evaluation': evidence}
    return gate


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--study', type=Path, required=True); p.add_argument('--seed', type=int, required=True); p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); result = validate(a.study, a.seed)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('x') as f: json.dump(result, f, indent=2); f.write('\n')
    print('PASS: three completed experts, both grids, 9965 aligned queries per expert/grid; no accuracy threshold')

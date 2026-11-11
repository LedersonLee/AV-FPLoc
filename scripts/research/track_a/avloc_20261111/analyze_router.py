"""Post-hoc A3 routing diagnostics; never feeds labels into model inference."""
import argparse
import json
from pathlib import Path

import numpy as np

from analyze_rays import correlation, ray_statistics
from matching import FOV
from predict_hash import sha


def router_statistics(z, target):
    target = np.asarray(target, dtype=float)
    assert target.ndim == 2 and target.shape[1] == 36
    n = len(target)
    means = np.zeros((n, 36, 3), dtype=float)
    scales = np.zeros_like(means)
    available = np.ones((36, 3), dtype=bool)
    available[:, 0] = False
    available[FOV, 0] = True
    for i, head in enumerate(['V', 'A', 'AV']):
        dirs = FOV if head == 'V' else np.arange(36)
        mu, scale = np.asarray(z[head + '_mu']), np.asarray(z[head + '_scale'])
        assert mu.shape == scale.shape == (n, len(dirs))
        assert np.isfinite(mu).all() and np.isfinite(scale).all() and (scale > 0).all()
        means[:, dirs, i] = mu
        scales[:, dirs, i] = scale
    weights = np.asarray(z['weights'], dtype=float)
    assert weights.shape == means.shape and np.isfinite(weights).all()
    assert (weights >= 0).all() and (weights <= 1).all()
    np.testing.assert_allclose(weights.sum(-1), 1, atol=2e-6, rtol=0)
    assert np.all(weights[:, ~available] == 0), 'Unavailable visual expert must have zero weight'
    fused = (weights * means).sum(-1)
    scale = np.sqrt((weights * (2 * scales**2 + (means - fused[..., None])**2)).sum(-1) / 2)
    np.testing.assert_allclose(fused, z['F_mu'], rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(scale, z['F_scale'], rtol=2e-5, atol=2e-6)
    assert np.isfinite(target).all()
    error = abs(means - target[..., None])
    uniform = available / available.sum(-1, keepdims=True)
    uniform_mu = (means * uniform).sum(-1)
    directions = []
    for d in range(36):
        active = available[d]
        w, err, b = weights[:, d, active], error[:, d, active], scales[:, d, active]
        # Tied minimum errors all count as correct choices; argmax weight ties
        # use the deterministic V/A/AV order, matching NumPy's first index rule.
        selected = err[np.arange(n), w.argmax(-1)]
        minimum = err.min(-1)
        directions.append(dict(direction_bin=d, angle_deg=d * 10,
            max_weight_expert_is_min_error_fraction=float(np.isclose(selected, minimum, atol=1e-8, rtol=0).mean()),
            max_weight_selection_excess_mae=float((selected - minimum).mean()),
            weighted_expert_abs_error=float((w * err).sum(-1).mean()),
            fused_mae=float(abs(fused[:, d] - target[:, d]).mean()),
            uniform_same_heads_mae=float(abs(uniform_mu[:, d] - target[:, d]).mean()),
            experts=[dict(expert=['V', 'A', 'AV'][i], mean_weight=float(weights[:, d, i].mean()),
                mae=float(error[:, d, i].mean()), mean_scale=float(scales[:, d, i].mean()),
                weight_abs_error_spearman=correlation(weights[:, d, i], error[:, d, i]),
                scale_abs_error_spearman=correlation(scales[:, d, i], error[:, d, i]),
                weight_scale_spearman=correlation(weights[:, d, i], scales[:, d, i]))
                for i in np.flatnonzero(active)]))
    return dict(queries=n, per_direction=directions,
        fused=ray_statistics(fused, scale, target, np.arange(36)),
        uniform_same_heads_mae=float(abs(uniform_mu-target).mean()),
        fused_minus_uniform_mae=float((abs(fused-target)-abs(uniform_mu-target)).mean()),
        replay='Saved weights reproduce saved fused mean and moment-matched scale')


def main():
    p = argparse.ArgumentParser()
    for key in ['predictions', 'data', 'out']:
        p.add_argument('--' + key, type=Path, required=True)
    a = p.parse_args()
    pred = json.loads((a.predictions / 'manifest.json').read_text())
    data = json.loads((a.data / 'manifest.json').read_text())
    assert pred['complete'] and pred['arm'] == 'A3' and pred['split'] == 'val'
    assert pred['queries'] == 9965 and pred['input_manifest_sha256'] == sha(a.data / 'manifest.json')
    expected = {r['scene'] for r in data['records'] if r['split'] == 'val'}
    assert len(expected) == len(pred['records']) == 19
    assert {r['scene'] for r in pred['records']} == expected
    arrays, targets, scenes, sources = {}, [], [], []
    for r in pred['records']:
        path = a.predictions / r['file']
        assert sha(path) == r['sha256']
        labels = a.data / r['scene'] / 'labels.npz'
        with np.load(path) as z, np.load(labels) as lab:
            np.testing.assert_array_equal(z['query_ids'], lab['query_ids'])
            assert len(z['query_ids']) == r['queries']
            target = lab['target'].copy()
            stats = router_statistics(z, target)
            scenes.append(dict(scene=r['scene'], **stats))
            for k in ['V_mu', 'V_scale', 'A_mu', 'A_scale', 'AV_mu', 'AV_scale', 'F_mu', 'F_scale', 'weights']:
                arrays.setdefault(k, []).append(z[k].copy())
            targets.append(target)
        sources.append(dict(scene=r['scene'], prediction_sha256=sha(path), labels_sha256=sha(labels)))
    target = np.concatenate(targets)
    assert len(target) == 9965
    summary = router_statistics({k: np.concatenate(v) for k, v in arrays.items()}, target)
    summary.update(complete=True, seed=pred['seed'], scenes=scenes, sources=sources,
        prediction_manifest_sha256=sha(a.predictions / 'manifest.json'), test_accessed=False,
        source_sha256={n: sha(Path(__file__).with_name(n)) for n in ['analyze_router.py', 'analyze_rays.py', 'matching.py', 'predict_hash.py']},
        scope='Reused Val19 descriptive diagnostics; no independent reliability or causal claim. Ray/query correlations are not independent observations. Uniform is an inference-only control using the same learned experts, not separately trained fixed routing. GT minimum-error choice is post-hoc only and never an inference proposal.')
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print('ROUTER ANALYSIS COMPLETE', len(target), flush=True)


if __name__ == '__main__':
    main()

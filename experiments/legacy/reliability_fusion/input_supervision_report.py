"""Paired position-cluster uncertainty and static report for Train/Val audit."""
import argparse
import csv
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.stats import rankdata
from scipy.special import expit, logit

from .input_supervision_audit import REL, SOURCE, read, save, resolve, pose, winner_at, recall


def aucs(y, probabilities):
    n1 = int(y.sum()); n0 = len(y) - n1
    if not n1 or not n0:
        return None
    ranks = rankdata(probabilities, axis=0)
    return (ranks[y.astype(bool)].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)


def cluster_intervals(out, rows, draws=1000):
    if (out / 'paired_auc_intervals.json').exists():
        return read(out / 'paired_auc_intervals.json')
    intervals = []
    for path in sorted((out / 'classifiers').glob('*.npz')):
        with np.load(path) as b:
            names = [n for n in b.files if n not in ('y', 'groups', 'global_ids')]
            y = b['y'].copy(); groups = b['groups'].copy(); ids = b['global_ids'].copy()
            prob = np.column_stack([b[n] for n in names])
        base = names.index('core14'); parts = []
        for scene in np.unique(groups):
            cluster_ids = np.array([str(rows[i]['pose_id']) for i in ids[groups == scene]])
            loc = np.flatnonzero(groups == scene)
            parts.append([loc[cluster_ids == p] for p in np.unique(cluster_ids)])
        rng = np.random.default_rng(20261021); diffs = []
        for _ in range(draws):
            sample = np.concatenate([np.concatenate([p[i] for i in rng.integers(0, len(p), len(p))]) for p in parts])
            a = aucs(y[sample], prob[sample])
            if a is not None:
                diffs.append(a - a[base])
        diffs = np.asarray(diffs)
        point = aucs(y, prob)
        method, criterion, weight_index = path.stem.split('_')
        for i, name in enumerate(names):
            intervals.append(dict(model=method, criterion=('xy', 'joint')[int(criterion)],
                action_weight=int(weight_index) * .05, variant=name,
                delta_auc=float(point[i]-point[base]), ci95=np.quantile(diffs[:, i], [.025, .975]).tolist(), draws=len(diffs)))
    result = dict(unit='Physical pose, all outcome-changing yaws resampled together; stratified within each fixed Val scene',
                  caveat='Conditional on these3 Val scenes; not an unseen-scene population confidence interval. Exploratory multiple comparisons.',
                  rows=intervals)
    save(out / 'paired_auc_intervals.json', result)
    return result


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def run(args):
    from acoustic_branch.contracts import sha
    out = args.root / REL; result = read(out / 'results.json'); report = args.report
    report.mkdir(parents=True, exist_ok=True)
    refs = read(args.root / SOURCE / 'scores_ready.json')['entries']
    rows = [r for ref in refs for r in read(out / 'scenes' / ref['scene_id'] / 'queries.json')]
    assert {r['split'] for r in rows} == {'train', 'val'} and len(rows) == 8091
    assert result['verification']['baseline_reproduced']
    # Intercept-only minimizer of the SAME scene-balanced logit target MSE.
    # This is a diagnostic counterexample, not another Val-selected model.
    target_means = []
    for sid in result['config']['selection']['train']:
        target_means.append(np.mean([logit(r['selected_target']) for r in rows if r['scene_id'] == sid and r['selected_valid']]))
    weight = float(expit(np.mean(target_means)))
    xy = {ref['scene_id']: np.load(resolve(ref, args.ssd) / 'xy.npy') for ref in refs}
    predicted = np.array([pose(winner_at(r['parts'], weight), xy[r['scene_id']]) for r in rows])
    fixed = np.array([pose(winner_at(r['parts'], .1), xy[r['scene_id']]) for r in rows])
    truth = np.array([r['truth'] for r in rows])
    frozen_fixed = np.concatenate([np.load(out / 'scenes' / ref['scene_id'] / 'arrays.npz')['grid_poses'][:, 2] for ref in refs])
    np.testing.assert_allclose(fixed, frozen_fixed, atol=1e-7, rtol=0)
    constant = dict(weight=weight, fit='Train scene-balanced mean of target logit; no input features, no Val fitting', splits={})
    for split in ('train', 'val'):
        ids = np.array([r['split'] == split for r in rows])
        losses = []
        for sid in result['config']['selection'][split]:
            tt = np.array([logit(r['selected_target']) for r in rows if r['scene_id'] == sid and r['selected_valid']])
            losses.append(dict(initial=float(np.mean((logit(.1)-tt)**2)), optimized=float(np.mean((logit(weight)-tt)**2))))
        constant['splits'][split] = dict(metrics=recall(predicted[ids], truth[ids], fixed[ids]),
            scene_balanced_loss_initial=float(np.mean([l['initial'] for l in losses])),
            scene_balanced_loss_optimized=float(np.mean([l['optimized'] for l in losses])))
    result['constant_logit_target_control'] = constant
    save(out / 'results.json', result)
    ci = cluster_intervals(out, rows)
    examples = read(out / 'nearest_examples.json')
    for pair in examples:
        for name in ('query', 'neighbour'):
            r = pair[name]; r['rgb'] = 'rgb/' + r['scene_id'] + '_' + str(r['sample_index']) + '.png'
            dest = report / r['rgb']; dest.parent.mkdir(exist_ok=True)
            if not dest.exists():
                shutil.copy2(args.ssd / r['image_path'], dest)
    for name, obj in [('results.json', result), ('examples.json', examples), ('paired_auc_intervals.json', ci),
                      ('feature_definitions.json', read(out / 'feature_definitions.json'))]:
        # Strict JSON, not Python's non-standard NaN literals in legacy labels.
        (report / name).write_text(json.dumps(clean(obj), indent=2, allow_nan=False) + '\n')
    shutil.copy2(out / 'classifier_metrics.csv', report / 'classifier_metrics.csv')
    shutil.copy2(Path(__file__).with_name('input_supervision_report.html'), report / 'index.html')
    verification = dict(status='passed', train_scenes=12, val_scenes=3, train_queries=6387, val_queries=1704, test_accessed=False,
        continuous_argmax_checks=result['verification']['envelope_argmax_checks'], baseline_reproduced=True,
        gt_free_features='No GT argument to engineered(); GT only builds success/outcome labels',
        normalization='Train only; classifier scalers refit within each Train group fold',
        source_receipts=[dict(scene_id=ref['scene_id'], split=ref['split'], receipt_sha256=ref['receipt_sha256']) for ref in refs],
        report_files={n: sha(report / n) for n in ('results.json', 'examples.json', 'paired_auc_intervals.json', 'index.html')},
        source_code={n: sha(Path(__file__).with_name(n)) for n in ('input_supervision_audit.py', 'input_supervision_report.py', 'input_supervision_report.html')})
    save(out / 'verification.json', verification); save(report / 'verification.json', verification)
    print('PAIRED AUC CHECKS', len(ci['rows']), flush=True)
    print('REPORT', report, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True); p.add_argument('--ssd', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True); run(p.parse_args())

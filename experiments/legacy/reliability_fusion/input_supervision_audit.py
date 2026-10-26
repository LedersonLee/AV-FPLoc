"""Train12/Val3-only feature/target audit. No Test arrays or RIRs are opened.

Continuous affine MAP envelopes avoid treating the old 0.05 weight grid as an
exact success set. GT is used ONLY for diagnostic labels/evaluation; all feature
construction is in a separate GT-free function. Classifiers are diagnostic, not
a newly selected fusion policy. No acoustic/visual encoder is updated.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.special import softmax, expit, logit
from scipy.spatial.distance import cdist

SOURCE = 'ReliabilityFusion/pilots/subset12_logit_v1'
REL = 'ReliabilityFusion/pilots/input_supervision_audit_v1'
CORE_NAMES = [f'{m}_{f}' for m in ('vision', 'acoustic') for f in
              ('separated_gap', 'spread_over_diagonal', 'peak_mass_1m', 'xy_entropy', 'yaw_entropy')]
CORE_NAMES += ['map_distance_over_diagonal', 'yaw_disagreement_fraction', 'xy_overlap', 'top50_overlap']
RESPONSE_WEIGHTS = [.05, .15, .2, .3, .5]


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    from acoustic_branch.contracts import atomic_json
    atomic_json(path, value)


def envelope(v, a):
    """All positive-width MAP-winner intervals of (1-w)*v+w*a, float64.

    Divide at intersections of endpoint winners. If another line lies above
    their intersection, recurse; otherwise the pair forms the upper envelope.
    Isolated tie ownership at endpoints is intentionally not an interval.
    """
    v = np.asarray(v, np.float64).ravel()
    d = np.asarray(a, np.float64).ravel() - v
    parts = []

    def visit(p, q, lo, hi, depth=0):
        if depth > 500:
            raise ValueError('Envelope recursion failed')
        if p == q:
            parts.append([lo, hi, int(p)])
            return
        den = d[q] - d[p]
        if den <= 0:
            raise ValueError('Endpoint slope order failed')
        t = float((v[p] - v[q]) / den)
        if not lo - 1e-9 <= t <= hi + 1e-9:
            raise ValueError('Intersection outside bracket')
        t = float(np.clip(t, lo, hi))
        scores = v + t * d
        r = int(scores.argmax())
        tol = 1e-11 * max(1., abs(float(scores[r])))
        if scores[r] - scores[p] <= tol:
            parts.extend([[lo, t, int(p)], [t, hi, int(q)]])
        else:
            visit(p, r, lo, t, depth + 1)
            visit(r, q, t, hi, depth + 1)

    visit(int(v.argmax()), int(np.asarray(a).argmax()), 0., 1.)
    result = []
    for lo, hi, idx in parts:
        if hi - lo <= 1e-13:
            continue
        if result and result[-1][2] == idx:
            result[-1][1] = hi
        else:
            result.append([lo, hi, idx])
    if abs(sum(hi - lo for lo, hi, _ in result) - 1) > 1e-8:
        raise ValueError('Incomplete envelope')
    return result


def winner_at(parts, w):
    for lo, hi, idx in parts:
        if lo - 1e-12 <= w <= hi + 1e-12:
            return int(idx)
    raise ValueError('Weight outside envelope')


def merge_intervals(intervals):
    out = []
    for lo, hi in sorted(intervals):
        if out and lo <= out[-1][1] + 1e-12:
            out[-1][1] = max(hi, out[-1][1])
        else:
            out.append([float(lo), float(hi)])
    return out


def measure(intervals):
    return sum(hi - lo for lo, hi in intervals)


def intersection(x, y):
    return sum(max(0., min(b, d) - max(a, c)) for a, b in x for c, d in y)


def inside(intervals, w):
    return any(lo <= w <= hi for lo, hi in intervals)


def pose(idx, xy):
    return np.r_[xy[idx // 36], idx % 36 * 10.]


def yawdiff(a, b):
    return np.abs((np.asarray(a) - np.asarray(b) + 180.) % 360. - 180.)


def engineered(v, a, xy, depth, ray_bank, parts):
    """NO GT, success labels, scene IDs, absolute coordinates or Laplace scale.

    Relative geometric distances are normalized by map diagonal. Residuals use
    UnLoc predicted depth and DESDF only, not captured/furnished/GT ray depth.
    """
    diag = max(float(np.linalg.norm(np.ptp(xy, axis=0))), 1.)
    fields = [np.asarray(v), np.asarray(a)]
    ids = [int(s.argmax()) for s in fields]
    points = [pose(i, xy) for i in ids]
    names, values, families = [], [], []

    def add(name, val, family):
        names.append(name); values.append(float(val)); families.append(family)

    angles = np.deg2rad(np.arange(-5, 6) * 10.)
    coords = np.tan(angles) * len(depth) * 3 / 8 + (len(depth) - 1) / 2
    rays = np.interp(coords, np.arange(len(depth)), depth) / np.cos(angles)
    for name, val in [('mean', depth.mean()), ('cv', depth.std() / max(abs(depth.mean()), 1e-3)),
                      ('range', np.ptp(depth)), ('adjacent_variation', np.abs(np.diff(depth)).mean())]:
        add('depth_' + name, val, 'depth')
    for k, field in enumerate(fields):
        label = ('vision', 'acoustic')[k]
        own, other = ids[k], ids[1-k]
        flat = field.ravel(); s = field.max(1); other_cell = other // 36
        add(label + '_other_pose_rank_fraction', np.mean(flat >= flat[other]), 'support')
        add(label + '_other_xy_rank_fraction', np.mean(s >= s[other_cell]), 'support')
        add(label + '_other_pose_score_deficit', flat[own] - flat[other], 'support')
        add(label + '_other_xy_score_deficit', s[own // 36] - s[other_cell], 'support')
        dist = np.linalg.norm(xy - points[1-k][:2], axis=1)
        z = (s - s.mean()) / max(s.std(), 1e-8)
        add(label + '_mass_at_other_1m', softmax(z)[dist <= 1].sum(), 'support')
        top = np.argsort(-s, kind='stable')[:50]; modes = []
        for idx in top:
            if not modes or np.min(np.linalg.norm(xy[modes] - xy[idx], axis=1)) > 1:
                modes.append(int(idx))
        add(label + '_top50_spatial_modes', len(modes), 'support')
    vs, acs = v.max(1), a.max(1)
    add('xy_score_correlation', np.corrcoef(vs, acs)[0, 1], 'support')
    topn = max(1, int(np.ceil(len(xy) * .01)))
    vt = np.argpartition(-vs, topn - 1)[:topn]; at = np.argpartition(-acs, topn - 1)[:topn]
    add('top1percent_xy_overlap', len(set(vt) & set(at)) / topn, 'support')
    base = pose(winner_at(parts, .1), xy)
    for w in RESPONSE_WEIGHTS:
        p = pose(winner_at(parts, w), xy)
        add(f'w{w:g}_move_over_diagonal', np.linalg.norm(p[:2] - base[:2]) / diag, 'response')
        add(f'w{w:g}_yaw_move_fraction', yawdiff(p[2], base[2]) / 180., 'response')
    selected = next((lo, hi) for lo, hi, idx in parts if lo <= .1 <= hi)
    add('fixed01_winner_interval_width', selected[1] - selected[0], 'response')
    add('fixed01_stability_left', .1 - selected[0], 'response')
    add('fixed01_stability_right', selected[1] - .1, 'response')
    add('winner_transitions_to05', sum(lo > .1 and lo < .5 for lo, _, _ in parts), 'response')
    bearings = (np.arange(36)[:, None] + np.arange(-5, 6)) % 36
    for name, pidx in [('vision', ids[0]), ('acoustic', ids[1]), ('fixed01', winner_at(parts, .1))]:
        pred = ray_bank[pidx // 36, bearings[pidx % 36]]
        delta = np.abs(pred - rays)
        add(name + '_ray_mae_m', delta.mean(), 'ray_fit')
        add(name + '_ray_median_ae_m', np.median(delta), 'ray_fit')
        add(name + '_ray_fraction_within05m', (delta <= .5).mean(), 'ray_fit')
        add(name + '_ray_relative_mae', delta.mean() / max(float(np.mean(np.abs(rays))), .1), 'ray_fit')
    result = np.asarray(values, np.float64)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite GT-free features')
    return result, names, families


def resolve(ref, ssd):
    folder = ref['folder']
    if folder.startswith('/cache/'):
        return ssd / SOURCE / 'cache' / folder[len('/cache/'):]
    if folder.startswith('/ssd/HSSD/'):
        return ssd / folder[len('/ssd/HSSD/'):]
    raise ValueError('Unapproved source path')


def extract_scene(task):
    root, ssd, out, ref = task
    root, ssd, out = map(Path, (root, ssd, out))
    from acoustic_branch.contracts import sha
    sid, split = ref['scene_id'], ref['split']
    if split not in ('train', 'val'):
        raise ValueError('Test/unknown split is forbidden')
    dest = out / 'scenes' / sid
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / 'complete.json').exists():
        receipt = read(dest / 'complete.json')
        assert sha(dest / 'arrays.npz') == receipt['arrays_sha256']
        assert sha(dest / 'queries.json') == receipt['queries_sha256']
        return receipt
    source = resolve(ref, ssd)
    assert sha(source / 'receipt.json') == ref['receipt_sha256']
    receipt = read(source / 'receipt.json')
    for name, digest in receipt['files'].items():
        assert sha(source / name) == digest, 'Source checksum changed: ' + sid + '/' + name
    arrays = {n: np.load(source / (n + '.npy'), mmap_mode='r') for n in ('scores', 'features', 'truth', 'xy')}
    queries = read(source / 'queries.json'); xy = arrays['xy']
    targets = ssd / SOURCE / 'cache/targets' / split / (sid + '.npz')
    tr = read(targets.with_suffix('.json'))
    assert sha(targets) == tr['sha256'] and tr['score_receipt_sha256'] == ref['receipt_sha256']
    with np.load(targets) as blob:
        labels = {n: blob[n].copy() for n in blob.files}
    # Old score banks used v1/vision; new Train banks used train_likelihood.
    # There is deliberately no fallback into any Test evaluation directory.
    if ref['base'] == 'old':
        depth_path = root / 'ReliabilityFusion/pilots/v1/vision' / split / (sid + '.npz')
        dr = read(depth_path.with_suffix('.json'))
        assert sha(depth_path) == dr['data_sha256']
    else:
        assert split == 'train'
        depth_path = ssd / 'UnLoc/train_likelihood/epoch020/depth' / (sid + '.npz')
    with np.load(depth_path) as blob:
        lut = {int(v): i for i, v in enumerate(blob['sample_indices'])}
        depths = blob['location'][[lut[q['sample_index']] for q in queries]].copy()
    payload = np.load(ssd / 'UnLoc/structural_targets' / sid / 'desdf.npy', allow_pickle=True).item()
    assert payload['max_range_m'] == 10 and payload['desdf_spacing_m'] == .1
    origin = np.array([payload['origin_x_m'], payload['origin_z_min_m']])
    ij = np.rint((xy - origin) / .1).astype(int)
    np.testing.assert_allclose(origin + ij * .1, xy, atol=1e-6, rtol=0)
    ray_bank = payload['desdf'][ij[:, 1], ij[:, 0]]
    rows = []; extra = []; poses_grid = []; envelope_checks = 0
    for i, (v, a) in enumerate(arrays['scores']):
        parts = envelope(v, a)
        # Independent dense argmax check at old grid AND all interval interiors.
        for w in np.r_[np.linspace(0, 1, 21), [(lo + hi) / 2 for lo, hi, _ in parts]]:
            idx = winner_at(parts, float(w)); direct = int(np.argmax((1-w)*v + w*a))
            if idx != direct:
                flat = ((1-w)*v + w*a).ravel()
                assert abs(flat[idx] - flat[direct]) <= 1e-9
            envelope_checks += 1
        gt = arrays['truth'][i]; pp = np.array([pose(idx, xy) for _, _, idx in parts])
        err = np.linalg.norm(pp[:, :2] - gt[:2], axis=1)
        angle = yawdiff(pp[:, 2], gt[2]); successes = {}
        for name, good in [('xy', err <= 1), ('joint', (err <= 1) & (angle <= 30))]:
            successes[name] = merge_intervals([[lo, hi] for (lo, hi, _), ok in zip(parts, good) if ok])
        # Preserve np.argmax's exact boundary-tie ownership for the old grid.
        grid_poses = np.array([pose(int(np.argmax((1-w)*v+w*a)), xy) for w in np.linspace(0, 1, 21)])
        np.testing.assert_allclose(grid_poses, labels['grid_poses'][i], atol=1e-7, rtol=0)
        vals, names, families = engineered(v, a, xy, depths[i], ray_bank, parts)
        item = dict(scene_id=sid, split=split, row=i, sample_index=queries[i]['sample_index'],
                    pose_id=queries[i]['pose_id'], image_path=queries[i]['image_path'], truth=gt.tolist(),
                    parts=parts, success=successes, selected_target=float(labels['interior_target'][i]),
                    selected_valid=bool(labels['valid'][i]), selected_bounds=labels['target_bounds'][i].tolist() if labels['valid'][i] else None)
        rows.append(item); extra.append(vals); poses_grid.append(grid_poses)
        if i % 100 == 0:
            save(dest / 'progress.json', dict(scene_id=sid, done=i, total=len(queries), test_accessed=False))
    np.savez_compressed(dest / 'arrays.npz', core=arrays['features'], extra=np.array(extra),
                        grid_poses=np.array(poses_grid), truth=arrays['truth'])
    save(dest / 'queries.json', rows)
    result = dict(scene_id=sid, split=split, queries=len(rows), feature_names=names, feature_families=families,
                  source_receipt_sha256=ref['receipt_sha256'], arrays_sha256=sha(dest / 'arrays.npz'),
                  queries_sha256=sha(dest / 'queries.json'), envelope_argmax_checks=envelope_checks,
                  depth_path=str(depth_path), test_accessed=False)
    save(dest / 'complete.json', result)
    return result


def recall(poses, truth, fixed):
    d = np.linalg.norm(poses[:, :2] - truth[:, :2], axis=1)
    a = yawdiff(poses[:, 2], truth[:, 2])
    fd = np.linalg.norm(fixed[:, :2] - truth[:, :2], axis=1)
    fa = yawdiff(fixed[:, 2], truth[:, 2])
    result = {f'recall_{t:g}m': float((d <= t).mean()) for t in (.1, .5, 1)}
    result.update({f'yaw_{t}deg': float((a <= t).mean()) for t in (10, 30, 50)})
    result.update(recall_1m_30deg=float(((d <= 1) & (a <= 30)).mean()), median_xy_m=float(np.median(d)), median_yaw_deg=float(np.median(a)))
    for name, before, after in [('xy', fd <= 1, d <= 1), ('joint', (fd <= 1) & (fa <= 30), (d <= 1) & (a <= 30))]:
        result['repair_' + name] = int((~before & after).sum()); result['harm_' + name] = int((before & ~after).sum())
    return result


def pair_statistics(rows, indices, distances):
    pairs = []; nearest = []
    for i, (nn, ds) in enumerate(zip(indices, distances)):
        for j, dist in zip(nn, ds):
            x, y = rows[i], rows[int(j)]; sx, sy = x['success']['joint'], y['success']['joint']
            both = bool(sx and sy); overlap = intersection(sx, sy) if both else 0.
            valid = x['selected_valid'] and y['selected_valid']
            selected_overlap = max(0., min(x['selected_bounds'][1], y['selected_bounds'][1]) - max(x['selected_bounds'][0], y['selected_bounds'][0])) if valid else None
            pair = dict(i=i, j=int(j), rms_distance=float(dist), both_feasible=both,
                        success_disjoint=bool(both and overlap <= 1e-12), jaccard=overlap / (measure(sx) + measure(sy) - overlap) if both else None,
                        selected_disjoint=bool(selected_overlap <= 1e-12) if valid else None,
                        target_delta=abs(x['selected_target'] - y['selected_target']) if valid else None,
                        feasible_but_selected_disjoint=bool(overlap > 1e-12 and selected_overlap is not None and selected_overlap <= 1e-12),
                        logit_mean_succeeds_both=None)
            if valid:
                w = float(expit((logit(x['selected_target']) + logit(y['selected_target'])) / 2))
                pair['logit_mean_succeeds_both'] = bool(inside(sx, w) and inside(sy, w))
            pairs.append(pair)
        nearest.append(pairs[-len(nn)])

    def summarize(ps):
        both = [p for p in ps if p['both_feasible']]; valid = [p for p in ps if p['selected_disjoint'] is not None]
        return dict(pairs=len(ps), both_feasible_pairs=len(both), selected_valid_pairs=len(valid),
                    success_disjoint_rate=float(np.mean([p['success_disjoint'] for p in both])) if both else None,
                    selected_disjoint_rate=float(np.mean([p['selected_disjoint'] for p in valid])) if valid else None,
                    feasible_but_selected_disjoint_rate=float(np.mean([p['feasible_but_selected_disjoint'] for p in valid])) if valid else None,
                    success_jaccard_mean=float(np.mean([p['jaccard'] for p in both])) if both else None,
                    target_delta_median=float(np.median([p['target_delta'] for p in valid])) if valid else None,
                    logit_mean_succeeds_both_rate=float(np.mean([p['logit_mean_succeeds_both'] for p in valid])) if valid else None,
                    distance_median=float(np.median([p['rms_distance'] for p in ps])))
    q = np.quantile([p['rms_distance'] for p in nearest], [.25, .5, .75]); bands = []
    for lo, hi in zip([-np.inf, *q], [*q, np.inf]):
        bands.append(dict(distance_lower=None if np.isneginf(lo) else float(lo), distance_upper=None if np.isposinf(hi) else float(hi),
                          **summarize([p for p in nearest if lo < p['rms_distance'] <= hi])))
    chosen = []
    for condition in (lambda p: p['success_disjoint'], lambda p: p['feasible_but_selected_disjoint'],
                      lambda p: p['both_feasible'] and not p['success_disjoint'], lambda p: not p['both_feasible']):
        chosen.extend(sorted((p for p in nearest if condition(p)), key=lambda p: p['rms_distance'])[:20])
    chosen = list({(p['i'], p['j']): p for p in chosen}.values())
    return dict(all=summarize(pairs), nearest=summarize(nearest), nearest_distance_quartiles=bands), chosen


def nearest_audit(rows, core, poses, out):
    train = np.array([r['split'] == 'train' for r in rows]); ti = np.flatnonzero(train); vi = np.flatnonzero(~train)
    mean, std = core[train].mean(0), core[train].std(0); std = np.where(std > 1e-6, std, 1.)
    x = (core - mean) / std; distance = cdist(x, x[train]) / np.sqrt(x.shape[1])
    # Train neighbours MUST be from other scenes, not adjacent yaws/positions.
    scenes = np.array([r['scene_id'] for r in rows])
    distance[scenes[:, None] == scenes[ti][None]] = np.inf
    order = np.argsort(distance, axis=1)[:, :50]; ids = ti[order]
    ds = np.take_along_axis(distance, order, axis=1)
    result = {}; examples = []
    for split, ix in [('train_cross_scene', ti), ('val_to_train', vi)]:
        for k in (1, 5, 20):
            # Local rows refer to global IDs; preserve lookup for pair_statistics.
            local_rows = [rows[i] for i in ix] + rows
            local_ids = ids[ix, :k] + len(ix)
            summary, close = pair_statistics(local_rows, local_ids, ds[ix, :k])
            result[f'{split}_k{k}'] = summary
            if k == 1:
                for p in close:
                    p.update(query=rows[ix[p['i']]], neighbour=rows[p['j'] - len(ix)])
                    examples.append(p)
        rng = np.random.default_rng(20261021); random_ids = []
        for i in ix:
            pool = ti[scenes[ti] != scenes[i]]; random_ids.append(rng.choice(pool, 20, replace=False))
        random_ids = np.array(random_ids)
        random_ds = np.linalg.norm(x[ix, None] - x[random_ids], axis=2) / np.sqrt(x.shape[1])
        result[split + '_random20'], _ = pair_statistics([rows[i] for i in ix] + rows, random_ids + len(ix), random_ds)
    # Train-label-only neighbourhood predictors: not GT selection on Val.
    grid = np.linspace(0, 1, 21); outputs = []
    good = np.array([[inside(r['success']['joint'], w) for w in grid] for r in rows])
    for k in (5, 20, 50):
        for rule in ('logit_mean', 'success_vote'):
            ws = []
            for ns in ids[:, :k]:
                valid = [j for j in ns if rows[j]['selected_valid']]
                if rule == 'logit_mean':
                    w = expit(np.mean([logit(rows[j]['selected_target']) for j in valid])) if valid else .1
                else:
                    count = good[ns].mean(0); choices = np.flatnonzero(count == count.max()); w = grid[choices[np.argmin(abs(grid[choices] - .1))]]
                ws.append(float(w))
            predicted = np.array([pose(winner_at(r['parts'], w), np.asarray(r['_xy'])) for r, w in zip(rows, ws)])
            for split, ix in [('train_cross_scene', ti), ('val', vi)]:
                outputs.append(dict(k=k, rule=rule, split=split, **recall(predicted[ix], np.array([r['truth'] for r in rows])[ix], poses[ix, 2])))
    result['neighbour_policies'] = outputs
    result['normalization'] = dict(mean=mean.tolist(), std=std.tolist(), fit_split='train only', rms_l2=True)
    save(out / 'nearest_analysis.json', result)
    for p in examples:
        for name in ('query', 'neighbour'):
            p[name] = {k: v for k, v in p[name].items() if k not in ('_xy', 'parts')}
    save(out / 'nearest_examples.json', examples)
    return result


def score_binary(y, p):
    from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, brier_score_loss
    if len(y) == 0 or len(np.unique(y)) < 2:
        return dict(n=len(y), repair=int(np.sum(y)), harm=int(np.sum(~np.asarray(y, bool))), auc=None, ap=None, balanced_accuracy=None, brier=None)
    return dict(n=len(y), repair=int(np.sum(y)), harm=int(np.sum(~np.asarray(y, bool))), auc=float(roc_auc_score(y, p)),
                ap=float(average_precision_score(y, p)), balanced_accuracy=float(balanced_accuracy_score(y, p >= .5)), brier=float(brier_score_loss(y, p)))


def classifier_task(task):
    """Fixed architecture/hyperparameters, four scene-held-out Train folds.

    Repair vs Harm conditions on a GT outcome change; unchanged queries are NOT
    relabelled as Harm. This is NOT a deployment-level policy evaluation.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits
    out, model_kind, criterion, action_index = task; out = Path(out)
    path = out / 'classifiers' / f'{model_kind}_{criterion}_{action_index}'
    if path.with_suffix('.json').exists() and path.with_suffix('.npz').exists():
        return read(path.with_suffix('.json'))
    with np.load(out / 'analysis_arrays.npz') as b:
        x, train, groups, good, folds = [b[k] for k in ('x', 'train', 'groups', 'good', 'feature_groups')]
    train = train.astype(bool); base = good[:, 2, criterion]; after = good[:, action_index, criterion]
    change = base != after; y = after[change].astype(int); g = groups[change]; xx = x[change]; tr = train[change]
    variants = {'core14': np.arange(14)}
    for family in ('depth', 'support', 'response', 'ray_fit'):
        variants['core14+' + family] = np.r_[np.arange(14), np.flatnonzero(folds == family)]
    variants['core14+all'] = np.arange(x.shape[1])
    result = []; probabilities = {}
    with threadpool_limits(limits=1):
        for name, cols in variants.items():
            def fit_predict(a, b):
                scaler = StandardScaler().fit(xx[a][:, cols]); xa = scaler.transform(xx[a][:, cols]); xb = scaler.transform(xx[b][:, cols])
                if model_kind == 'linear':
                    model = LogisticRegression(C=1., class_weight='balanced', solver='lbfgs', max_iter=2000, random_state=42)
                    model.fit(xa, y[a])
                else:
                    model = HistGradientBoostingClassifier(max_iter=150, learning_rate=.05, max_leaf_nodes=7,
                        min_samples_leaf=30, l2_regularization=1., early_stopping=False, random_state=42)
                    yy = y[a]; counts = np.bincount(yy, minlength=2); weights = len(yy) / (2 * counts[yy])
                    model.fit(xa, yy, sample_weight=weights)
                return model.predict_proba(xb)[:, 1]
            train_ids, val_ids = np.flatnonzero(tr), np.flatnonzero(~tr)
            oof = np.zeros(len(train_ids))
            for fit, check in GroupKFold(4).split(xx[tr], y[tr], g[tr]):
                oof[check] = fit_predict(train_ids[fit], train_ids[check])
            vp = fit_predict(train_ids, val_ids)
            per_scene = {str(sid): score_binary(y[val_ids][g[val_ids] == sid], vp[g[val_ids] == sid]) for sid in np.unique(g[val_ids])}
            result.append(dict(variant=name, input_count=len(cols), model=model_kind, criterion=('xy', 'joint')[criterion],
                action_weight=float(np.linspace(0, 1, 21)[action_index]), train_oof=score_binary(y[train_ids], oof), val=score_binary(y[val_ids], vp), per_scene=per_scene))
            probabilities[name] = vp
    path.parent.mkdir(exist_ok=True)
    np.savez_compressed(path.with_suffix('.npz'), y=y[val_ids], groups=g[val_ids], global_ids=np.flatnonzero(change)[val_ids], **probabilities)
    save(path.with_suffix('.json'), result)
    return result


def distribution(values):
    a = np.asarray(values, float)
    return dict(n=len(a), mean=float(a.mean()), p10=float(np.quantile(a, .1)), median=float(np.median(a)), p90=float(np.quantile(a, .9))) if len(a) else dict(n=0)


def run(args):
    from acoustic_branch.contracts import sha
    root, ssd = args.root, args.ssd; out = root / REL; out.mkdir(parents=True, exist_ok=True)
    source = root / SOURCE; cfg = read(source / 'config.json'); refs = read(source / 'scores_ready.json')['entries']
    if {r['split'] for r in refs} != {'train', 'val'}:
        raise ValueError('Only Train/Val allowed')
    assert len(set(cfg['selection']['train']) & set(cfg['selection']['val'])) == 0
    for ref in refs:
        assert ref['scene_id'] in cfg['selection'][ref['split']]
    config = dict(source_config_sha256=sha(source / 'config.json'), source_scores_sha256=sha(source / 'scores_ready.json'),
        selection={k: cfg['selection'][k] for k in ('train', 'val')}, core_names=CORE_NAMES, normalization='Train only',
        neighbour_reference='Train only, Train query excludes its entire scene', neighbour_k=[1, 5, 20, 50],
        classifier_actions=[.2, .5], criteria=['xy<=1m', 'xy<=1m and yaw<=30deg'],
        classifier_models=['class-balanced logistic C1', 'class-balanced HGB150,lr.05,leaves7,minleaf30,L2=1'],
        classifier_validation='4fold scene-grouped OOF on Train, frozen fit allTrain -> Val; no Val tuning',
        success_set='Continuous affine MAP envelope; excludes zero-width boundary ties; tol1e-11',
        test_accessed=False, feature_gt_free=True,
        caveat='Exploratory same Val3 repeatedly inspected. Features/targets are not proven sufficient or irreducibly ambiguous. No Test files opened.')
    save(out / 'config.json', config)
    tasks = [(str(root), str(ssd), str(out), r) for r in refs]
    progress = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        fs = [pool.submit(extract_scene, task) for task in tasks]
        for f in as_completed(fs):
            info = f.result(); progress.append(info)
            save(out / 'status.json', dict(status='extracting', done=len(progress), total=len(tasks), last_scene=info['scene_id'], test_accessed=False))
            print('extracted', info['scene_id'], info['queries'], flush=True)
    rows = []; cores = []; extras = []; poses = []; truths = []
    for ref in refs:
        p = out / 'scenes' / ref['scene_id']; info = read(p / 'complete.json'); rr = read(p / 'queries.json')
        source_folder = resolve(ref, ssd); xy = np.load(source_folder / 'xy.npy')
        for r in rr:
            r['_xy'] = xy
        with np.load(p / 'arrays.npz') as b:
            cores.extend(b['core']); extras.extend(b['extra']); poses.extend(b['grid_poses']); truths.extend(b['truth'])
        rows.extend(rr)
    core, extra, poses, truth = map(np.array, (cores, extras, poses, truths))
    assert core.shape == (8091, 14)
    train = np.array([r['split'] == 'train' for r in rows]); assert train.sum() == 6387 and (~train).sum() == 1704
    good = np.stack([np.linalg.norm(poses[:, :, :2] - truth[:, None, :2], axis=2) <= 1,
                     (np.linalg.norm(poses[:, :, :2] - truth[:, None, :2], axis=2) <= 1) & (yawdiff(poses[:, :, 2], truth[:, None, 2]) <= 30)], axis=-1)
    save(out / 'feature_definitions.json', dict(core=CORE_NAMES, extra=info['feature_names'], families=info['feature_families'], test_accessed=False))
    np.savez_compressed(out / 'analysis_arrays.npz', x=np.c_[core, extra], train=train, groups=np.array([r['scene_id'] for r in rows]),
                        good=good, feature_groups=np.array(['core'] * 14 + info['feature_families']))
    # Source prediction baseline must match the earlier frozen Val report.
    baseline = recall(poses[~train, 2], truth[~train], poses[~train, 2])
    original = read(source / 'evaluation.json')['methods']['fixed_01']
    for name in ('recall_0.1m', 'recall_0.5m', 'recall_1m', 'recall_1m_30deg'):
        np.testing.assert_allclose(baseline[name], original[name], atol=1e-12, rtol=0)
    save(out / 'status.json', dict(status='nearest_neighbour_analysis', test_accessed=False))
    neighbours = nearest_audit(rows, core, poses, out)
    save(out / 'status.json', dict(status='repair_harm_classifiers', test_accessed=False))
    classifiers = []
    ctasks = [(str(out), m, c, a) for m in ('linear', 'nonlinear') for c in (0, 1) for a in (4, 10)]
    with ProcessPoolExecutor(max_workers=min(args.workers, 8)) as pool:
        for result in pool.map(classifier_task, ctasks):
            classifiers.extend(result)
    sets = {}
    for split in ('train', 'val'):
        rs = [r for r in rows if r['split'] == split]
        sets[split] = dict(queries=len(rs), continuous_joint_feasible=sum(bool(r['success']['joint']) for r in rs),
            successful_union_width=distribution([measure(r['success']['joint']) for r in rs if r['success']['joint']]),
            selected_interval_width=distribution([r['selected_bounds'][1] - r['selected_bounds'][0] for r in rs if r['selected_valid']]),
            success_components=distribution([len(r['success']['joint']) for r in rs if r['success']['joint']]),
            narrow_selected_under001=sum(r['selected_valid'] and r['selected_bounds'][1]-r['selected_bounds'][0] < .01 for r in rs),
            point_targets_for_already_correct=distribution([r['selected_target'] for r in rs if r['selected_valid'] and inside(r['success']['joint'], .1)]),
            point_targets_for_repairable_failure=distribution([r['selected_target'] for r in rs if r['selected_valid'] and not inside(r['success']['joint'], .1)]),
            target_moves_already_correct_over005=sum(r['selected_valid'] and inside(r['success']['joint'], .1) and abs(r['selected_target']-.1) > .05 for r in rs),
            baseline=recall(poses[train if split == 'train' else ~train, 2], truth[train if split == 'train' else ~train], poses[train if split == 'train' else ~train, 2]))
    result = dict(status='complete', config=config, feature_count=core.shape[1]+extra.shape[1], success_sets=sets,
                  neighbours=neighbours, classifiers=classifiers, test_accessed=False,
                  verification=dict(envelope_argmax_checks=sum(i['envelope_argmax_checks'] for i in progress), baseline_reproduced=True,
                                    train_queries=6387, val_queries=1704, train_scenes=12, val_scenes=3))
    save(out / 'results.json', result)
    with (out / 'classifier_metrics.csv').open('w') as f:
        columns = ['model', 'criterion', 'action_weight', 'variant', 'input_count', 'split', 'n', 'repair', 'harm', 'auc', 'ap', 'balanced_accuracy', 'brier']
        writer = csv.DictWriter(f, fieldnames=columns); writer.writeheader()
        for r in classifiers:
            for split in ('train_oof', 'val'):
                writer.writerow({**{k: r[k] for k in columns[:5]}, 'split': split, **r[split]})
    save(out / 'status.json', dict(status='complete', test_accessed=False))
    print('COMPLETE', out / 'results.json', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True); p.add_argument('--ssd', type=Path, required=True)
    p.add_argument('--workers', type=int, default=6); run(p.parse_args())

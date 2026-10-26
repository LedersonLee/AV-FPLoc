"""Existing core14 inputs: model reliance versus outcome-label usefulness.

Train12/Val3 ONLY. Never opens Test files. No encoder update or new fusion-gate
training. Last-gate permutation tests measure reliance, NOT causal usefulness.
Scene-held-out classifier drop/only refits measure conditional/marginal outcome
information under explicitly fixed Repair/Harm labels, not scalar-weight loss.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import shutil

import numpy as np
from scipy.special import logit
from scipy.stats import rankdata, spearmanr

from .input_supervision_audit import (
    REL as INPUT_SOURCE, SOURCE as GATE_SOURCE, CORE_NAMES, read, save, recall,
    resolve, score_binary, winner_at, pose,
)

REL = 'ReliabilityFusion/pilots/core14_importance_v1'
GROUPS = {'vision': list(range(5)), 'acoustic': list(range(5, 10)), 'interaction': list(range(10, 14))}


def variants():
    result = {'full14': list(range(14))}
    for i, name in enumerate(CORE_NAMES):
        result['drop:' + name] = [j for j in range(14) if i != j]
        result['only:' + name] = [i]
    for group, ids in GROUPS.items():
        result['drop_group:' + group] = [j for j in range(14) if j not in ids]
        result['only_group:' + group] = ids
    return result


def within_scene_permutation(groups, seed):
    rng = np.random.default_rng(seed); out = np.arange(len(groups))
    for group in np.unique(groups):
        ids = np.flatnonzero(groups == group); out[ids] = rng.permutation(ids)
    return out


def auc_columns(y, p):
    n1 = int(y.sum()); n0 = len(y)-n1
    if not n1 or not n0:
        return None
    ranks = rankdata(p, axis=0)
    return (ranks[y.astype(bool)].sum(0)-n1*(n1+1)/2)/(n1*n0)


def paired_ci(y, predictions, scenes, physical, repeats=1000):
    groups = []
    for scene in np.unique(scenes):
        ids = np.flatnonzero(scenes == scene)
        groups.append([ids[physical[ids] == p] for p in np.unique(physical[ids])])
    rng = np.random.default_rng(20261021); samples = []
    for _ in range(repeats):
        take = np.concatenate([np.concatenate([g[i] for i in rng.integers(len(g), size=len(g))]) for g in groups])
        values = auc_columns(y[take], predictions[take])
        if values is not None:
            samples.append(values-values[0])
    return np.quantile(samples, [.025, .975], axis=0).T


def classifier(task):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits
    root, model_kind, criterion, action, repeats = task; root = Path(root)
    out = root/REL; src = root/INPUT_SOURCE; stem = f'{model_kind}_{criterion}_{action}'
    dest = out/'classifiers'/stem; dest.parent.mkdir(exist_ok=True)
    if dest.with_suffix('.json').exists():
        return read(dest.with_suffix('.json'))
    with np.load(src/'analysis_arrays.npz') as b:
        x = b['x'][:, :14].copy(); train = b['train'].astype(bool); groups = b['groups'].copy(); good = b['good'].copy()
    metadata = read(out/'query_index.json')
    physical = np.array([str(r['pose_id']) for r in metadata])
    before, after = good[:, 2, criterion], good[:, action, criterion]
    changed = before != after; global_ids = np.flatnonzero(changed)
    x, y, tr, groups, physical = x[changed], after[changed].astype(int), train[changed], groups[changed], physical[changed]
    train_ids, val_ids = np.flatnonzero(tr), np.flatnonzero(~tr)
    result = []; val_probs = []; train_probs = []; specs = variants()
    with threadpool_limits(limits=1):
        for name, cols in specs.items():
            def fit(a, b):
                scaler = StandardScaler().fit(x[a][:, cols]); xa = scaler.transform(x[a][:, cols]); xb = scaler.transform(x[b][:, cols])
                if model_kind == 'linear':
                    m = LogisticRegression(C=1., class_weight='balanced', solver='lbfgs', max_iter=2000, random_state=42)
                    m.fit(xa, y[a])
                else:
                    m = HistGradientBoostingClassifier(max_iter=150, learning_rate=.05, max_leaf_nodes=7,
                        min_samples_leaf=30, l2_regularization=1., early_stopping=False, random_state=42)
                    counts = np.bincount(y[a], minlength=2)
                    m.fit(xa, y[a], sample_weight=len(a)/(2*counts[y[a]]))
                return m.predict_proba(xb)[:, 1]
            oof = np.zeros(len(train_ids))
            for a, b in GroupKFold(4).split(x[tr], y[tr], groups[tr]):
                oof[b] = fit(train_ids[a], train_ids[b])
            vp = fit(train_ids, val_ids)
            if name == 'full14':
                with np.load(src/'classifiers'/f'{model_kind}_{criterion}_{action}.npz') as previous:
                    np.testing.assert_array_equal(previous['global_ids'], global_ids[val_ids])
                    np.testing.assert_allclose(previous['core14'], vp, atol=1e-10, rtol=0)
            result.append(dict(variant=name, columns=cols, input_count=len(cols), train_oof=score_binary(y[train_ids], oof),
                val=score_binary(y[val_ids], vp), per_scene={sid: score_binary(y[val_ids][groups[val_ids] == sid], vp[groups[val_ids] == sid]) for sid in np.unique(groups[val_ids])}))
            val_probs.append(vp); train_probs.append(oof)
    vp = np.column_stack(val_probs); intervals = paired_ci(y[val_ids], vp, groups[val_ids], physical[val_ids], repeats)
    for i, row in enumerate(result):
        row.update(val_delta_auc=row['val']['auc']-result[0]['val']['auc'], val_delta_ci95=intervals[i].tolist(),
                   train_oof_delta_auc=row['train_oof']['auc']-result[0]['train_oof']['auc'])
    np.savez_compressed(dest.with_suffix('.npz'), val_probs=vp, train_probs=np.column_stack(train_probs),
                        val_y=y[val_ids], train_y=y[train_ids], global_val_ids=global_ids[val_ids], names=np.array(list(specs)))
    value = dict(model=model_kind, criterion=('xy', 'joint')[criterion], action_weight=action*.05,
                 variants=result, baseline_reproduced=True, test_accessed=False)
    save(dest.with_suffix('.json'), value)
    return value


def gate_permutations(root, ssd, repeats):
    import torch
    from .model import Gate
    from acoustic_branch.contracts import sha
    torch.set_num_threads(1)
    out = root/REL; src = root/INPUT_SOURCE; old = root/GATE_SOURCE
    if (out/'gate_permutations.json').exists():
        return read(out/'gate_permutations.json')
    metadata = read(out/'query_index.json'); refs = read(old/'scores_ready.json')['entries']
    xy = {r['scene_id']: np.load(resolve(r, ssd)/'xy.npy') for r in refs}
    rows = [r for ref in refs for r in read(src/'scenes'/ref['scene_id']/'queries.json')]
    with np.load(src/'analysis_arrays.npz') as b:
        x = b['x'][:, :14].copy(); train = b['train'].astype(bool); good = b['good'][:, :, 1].copy()
    groups = np.array([r['scene_id'] for r in rows]); gt = np.array([r['truth'] for r in rows])
    fixed = np.array([pose(winner_at(r['parts'], .1), xy[r['scene_id']]) for r in rows])
    labels = np.array([logit(r['selected_target']) if r['selected_valid'] else 0. for r in rows])
    valid = np.array([r['selected_valid'] for r in rows]); grid = np.linspace(0, 1, 21)
    # set_loss was supervised only where the existing selected_valid mask holds.
    features = {name: [i] for i, name in enumerate(CORE_NAMES)}
    features.update({'group:' + k: v for k, v in GROUPS.items()})
    evaluated = []
    report = read(old/'evaluation.json')
    for kind in ('logit_regression', 'success_set'):
        for seed in (42, 43):
            path = old/'runs'/f'{kind}_seed{seed}'/'last.pt'; state = torch.load(path, map_location='cpu', weights_only=False)
            model = Gate(14).eval(); model.load_state_dict(state['model'])
            mean, std = np.asarray(state['mean']), np.asarray(state['std'])

            def evaluate(xx):
                with torch.inference_mode():
                    raw = model.net(torch.tensor((xx-mean)/std, dtype=torch.float32)).squeeze(-1).numpy()
                w = 1/(1+np.exp(-raw))
                pp = np.array([pose(winner_at(r['parts'], float(ww)), xy[r['scene_id']]) for r, ww in zip(rows, w)])
                if kind == 'logit_regression':
                    loss = (raw-labels)**2
                else:
                    loss = np.min(np.where(good, (w[:, None]-grid)**2, np.inf), axis=1)
                output = {}
                for split, mask in [('train', train), ('val', ~train)]:
                    output[split] = recall(pp[mask], gt[mask], fixed[mask])
                    output[split]['loss'] = float(loss[mask & valid].mean())
                return output

            baseline = evaluate(x)
            previous = report['methods'][f'{kind}_seed{seed}_last']
            for name in ('recall_0.1m', 'recall_0.5m', 'recall_1m', 'recall_1m_30deg'):
                np.testing.assert_allclose(baseline['val'][name], previous[name], atol=1e-12, rtol=0)
            measures = []
            for name, cols in features.items():
                scores = []
                for rep in range(repeats):
                    order = within_scene_permutation(groups, 1000+rep)
                    perturbed = x.copy(); perturbed[:, cols] = x[order][:, cols]
                    scores.append(evaluate(perturbed))
                metrics = {}
                for split in ('train', 'val'):
                    metrics[split] = {}
                    for key in baseline[split]:
                        vals = np.array([s[split][key]-baseline[split][key] for s in scores])
                        metrics[split][key] = dict(delta_mean=float(vals.mean()), shuffle_sd=float(vals.std()),
                                                  shuffle_p025_p975=np.quantile(vals, [.025, .975]).tolist())
                measures.append(dict(feature=name, columns=cols, metrics=metrics))
            evaluated.append(dict(kind=kind, seed=seed, epoch=state['epoch'], checkpoint_sha256=sha(path), baseline=baseline, features=measures))
            save(out/'gate_progress.json', dict(done=len(evaluated), total=4, last_kind=kind, seed=seed, test_accessed=False))
    value = dict(repeats=repeats, shuffle='Within each scene, same permutation jointly for all columns in a group; feature correlation can be broken',
                 interpretation='Sensitivity of a trained underperforming Last model, NOT intrinsic feature usefulness. Best epoch0 constant gate has zero importance by construction.',
                 runs=evaluated, test_accessed=False)
    save(out/'gate_permutations.json', value)
    return value


def run(args):
    from acoustic_branch.contracts import sha
    out = args.root/REL; out.mkdir(parents=True, exist_ok=True)
    source = args.root/INPUT_SOURCE; data = np.load(source/'analysis_arrays.npz')
    refs = read(args.root/GATE_SOURCE/'scores_ready.json')['entries']
    assert {r['split'] for r in refs} == {'train', 'val'}
    rows = [r for ref in refs for r in read(source/'scenes'/ref['scene_id']/'queries.json')]
    assert len(rows) == 8091 and sum(r['split'] == 'train' for r in rows) == 6387
    save(out/'query_index.json', [{k: r[k] for k in ('scene_id', 'split', 'pose_id', 'sample_index')} for r in rows])
    cfg = dict(source_arrays_sha256=sha(source/'analysis_arrays.npz'), source_config_sha256=sha(source/'config.json'),
        features=CORE_NAMES, groups=GROUPS, variants=variants(), train_scenes=12, val_scenes=3, train_queries=6387, val_queries=1704,
        input_scalers='Train only; refit within each grouped Train fold', classifier_validation='4fold scene-disjoint Train OOF; fixed allTrain fit -> Val',
        diagnostic_labels='Repair vs Harm relative fixedw.1 among changed outcomes, controlled actions .2,.5; unchanged excluded only from diagnostic classifiers',
        class_weights='balanced from Train labels only', model_settings='Same as input_supervision_audit_v1; logistic C1 and HGB150/leaves7/leaf30/lr.05/L2=1',
        gate_models='Existing Last40, two losses x seeds42,43; no gate training', gate_permutation_repeats=args.repeats,
        bootstrap_repeats=args.bootstrap, test_accessed=False,
        caveat='Exploratory development Val3; dependence/redundancy and target choice affect importance. No Val-driven subset adopted or fusion policy selected.')
    save(out/'config.json', cfg)
    train = data['train'].astype(bool); x = data['x'][:, :14]
    corr = spearmanr(x[train]).correlation
    save(out/'correlations.json', dict(names=CORE_NAMES, train_spearman=corr.tolist(),
        strong_pairs=[dict(first=CORE_NAMES[i], second=CORE_NAMES[j], rho=float(corr[i, j])) for i in range(14) for j in range(i) if abs(corr[i,j]) >= .8]))
    save(out/'status.json', dict(status='classifier_refits', test_accessed=False))
    result = []
    tasks = [(str(args.root), m, c, a, args.bootstrap) for m in ('linear', 'nonlinear') for c in (0, 1) for a in (4, 10)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        fs = [pool.submit(classifier, t) for t in tasks]
        for f in as_completed(fs):
            r = f.result(); result.append(r)
            save(out/'status.json', dict(status='classifier_refits', done=len(result), total=8, test_accessed=False))
            print('classifier', r['model'], r['criterion'], r['action_weight'], flush=True)
    save(out/'status.json', dict(status='existing_gate_permutations', test_accessed=False))
    gates = gate_permutations(args.root, args.ssd, args.repeats)
    result = sorted(result, key=lambda r: (r['model'], r['criterion'], r['action_weight']))
    final = dict(status='complete', config=cfg, classifiers=result, gate_permutations=gates,
                 correlations=read(out/'correlations.json'), test_accessed=False,
                 verification=dict(classifier_full14_baselines_reproduced=8, gate_last_baselines_reproduced=4, train_queries=6387, val_queries=1704))
    save(out/'results.json', final)
    save(out/'status.json', dict(status='complete', test_accessed=False))
    print('COMPLETE', out/'results.json', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True); p.add_argument('--ssd', type=Path, required=True)
    p.add_argument('--workers', type=int, default=8); p.add_argument('--repeats', type=int, default=20); p.add_argument('--bootstrap', type=int, default=1000)
    run(p.parse_args())

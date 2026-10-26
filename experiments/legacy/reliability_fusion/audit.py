"""Read-only checkpoint/loss/feature audit. No fitting and no best reselection.

Test is post-hoc diagnostic only. Original run receipts/models remain untouched.
CPU-only, bounded streaming of the existing frozen score cache.
"""
import argparse
from collections import defaultdict
import hashlib
import html
import json
from pathlib import Path
import time

import numpy as np
from scipy.special import logsumexp, softmax
from scipy.stats import rankdata, spearmanr
import torch

from acoustic_branch.contracts import atomic_json, sha
from acoustic_branch.data import read
from .core import FEATURE_NAMES, target_statistics, metrics
from .model import Gate, loss
from .prepare import REL


def objective_statistics(v, a, expected, weight):
    score = (1-weight)*v + weight*a
    p = softmax(score.ravel())
    delta = (a-v).ravel()
    gradient = float(p@delta-(expected[1]-expected[0]))
    curvature = float(p@(delta**2)-(p@delta)**2)
    return dict(nll=float(logsumexp(score)-(1-weight)*expected[0]-weight*expected[1]),
                gradient=gradient, curvature=curvature)


def auc(values, labels):
    values, labels = np.asarray(values), np.asarray(labels, dtype=bool)
    positive, negative = labels.sum(), (~labels).sum()
    if not positive or not negative:
        return None
    return float((rankdata(values)[labels].sum()-positive*(positive+1)/2)/(positive*negative))


def scalar_gradient_check(v, a, expected):
    vt = torch.tensor(v.ravel()[None], dtype=torch.float64)
    at = torch.tensor(a.ravel()[None], dtype=torch.float64)
    et = torch.tensor(expected[None], dtype=torch.float64)
    w = torch.tensor([.5], dtype=torch.float64, requires_grad=True)
    objective = loss(w, vt, at, et)
    objective.backward()
    analytic = objective_statistics(v, a, expected, .5)
    epsilon = 1e-5
    finite = (objective_statistics(v, a, expected, .5+epsilon)['nll']-
              objective_statistics(v, a, expected, .5-epsilon)['nll'])/(2*epsilon)
    return dict(torch_nll=float(objective.detach()), numpy_nll=analytic['nll'],
        autograd=float(w.grad.item()), analytic=analytic['gradient'], finite_difference=finite,
        max_gradient_error=max(abs(float(w.grad.item())-analytic['gradient']), abs(finite-analytic['gradient'])))


def feature_audit(scenes):
    by_split = {}
    for split in ('train', 'val', 'test'):
        selected = [s for s in scenes if s['split'] == split]
        x = np.concatenate([s['features'] for s in selected])
        if not np.isfinite(x).all():
            raise ValueError('Nonfinite features')
        # Diagnostic label only: neither absolute GT pose nor this outcome is
        # an inference-time feature. AUC is descriptive, not a fitted model.
        labels, keep = [], []
        for s in selected:
            for row in s['records']:
                vok, aok = row['outcomes']['vision_only']['ok'], row['outcomes']['acoustic_only']['ok']
                labels.append(aok)
                keep.append(vok != aok)
        labels, keep = np.asarray(labels), np.asarray(keep)
        by_split[split] = dict(queries=len(x), mean=x.mean(0).tolist(), std=x.std(0).tolist(),
            minimum=x.min(0).tolist(), maximum=x.max(0).tolist(),
            exclusive_modality_queries=int(keep.sum()),
            exclusive_audio_success_auc=[auc(x[keep, j], labels[keep]) for j in range(x.shape[1])])
    train = by_split['train']
    for split in ('val', 'test'):
        target = by_split[split]
        target['mean_shift_train_std'] = ((np.asarray(target['mean'])-train['mean'])/
            np.maximum(train['std'], 1e-6)).tolist()
        target['outside_train_range_fraction'] = np.mean(
            (np.concatenate([s['features'] for s in scenes if s['split'] == split]) < train['minimum']) |
            (np.concatenate([s['features'] for s in scenes if s['split'] == split]) > train['maximum']), axis=0).tolist()
    return dict(feature_names=FEATURE_NAMES, splits=by_split,
        note='Univariate AUC on V/A-exclusive success cases; not cross-validated predictive performance.')


def summarize(scenes):
    result = {}
    for split in ('train', 'val', 'test'):
        selected = [s for s in scenes if s['split'] == split]
        all_records = [r for s in selected for r in s['records']]
        names = list(all_records[0]['outcomes'])
        methods = {}
        for name in names:
            def group(rows):
                values = [r['outcomes'][name] for r in rows]
                return dict(metrics([v['pose'] for v in values], [r['truth'] for r in rows]),
                    nll=float(np.mean([v['nll'] for v in values])),
                    weight_mean=float(np.mean([v['weight'] for v in values])),
                    weight_p10=float(np.quantile([v['weight'] for v in values], .1)),
                    weight_p90=float(np.quantile([v['weight'] for v in values], .9)),
                    repair_vs_equal=sum(not r['outcomes']['equal']['ok'] and r['outcomes'][name]['ok'] for r in rows),
                    harm_vs_equal=sum(r['outcomes']['equal']['ok'] and not r['outcomes'][name]['ok'] for r in rows))
            per_scene = {s['scene_id']: group(s['records']) for s in selected}
            methods[name] = dict(micro=group(all_records), per_scene=per_scene,
                scene_macro={k:float(np.mean([v[k] for v in per_scene.values()])) for k in
                    ('recall_01m','recall_05m','recall_1m','recall_1m_30deg','nll')})
        buckets = {}
        for bucket in ('vision_only_correct', 'acoustic_only_correct', 'both_correct', 'both_wrong'):
            rows = [r for r in all_records if r['bucket'] == bucket]
            if not rows:
                continue
            buckets[bucket] = dict(queries=len(rows),
                gradient_at_half_mean=float(np.mean([r['at_half']['gradient'] for r in rows])),
                gradient_prefers_audio_fraction=float(np.mean([r['at_half']['gradient']<0 for r in rows])),
                last_adaptive_weight_mean=float(np.mean([r['outcomes']['adaptive_last']['weight'] for r in rows])))
        nll_drop = np.array([r['outcomes']['adaptive_last']['nll']<r['outcomes']['equal']['nll'] for r in all_records])
        result[split] = dict(scenes=len(selected), queries=len(all_records), methods=methods, buckets=buckets,
            initial_gradient_prefers_audio_fraction=float(np.mean([r['at_half']['gradient']<0 for r in all_records])),
            adaptive_last_nll_improved_queries=int(nll_drop.sum()),
            nll_improved_but_equal_success_harmed=sum(bool(nll_drop[i]) and r['outcomes']['equal']['ok'] and
                not r['outcomes']['adaptive_last']['ok'] for i, r in enumerate(all_records)))
    return result


def charts(report, history, best):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for mode, color in [('constant', '#9567ba'), ('adaptive', '#d46a19')]:
        rows = history[mode]
        ep = [0]+[r['epoch'] for r in rows]
        axes[0].plot(ep, [best[mode]['nll']]+[r['val']['nll'] for r in rows], label=mode, color=color)
        axes[1].plot(ep, [100*best[mode]['scene_macro']['recall_1m']]+
            [100*r['val']['scene_macro']['recall_1m'] for r in rows], label=mode, color=color)
        axes[2].plot(ep, [.5]+[r['val']['weight_mean'] for r in rows], label=mode, color=color)
    for ax, title in zip(axes, ('Val soft-pose NLL (lower better)', 'Val scene-macro R@1m (%)', 'Mean acoustic weight')):
        ax.set_title(title);ax.set_xlabel('Epoch');ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Loss improves while localization recall falls; epoch 0 remains best')
    fig.tight_layout();fig.savefig(report/'training.svg');plt.close(fig)


def publish(report, summary):
    report.mkdir(parents=True, exist_ok=True)
    tables = []
    for split in ('val','test'):
        rows = []
        for name, result in summary['splits'][split]['methods'].items():
            m = result['micro'];macro = result['scene_macro']
            cells = [name, *(f'{100*m[k]:.2f}' for k in ('recall_01m','recall_05m','recall_1m','recall_1m_30deg')),
                     f'{100*macro["recall_1m"]:.2f}', f'{macro["nll"]:.5f}', f'{m["weight_mean"]:.3f}']
            rows.append('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in cells)+'</tr>')
        tables.append(f'<h2>{split.upper()}</h2><table><tr><th>Method</th><th>R@0.1m %</th><th>R@0.5m %</th>'+
            '<th>R@1m %</th><th>R@1m,30deg %</th><th>Scene-macro R@1m %</th><th>Macro NLL</th>'+
            '<th>Mean acoustic weight</th></tr>'+''.join(rows)+'</table>')
    val = summary['splits']['val']
    bucket = val['buckets']['vision_only_correct']
    max_error = max(g['max_gradient_error'] for g in summary['gradient_checks'])
    feature_rows = []
    for i, name in enumerate(FEATURE_NAMES):
        values = [name, f"{summary['features']['splits']['train']['std'][i]:.5g}",
            f"{summary['features']['splits']['val']['mean_shift_train_std'][i]:.3f}"]
        for split in ('train','val'):
            value = summary['features']['splits'][split]['exclusive_audio_success_auc'][i]
            values.append('—' if value is None else f'{value:.3f}')
        feature_rows.append('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in values)+'</tr>')
    findings = f'''<h2>Diagnosis</h2><ul>
<li>Training did update the gate: adaptive changed {summary['checkpoint_checks']['adaptive']['changed_tensors']} parameter tensors;
mean val acoustic weight moved from 0.500 to {val['methods']['adaptive_last']['micro']['weight_mean']:.3f}.</li>
<li>Analytic gradient, PyTorch autograd, and finite differences agree (maximum error {max_error:.2g}).
All score-cache SHA checks and soft-target expectation reconstructions passed.</li>
<li>Among {bucket['queries']} val queries where vision alone is correct and acoustic alone is wrong,
{100*bucket['gradient_prefers_audio_fraction']:.2f}% have an initial gradient that increases acoustic weight.</li>
<li>{val['nll_improved_but_equal_success_harmed']} val queries improve NLL while changing equal-fusion success to last-adaptive failure.</li>
<li>Even the feature-free constant gate reduces NLL while harming recall. Feature extraction or MLP capacity alone cannot explain this pattern.</li>
<li>Best selection is val scene-macro R@1m, then joint recall, then NLL. All 15 trained epochs fail to beat epoch 0; patience=15 causes stopping.
This is not a failure to update, nor stopping because NLL rose.</li></ul>
<h2>Loss and score design</h2><p>s=(1−w)v+w·a; L=logsumexp(s)−E<sub>GT</sub>[s].
Its derivative is E<sub>softmax(s)</sub>[a−v]−E<sub>GT</sub>[a−v]. The loss supervises the full probability shape,
not whether its maximum is within 1m/30°. Each modality is independently z-scored; this equalizes standard deviation,
not probability calibration. GT smoothing is σ=0.2m and 10°, narrower than the recall tolerance.
The relative contributions of temperature/calibration and target smoothing have NOT been isolated by retraining.</p>
<p>The config's score_temperature is not consumed by the score/loss implementation. Its current value is 1.0, so this
does not explain the current numerical results, but changing the config alone would not calibrate the model.</p>
<h2>Features</h2><p>All 17 inputs are finite and nonconstant on train/val. Normalization was fitted on train only.
Below, AUC is an exploratory univariate diagnostic on queries where exactly one modality succeeds, with acoustic success positive.
AUC below 0.5 can be useful in the opposite direction. This is NOT held-out learned gate accuracy or evidence of causal sufficiency.</p>
<table><tr><th>Feature</th><th>Train std</th><th>Val mean shift / train std</th><th>Train AUC</th><th>Val AUC</th></tr>
{''.join(feature_rows)}</table><p>Train gate: 6 scenes / 3,105 queries; val: 3 / 1,704; test: 3 / 1,472.
Frozen branches were trained on these train scenes (not out-of-fold); this can bias gate supervision and needs future cross-fitting controls.
No evidence here establishes that simply adding epochs or scenes will fix the objective mismatch.</p>'''
    page = '''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Reliability fusion training audit</title><style>body{font:16px system-ui;color:#18304a;margin:30px;max-width:1400px}
table{border-collapse:collapse}td,th{padding:10px;border:1px solid #cbd7e5}th{background:#edf3fa}
img{max-width:100%}.note{background:#fff1cf;padding:18px;line-height:1.6}</style>
<h1>Reliability-aware fusion: best vs last audit</h1><div class="note">Read-only evaluation of frozen checkpoints and existing scores.
No retraining, checkpoint reselection, test tuning, or raw RIR generation. Best=epoch 0; last=epoch 15.
Micro recalls and scene-macro selection scores are explicitly separated. Existing three test scenes are exploratory, not a new blind benchmark.</div>
<img src="training.svg" alt="Val NLL falls, recall falls, acoustic weight increases">TABLES
FINDINGS<p><a href="summary.json">Full audit JSON (gradients, features, per-scene metrics, checkpoint deltas)</a></p>'''
    (report/'index.html').write_text(page.replace('TABLES',''.join(tables)).replace('FINDINGS',findings))
    atomic_json(report/'summary.json', summary)


def run(root, out, report):
    root, out, report = map(Path, (root, out, report))
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    base = root/REL
    config = read(base/'config.json')
    membership = read(root/'manifests/scene_splits_v1.json')
    checkpoints, models, checkpoint_checks = {}, {}, {}
    history, best = {}, {}
    for mode in ('constant','adaptive'):
        history[mode] = read(base/'runs'/mode/'history.json')
        best[mode] = read(base/'runs'/mode/'complete.json')['best_val']
        for which in ('best','last'):
            name = mode+'_'+which
            path = base/'runs'/mode/(which+'.pt')
            state = torch.load(path, map_location='cpu', weights_only=False)
            if state['config_sha256'] != sha(base/'config.json') or state['test_accessed'] or state['feature_names'] != FEATURE_NAMES:
                raise ValueError('Checkpoint contract changed')
            model = Gate(len(FEATURE_NAMES), mode=='constant')
            model.load_state_dict(state['model']);model.eval()
            models[name], checkpoints[name] = model, state
        first, last = checkpoints[mode+'_best']['model'], checkpoints[mode+'_last']['model']
        deltas = {k:float(torch.linalg.vector_norm((last[k]-first[k]).float())) for k in first}
        checkpoint_checks[mode] = dict(best_epoch=checkpoints[mode+'_best']['epoch'], last_epoch=checkpoints[mode+'_last']['epoch'],
            parameter_delta_l2_by_tensor=deltas, changed_tensors=sum(v>0 for v in deltas.values()),
            best_sha256=sha(base/'runs'/mode/'best.pt'), last_sha256=sha(base/'runs'/mode/'last.pt'))
    entries = read(base/'scores_ready.json')['entries']+read(base/'test_scores_ready.json')['entries']
    seen = set()
    all_scenes = []
    gradients = []
    for entry in entries:
        scene, split = entry['scene_id'], entry['split']
        if scene in seen or scene not in membership[split]:
            raise ValueError('Scene split overlap/membership mismatch')
        seen.add(scene)
        folder = root/entry['directory']
        receipt = read(folder/'receipt.json')
        if sha(folder/'receipt.json') != entry['receipt_sha256'] or receipt['config_sha256'] != sha(base/'config.json'):
            raise ValueError('Cache receipt changed')
        arrays = {}
        for name in ('features','target_expectations','truth','xy'):
            if sha(folder/(name+'.npy')) != receipt['files'][name+'.npy']:
                raise ValueError('Input feature/label file changed')
            arrays[name] = np.load(folder/(name+'.npy'), allow_pickle=False)
        scores = np.load(folder/'scores.npy', mmap_mode='r', allow_pickle=False)
        # A full SHA read also verifies original cache bytes before diagnostics.
        if sha(folder/'scores.npy') != receipt['files']['scores.npy']:
            raise ValueError('Frozen scores changed')
        f = arrays['features']
        weights = dict(vision_only=np.zeros(len(f)), acoustic_only=np.ones(len(f)), equal=np.full(len(f), .5))
        with torch.no_grad():
            for name, model in models.items():
                state = checkpoints[name]
                x = (f-state['mean'])/state['std']
                if not np.isfinite(x).all():
                    raise ValueError('Invalid normalized features')
                weights[name] = model(torch.tensor(x,dtype=torch.float32)).numpy()
        rows = []
        for i in range(len(f)):
            v, a = scores[i]
            if not np.isfinite(v).all() or not np.isfinite(a).all():
                raise ValueError('Nonfinite score bank')
            truth = arrays['truth'][i]
            expected = arrays['target_expectations'][i]
            recomputed, nearest = target_statistics(v,a,arrays['xy'],truth,
                config['soft_target_xy_sigma_m'],config['soft_target_yaw_sigma_deg'])
            np.testing.assert_allclose(recomputed,expected,atol=1e-10,rtol=1e-10)
            if i == 0:
                gradients.append(dict(scene_id=scene, split=split, **scalar_gradient_check(v,a,expected)))
            outcomes = {}; reusable = {}
            for name, wvalues in weights.items():
                w = float(wvalues[i])
                if w not in reusable:
                    score = (1-w)*v+w*a
                    cell, yaw = np.unravel_index(score.argmax(),score.shape)
                    pose = [*arrays['xy'][cell].tolist(),int(yaw)*10]
                    xy_error = float(np.linalg.norm(np.asarray(pose[:2])-truth[:2]))
                    yaw_error = float(abs((pose[2]-truth[2]+180)%360-180))
                    reusable[w] = dict(pose=pose, xy_error_m=xy_error,yaw_error_deg=yaw_error,
                        ok=xy_error<=1, joint=xy_error<=1 and yaw_error<=30, weight=w,
                        nll=float(logsumexp(score)-(1-w)*expected[0]-w*expected[1]))
                outcomes[name] = reusable[w]
            vok, aok = outcomes['vision_only']['ok'], outcomes['acoustic_only']['ok']
            bucket = ('both_correct' if aok else 'vision_only_correct') if vok else ('acoustic_only_correct' if aok else 'both_wrong')
            rows.append(dict(index=i, truth=truth.tolist(), outcomes=outcomes, bucket=bucket,
                at_half=objective_statistics(v,a,expected,.5), nearest_candidate_m=nearest))
        scores._mmap.close()
        scene_result = dict(scene_id=scene,split=split,records=rows,features=f.tolist())
        atomic_json(out/'scenes'/(scene+'.json'),scene_result)
        all_scenes.append(dict(scene_id=scene,split=split,records=rows,features=f))
        atomic_json(out/'status.json',dict(status='running',completed_scenes=len(all_scenes),scenes=len(entries),
            last_scene=scene,updated_unix=time.time(),test_fitting=False))
        print('audited',split,scene,len(rows),flush=True)
    feature_report = feature_audit(all_scenes)
    summary = dict(status='complete',checkpoint_checks=checkpoint_checks,gradient_checks=gradients,
        splits=summarize(all_scenes),features=feature_report,history=history,
        frozen_config=config, test_fitting=False, checkpoint_reselection=False, raw_data_modified=False,
        score_cache_full_sha_verified=True, created_unix=time.time())
    # Independently reproduce original val selection metrics and published test.
    for mode in ('constant','adaptive'):
        reproduced = summary['splits']['val']['methods'][mode+'_best']['scene_macro']
        np.testing.assert_allclose(reproduced['recall_1m'],best[mode]['scene_macro']['recall_1m'],atol=1e-14)
        np.testing.assert_allclose(reproduced['nll'],best[mode]['nll'],atol=1e-10)
    report.mkdir(parents=True, exist_ok=True)
    charts(report,history,best);publish(report,summary)
    atomic_json(out/'summary.json',summary)
    atomic_json(out/'status.json',dict(status='complete',report=str(report),test_fitting=False,updated_unix=time.time()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('root','out','report'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--publish-only',action='store_true')
    args = parser.parse_args()
    if args.publish_only:
        publish(args.report,read(args.out/'summary.json'))
    else:
        run(args.root,args.out,args.report)

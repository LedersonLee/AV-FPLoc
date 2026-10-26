"""Read-only Train12/Val3 diagnosis of the frozen 128/64 capacity runs.

No fitting, target changes or Test access. Predictions were already checked
against canonical dense argmax by capacity_report; we recheck envelope decoding,
success-set membership, target loss and paired Recall identities here.
"""
import argparse
import csv
from pathlib import Path
import shutil

import numpy as np

from acoustic_branch.contracts import atomic_json, sha
from .capacity_probe import REL as CAPACITY
from .input_supervision_audit import read, yawdiff
from .response_input_localization import REL as DATA, load_scene, metrics, predict

REL = 'ReliabilityFusion/analysis/checkpoint_drift_128_v1'


def logit(w):
    w = np.clip(np.asarray(w, dtype=float), 1e-12, 1-1e-12)
    return np.log(w)-np.log1p(-w)


def merged(intervals):
    """Merge adjacent successful affine winners into connected success sets."""
    out = []
    for lo, hi in sorted(intervals):
        if out and lo <= out[-1][1]+1e-12:
            out[-1][1] = max(hi, out[-1][1])
        else:
            out.append([float(lo), float(hi)])
    return out


def component(intervals, w):
    return next((j for j, (lo, hi) in enumerate(intervals) if lo <= w <= hi), None)


def distance(intervals, w):
    return min([max(lo-w, w-hi, 0.) for lo, hi in intervals] or [np.nan])


def status(p, t):
    d = float(np.linalg.norm(np.asarray(p)[:2]-np.asarray(t)[:2]))
    a = float(yawdiff(p[2], t[2]))
    return dict(xy_m=d, yaw_deg=a, xy=d <= 1, joint=d <= 1 and a <= 30)


def transition(b, l):
    return 'stay_success' if b and l else 'harm' if b else 'repair' if l else 'stay_failure'


def diagnose(intervals, wb, wl, target, valid):
    """GT is used only for this post-hoc diagnosis, never as a model input."""
    intervals = merged(intervals)
    j = component(intervals, wb)
    target_j = component(intervals, target) if valid else None
    out = dict(success_components=intervals,
               best_component=j, target_component=target_j,
               best_safe_margin=None if j is None else min(wb-intervals[j][0], intervals[j][1]-wb),
               last_distance_to_success=distance(intervals, wl) if intervals else None,
               exit_direction='not_applicable', exit_boundary=None, overshoot=None,
               target_relation='not_supervised' if not valid else 'best_not_success' if j is None
               else 'same_component' if j == target_j else 'different_component',
               residual_relation=None)
    if j is not None and component(intervals, wl) is None:
        lo, hi = intervals[j]
        edge = hi if wl > hi else lo
        out.update(exit_direction='higher_w' if wl > hi else 'lower_w',
                   exit_boundary=edge, overshoot=abs(wl-edge))
    if valid:
        # In logit space, moving past the scalar target and continuing to the
        # opposite side is distinguishable from merely approaching it.
        out['residual_relation'] = 'crossed_target' if (wb-target)*(wl-target) < 0 else 'same_side'
    return out


def stats(rows):
    valid = [r for r in rows if r['valid']]
    def mean(key, rr=valid):
        return float(np.mean([r[key] for r in rr])) if rr else None
    def median(key, rr=rows):
        vals = [r[key] for r in rr if r.get(key) is not None]
        return float(np.median(vals)) if vals else None
    result = dict(queries=len(rows), supervised=len(valid), loss_best=mean('loss_best'), loss_last=mean('loss_last'),
                  loss_improved=sum(r['loss_delta'] < -1e-6 for r in valid),
                  loss_worsened=sum(r['loss_delta'] > 1e-6 for r in valid),
                  weight_best_mean=mean('weight_best', rows), weight_last_mean=mean('weight_last', rows),
                  median_abs_weight_delta=median('abs_weight_delta'),
                  median_best_safe_margin=median('best_safe_margin'),
                  median_last_distance_to_success=median('last_distance_to_success'))
    for key in ('exit_direction', 'target_relation', 'residual_relation', 'failure_kind'):
        result[key] = {str(k): sum(r[key] == k for r in rows) for k in sorted({r[key] for r in rows}, key=str)}
    return result


def main(args):
    root, ssd, report = args.root, args.ssd, args.report
    out = root/REL
    out.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    cfg = read(root/CAPACITY/'config.json')
    verification = read(root/CAPACITY/'verification.json')
    dense_verification = read(root/CAPACITY/'dense_verification.json')
    assert verification['status'] == 'passed' and not verification['test_accessed']
    assert verification['evaluation_sha256'] == sha(root/CAPACITY/'evaluation.json')
    ready = read(root/DATA/'ready.json')
    assert sha(root/DATA/'ready.json') == cfg['data_ready_sha256']
    refs = sorted(ready['scenes'], key=lambda r: r['scene_id'])
    assert {r['split'] for r in refs} == {'train', 'val'}
    scenes = {}
    for r in refs:
        sid = r['scene_id']
        assert sid in cfg['selection'][r['split']]
        assert sha(root/DATA/'scenes'/sid/'queries.json') == r['queries_sha256']
        scenes[sid] = load_scene(root/DATA, sid, root, ssd)
    assert len([s for s in scenes.values() if s['split'] == 'train']) == 12
    assert len([s for s in scenes.values() if s['split'] == 'val']) == 3
    assert not set(cfg['selection']['train']) & set(cfg['selection']['val'])
    allrows, summaries, histories, run_hashes = [], [], {}, {}
    checks = 0
    for seed in (42, 43):
        run = f'128_64_seed{seed}'
        path = root/CAPACITY/'runs'/run/'complete.json'
        r = read(path)
        assert r['hidden'] == [128, 64] and r['input_count'] == 58
        assert r['best_sha256'] == sha(path.parent/'best.pt')
        assert r['last_sha256'] == sha(path.parent/'last.pt')
        run_hashes[run] = sha(path)
        assert run_hashes[run] == dense_verification['run_hashes'][run]
        histories[str(seed)] = r['history']
        for split in ('train', 'val'):
            rows, pp, tt, ff = [], {'best': [], 'last': []}, [], []
            for s in scenes.values():
                if s['split'] != split:
                    continue
                sid = s['scene_id']
                predictions = {phase: r['evaluated'][phase][split]['predictions'][sid] for phase in ('best', 'last')}
                for phase in ('best', 'last'):
                    np.testing.assert_allclose(predict(s, predictions[phase]['weights']), predictions[phase]['poses'], atol=1e-8, rtol=0)
                    checks += len(s['truth'])
                    pp[phase].extend(predictions[phase]['poses'])
                tt.extend(s['truth']); ff.extend(s['grid_poses'][:, 2])
                for i, q in enumerate(s['queries']):
                    wb, wl = [float(predictions[p]['weights'][i]) for p in ('best', 'last')]
                    target, valid = float(s['interior_target'][i]), bool(s['valid'][i])
                    ps = {p: predictions[p]['poses'][i] for p in ('best', 'last')}
                    states = {p: status(ps[p], s['truth'][i]) for p in ps}
                    states['fixed'] = status(s['grid_poses'][i, 2], s['truth'][i])
                    d = diagnose(q['success']['joint'], wb, wl, target, valid)
                    for phase, w in [('best', wb), ('last', wl)]:
                        assert (component(d['success_components'], w) is not None) == states[phase]['joint']
                    if valid:
                        assert component(d['success_components'], target) is not None
                    lb, ll = [(float(logit(w))-float(logit(target)))**2 if valid else None for w in (wb, wl)]
                    row = dict(seed=seed, split=split, scene_id=sid, sample_index=q['sample_index'], pose_id=q['pose_id'],
                               image_path=q['image_path'], row=i, truth=s['truth'][i].tolist(),
                               weight_best=wb, weight_last=wl, abs_weight_delta=abs(wl-wb),
                               best_epoch=r['evaluated']['best']['epoch'], last_epoch=40,
                               target=target if valid else None, valid=valid, loss_best=lb, loss_last=ll,
                               loss_delta=ll-lb if valid else None, poses=ps, states=states,
                               joint_transition=transition(states['best']['joint'], states['last']['joint']),
                               xy_transition=transition(states['best']['xy'], states['last']['xy']),
                               fixed_best=transition(states['fixed']['joint'], states['best']['joint']),
                               fixed_last=transition(states['fixed']['joint'], states['last']['joint']),
                               failure_kind='xy_and_yaw' if not states['last']['xy'] and states['last']['yaw_deg'] > 30
                               else 'xy_only' if not states['last']['xy'] else 'yaw_only' if states['last']['yaw_deg'] > 30 else 'success', **d)
                    rows.append(row)
            for phase in ('best', 'last'):
                m = metrics(pp[phase], tt, ff)
                assert m == r['evaluated'][phase][split]['micro']
                reconstructed = np.mean([z['loss_'+phase] for z in rows if z['valid']])
                np.testing.assert_allclose(reconstructed, r['evaluated'][phase][split]['loss'], atol=2e-5, rtol=2e-5)
            cohorts = {c: stats([z for z in rows if z['joint_transition'] == c])
                       for c in ('stay_success', 'harm', 'repair', 'stay_failure')}
            harm = [z for z in rows if z['joint_transition'] == 'harm']
            improve_harm = [z for z in harm if z['valid'] and z['loss_delta'] < -1e-6]
            nb, nl = [sum(z['states'][p]['joint'] for z in rows) for p in ('best', 'last')]
            assert nl-nb == cohorts['repair']['queries']-cohorts['harm']['queries']
            # Fixed-.1 Harm counts can hide best->last losses among queries
            # that fixed-.1 already failed. Keep both reference definitions.
            fixed_harm = dict(
                best=sum(z['fixed_best'] == 'harm' for z in rows),
                last=sum(z['fixed_last'] == 'harm' for z in rows),
                newly_harmed=sum(z['states']['fixed']['joint'] and z['joint_transition'] == 'harm' for z in rows),
                recovered=sum(z['states']['fixed']['joint'] and z['joint_transition'] == 'repair' for z in rows))
            assert fixed_harm['last']-fixed_harm['best'] == fixed_harm['newly_harmed']-fixed_harm['recovered']
            losses = {}
            for c in cohorts:
                rr = [z for z in rows if z['joint_transition'] == c and z['valid']]
                losses[c] = float(sum(z['loss_delta'] for z in rr)/sum(z['valid'] for z in rows))
            np.testing.assert_allclose(sum(losses.values()),
                r['evaluated']['last'][split]['loss']-r['evaluated']['best'][split]['loss'], atol=2e-5, rtol=2e-5)
            scenes_summary = []
            for sid in sorted({z['scene_id'] for z in rows}):
                sr = [z for z in rows if z['scene_id'] == sid]
                scenes_summary.append(dict(scene_id=sid,
                    best_joint=float(np.mean([z['states']['best']['joint'] for z in sr])),
                    last_joint=float(np.mean([z['states']['last']['joint'] for z in sr])),
                    repair=sum(z['joint_transition'] == 'repair' for z in sr), harm=sum(z['joint_transition'] == 'harm' for z in sr), **stats(sr)))
            summaries.append(dict(seed=seed, split=split, queries=len(rows), best_epoch=r['evaluated']['best']['epoch'],
                best={k: v for k, v in r['evaluated']['best'][split].items() if k != 'predictions'},
                last={k: v for k, v in r['evaluated']['last'][split].items() if k != 'predictions'},
                cohorts=cohorts, loss_improved_harm=stats(improve_harm), fixed_harm=fixed_harm,
                loss_delta_contributions=losses, scenes=scenes_summary))
            allrows.extend(rows)
            print(split, seed, 'harm', len(harm), 'repair', cohorts['repair']['queries'], 'loss-improved harm', len(improve_harm), flush=True)
    # Deterministic examples, not a best-looking-query selection: for each
    # split/seed take first distinct pose for improved-loss harm, worsened-loss
    # harm and repair, then a second improved-loss harm if available.
    chosen = []
    for split in ('val', 'train'):
        for seed in (42, 43):
            seen = set()
            for category in ('harm_loss_down', 'harm_loss_up', 'repair', 'harm_loss_down'):
                for z in allrows:
                    key = (z['scene_id'], z['pose_id'])
                    if z['split'] != split or z['seed'] != seed or key in seen:
                        continue
                    match = z['joint_transition'] == 'repair' if category == 'repair' else z['joint_transition'] == 'harm' and z['valid'] and (z['loss_delta'] < -1e-6 if category == 'harm_loss_down' else z['loss_delta'] > 1e-6)
                    if match:
                        chosen.append(z);seen.add(key);break
    for folder in ('rgb', 'maps', 'heatmaps'):
        (report/folder).mkdir(exist_ok=True)
    banks, gallery = {}, []
    for z in chosen:
        sid, i = z['scene_id'], z['row']; s = scenes[sid]
        stem = f"{z['seed']}_{sid}_{z['sample_index']}"
        v, a = np.load(s['source']/'scores.npy', mmap_mode='r')[i]
        layers = np.array([((1-w)*v+w*a).max(1) for w in (z['weight_best'], z['weight_last'])], dtype='<f4')
        layers.tofile(report/'heatmaps'/(stem+'.bin'))
        shutil.copy2(ssd/z['image_path'], report/'rgb'/(stem+'.png'))
        shutil.copy2(root/'2DFP'/sid/'map.png', report/'maps'/(sid+'.png'))
        banks[sid] = dict(xy=s['xy'].tolist(), metadata=read(root/'2DFP'/sid/'floorplan_metadata.json'))
        gallery.append(dict(**z, image='rgb/'+stem+'.png', heatmap='heatmaps/'+stem+'.bin'))
    result = dict(summaries=summaries, model=dict(hidden=[128,64], inputs=58, parameters=15873,
        loss='MSE(logit(w), logit(frozen interior target)); eligible queries only',
        fixed_target=True, epochs=40, seeds=[42,43], test_accessed=False),
        scope=dict(train_scenes=12, val_scenes=3, train_queries=6387, val_queries=1704),
        notice='Exploratory Train/Val diagnosis; Val selected Best and is reused for analysis. No causal proof of input or supervision deficiency; no new fitting. Joint success: XY<=1m AND yaw<=30deg. Repair/Harm reference Best unless explicitly fixed .1.',
        interpretation=dict(
            primary='Both Train loss and Train Recall improve; Val loss worsens and Val Recall falls. Most supervised Val Harm also has worse individual loss (107/109). This is consistent with overfitting/generalization degradation, not evidence that the optimizer stopped fitting Train.',
            secondary='2/109 supervised Val Harm reduce loss by approaching a target in a DIFFERENT connected success component, stopping in a failure gap. Point-target mismatch is real but does not account for most Val Harm here.',
            scope='Seed42/43 are paired observations of the same queries; do not pool them as independent samples or unique queries. A bad Last does not identify whether targets, input adequacy, regularization or optimizer are the ultimate cause.',
            eligibility='Frozen valid uses successful 21-point grid and usable interior target; ineligible does NOT necessarily mean no successful continuous weight. All queries remain in Recall.'),
        test_accessed=False)
    atomic_json(out/'summary.json', result);atomic_json(report/'summary.json', result)
    atomic_json(report/'records.json', allrows);atomic_json(report/'gallery.json', gallery)
    atomic_json(report/'banks.json', banks);atomic_json(report/'histories.json', histories)
    fields = ['seed','split','scene_id','sample_index','pose_id','joint_transition','xy_transition','valid','weight_best','weight_last','target','loss_best','loss_last','loss_delta','best_safe_margin','exit_direction','exit_boundary','overshoot','last_distance_to_success','target_relation','residual_relation','failure_kind']
    with (report/'queries.csv').open('w') as f:
        w = csv.DictWriter(f, fields, extrasaction='ignore');w.writeheader();w.writerows(allrows)
    with (report/'metrics.csv').open('w') as f:
        w = csv.writer(f);keys=list(summaries[0]['best']['micro']);w.writerow(['split','seed','phase','epoch','loss',*keys])
        for r in summaries:
            for p in ('best','last'):
                w.writerow([r['split'],r['seed'],p,r['best_epoch'] if p=='best' else 40,r[p]['loss'],*[r[p]['micro'][k] for k in keys]])
    verified = dict(status='passed', test_accessed=False, run_hashes=run_hashes,
        prepared_ready_sha256=sha(root/DATA/'ready.json'), capacity_verification_sha256=sha(root/CAPACITY/'verification.json'),
        envelope_pose_checks=checks, success_membership_checks=checks, reconstructed_loss_matches=True,
        paired_recall_identity=True, fixed_harm_identity=True, source_sha256=sha(Path(__file__)),
        records_sha256=sha(report/'records.json'), scope=result['scope'])
    atomic_json(out/'verification.json', verified);atomic_json(report/'verification.json', verified)
    shutil.copy2(Path(__file__).with_suffix('.html'),report/'index.html')
    print('REPORT',report,flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--ssd',type=Path,required=True);p.add_argument('--report',type=Path,required=True);main(p.parse_args())

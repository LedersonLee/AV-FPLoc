"""TRAIN ONLY: independent scalar objective audit, no learned gate or Test I/O.

Predeclared main arm: expected Huber pose cost (1m,30deg units, yaw weight1,
T1) plus lambda1 peak-preservation on baseline-joint-correct queries. Lambda10,
unprotected risk, and a GT-hard-protection bound are explicit controls. NLL is
the earlier soft-GT CE (.2m,10deg); convex scalar minimization uses bisection.
Risk uses fine scouts plus all relevant piecewise-linear kinks and bounded
multi-start refinement. It is numerical search, NOT a proof of global optimality.
"""
import argparse
import csv
from pathlib import Path
import time

import numpy as np
import torch

from acoustic_branch.contracts import atomic_json, immutable_json, sha
from .input_supervision_audit import (
    SOURCE, REL as AUDIT, read, resolve, envelope, pose, yawdiff, recall,
)

REL = 'ReliabilityFusion/pilots/independent_pose_risk_v1'
METHODS = ['fixed_01', 'nll_t1', 'nll_t05', 'nll_t025', 'risk_t1',
           'risk_protect1', 'risk_protect10', 'risk_hard_bound', 'joint_success_oracle']
LABELS = dict(fixed_01='고정 acoustic w=0.1', nll_t1='독립 NLL · T=1',
    nll_t05='독립 NLL · T=0.5', nll_t025='독립 NLL · T=0.25',
    risk_t1='새 기대 pose 오차 · 보호 없음', risk_protect1='새 기대 pose 오차 + 보호 λ=1 · 주 조건',
    risk_protect10='새 기대 pose 오차 + 보호 λ=10',
    risk_hard_bound='새 기대 pose 오차 · GT hard 보호 상한 대조',
    joint_success_oracle='GT Joint 성공 oracle · 연속 가중치')


def huber_np(value):
    value = np.abs(value)
    return np.where(value <= 1, .5*value**2, value-.5)


def costs_and_target(xy, truth, yaws=36):
    distance = np.linalg.norm(xy-truth[:2], axis=1)
    angle = yawdiff(np.arange(yaws)*10., truth[2])
    cost = huber_np(distance[:, None]/1.) + huber_np(angle[None, :]/30.)
    target = -.5*(distance[:, None]/.2)**2-.5*(angle[None, :]/10.)**2
    target = np.exp(target-target.max()); target /= target.sum()
    good = (distance[:, None] <= 1) & (angle[None, :] <= 30)
    return cost.ravel(), target.ravel(), good.ravel()


class PosteriorBatch:
    def __init__(self, scores, cost, target, good, protect, device='cuda', weight_chunk=16):
        self.device = device; self.chunk = weight_chunk
        self.v = torch.as_tensor(np.ascontiguousarray(scores[:, 0].reshape(len(scores), -1)), dtype=torch.float64, device=device)
        self.d = torch.as_tensor(np.ascontiguousarray(scores[:, 1].reshape(len(scores), -1)), dtype=torch.float64, device=device)-self.v
        self.cost = torch.as_tensor(cost, dtype=torch.float64, device=device)
        target = torch.as_tensor(target, dtype=torch.float64, device=device)
        self.tv = (target*self.v).sum(1); self.td = (target*self.d).sum(1)
        self.good = torch.as_tensor(good, dtype=torch.bool, device=device)
        self.protect = torch.as_tensor(protect, dtype=torch.bool, device=device)
        self.b = len(scores)

    @torch.inference_mode()
    def evaluate(self, weights, temperature=1., gradient=False, poses=False):
        weights = np.asarray(weights, np.float64)
        if weights.ndim == 1:
            weights = weights[:, None]
        if weights.shape[0] != self.b or np.any(weights < 0) or np.any(weights > 1):
            raise ValueError('Bad scalar search weights')
        outputs = {n: [] for n in ('risk', 'penalty', 'nll', 'gradient', 'ids') if n not in ('gradient', 'ids') or (n == 'gradient' and gradient) or (n == 'ids' and poses)}
        for first in range(0, weights.shape[1], self.chunk):
            w = torch.as_tensor(weights[:, first:first+self.chunk], dtype=torch.float64, device=self.device)
            score = (self.v[:, None]+w[:, :, None]*self.d[:, None])/temperature
            maximum, idx = score.max(2)
            positive = torch.where(self.good[:, None], score, -torch.inf).amax(2)
            penalty = torch.where(self.protect[:, None], (maximum-positive)*temperature, 0.).clamp_min(0)
            score.sub_(maximum[:, :, None]).exp_()
            z = score.sum(2)
            risk = torch.einsum('bwn,bn->bw', score, self.cost)/z
            nll = maximum+z.log()-(self.tv[:, None]+w*self.td[:, None])/temperature
            values = dict(risk=risk, penalty=penalty, nll=nll)
            if gradient:
                values['gradient'] = (torch.einsum('bwn,bn->bw', score, self.d)/z-self.td[:, None])/temperature
            if poses:
                values['ids'] = idx
            for name in outputs:
                outputs[name].append(values[name].cpu().numpy())
        return {n: np.concatenate(v, axis=1) for n, v in outputs.items()}


def nll_weights(batch, temperature, steps=40):
    lo, hi = np.zeros(batch.b), np.ones(batch.b)
    ends = batch.evaluate(np.column_stack((lo, hi)), temperature, gradient=True)['gradient']
    left, right = ends[:, 0] >= 0, ends[:, 1] <= 0
    flat = (abs(ends[:, 0]) < 1e-12) & (abs(ends[:, 1]) < 1e-12)
    for _ in range(steps):
        mid = (lo+hi)/2
        derivative = batch.evaluate(mid, temperature, gradient=True)['gradient'][:, 0]
        hi = np.where(derivative >= 0, mid, hi); lo = np.where(derivative < 0, mid, lo)
    result = np.where(left, 0., np.where(right, 1., (lo+hi)/2))
    result[flat] = .1
    return result


def shortlist(weights, risk, penalty, lam, domains=None):
    """All observed local minima, plus all domain endpoints, not just one start."""
    picks, brackets = [], []
    for row in range(len(weights)):
        w = weights[row]; f = risk[row]+lam*penalty[row]
        allowed = domains[row] if domains is not None else [[0., 1.]]
        points = []; intervals = []
        for left, right in allowed:
            ids = np.flatnonzero((w >= left-1e-14) & (w <= right+1e-14))
            if not len(ids):
                raise ValueError('No sample in allowed domain')
            # Duplicate padded weights are removed before defining neighbours.
            ids = ids[np.r_[True, np.diff(w[ids]) > 1e-13]]
            vals = f[ids]
            for k, idx in enumerate(ids):
                if k == 0 or k == len(ids)-1 or (vals[k] <= vals[k-1] and vals[k] <= vals[k+1]):
                    points.append((float(w[idx]), float(f[idx])))
                    if 0 < k < len(ids)-1:
                        intervals.append((float(w[ids[k-1]]), float(w[ids[k+1]])))
        best = min(points, key=lambda p: (p[1], abs(p[0]-.1)))
        picks.append(best); brackets.append(intervals)
    return picks, brackets


def refine(batch, weights, risk, penalty, lam, domains=None, steps=35):
    picks, brackets = shortlist(weights, risk, penalty, lam, domains)
    counts = np.array([len(r) for r in brackets]); m = max(int(counts.max()), 1)
    left = np.array([[*(a for a, b in row), *([.1]*(m-len(row)))] for row in brackets])
    right = np.array([[*(b for a, b in row), *([.1]*(m-len(row)))] for row in brackets])
    phi = (np.sqrt(5.)-1)/2
    c = right-phi*(right-left); d = left+phi*(right-left)
    def evaluate(w):
        z = batch.evaluate(w); return z['risk']+lam*z['penalty']
    fc, fd = evaluate(c), evaluate(d)
    for _ in range(steps):
        keep_left = fc <= fd
        new_left = np.where(keep_left, left, c); new_right = np.where(keep_left, d, right)
        new_c = np.where(keep_left, new_right-phi*(new_right-new_left), d)
        new_d = np.where(keep_left, c, new_left+phi*(new_right-new_left))
        # One batched evaluation per step; retained values are reused.
        fresh = evaluate(np.where(keep_left, new_c, new_d))
        fc, fd = np.where(keep_left, fresh, fd), np.where(keep_left, fc, fresh)
        left, right, c, d = new_left, new_right, new_c, new_d
    ws = np.where(fc <= fd, c, d); fs = np.minimum(fc, fd)
    for i, (w0, f0) in enumerate(picks):
        for j in range(counts[i]):
            if fs[i, j] < f0-1e-12:
                w0, f0 = float(ws[i, j]), float(fs[i, j])
        picks[i] = (w0, f0)
    return np.array([p[0] for p in picks]), np.array([p[1] for p in picks])


def inset_intervals(intervals):
    out = []
    for lo, hi in intervals:
        if hi-lo <= 1e-12:
            continue
        eps = min(1e-8, (hi-lo)*1e-3)
        out.append([lo+eps, hi-eps])
    return out


def oracle_weight(row, xy, truth):
    fixed = pose(row['fixed_id'], xy)
    if np.linalg.norm(fixed[:2]-truth[:2]) <= 1 and yawdiff(fixed[2], truth[2]) <= 30:
        return .1
    options = []
    for lo, hi, idx in row['parts']:
        p = pose(idx, xy); distance = np.linalg.norm(p[:2]-truth[:2]); angle = float(yawdiff(p[2], truth[2]))
        if distance <= 1 and angle <= 30:
            w = (lo+hi)/2
            options.append((distance, angle, abs(w-.1), w))
    return min(options)[-1] if options else .1


def process_scene(root, ssd, out, ref, config):
    sid = ref['scene_id']; dest = out/'scenes'/sid; dest.mkdir(parents=True, exist_ok=True)
    if (dest/'complete.json').exists():
        result = read(dest/'complete.json')
        assert result['config_sha256'] == sha(out/'config.json') and result['data_sha256'] == sha(dest/'records.json')
        return result
    if ref['split'] != 'train' or sid not in config['train_scenes']:
        raise ValueError('Only approved TRAIN data permitted')
    folder = resolve(ref, ssd); receipt = read(folder/'receipt.json')
    assert sha(folder/'receipt.json') == ref['receipt_sha256']
    for name, digest in receipt['files'].items():
        assert sha(folder/name) == digest
    arrays = {n: np.load(folder/(n+'.npy'), mmap_mode='r') for n in ('scores', 'truth', 'xy')}
    ap = root/AUDIT/'scenes'/sid; ar = read(ap/'complete.json')
    assert ar['split'] == 'train' and ar['source_receipt_sha256'] == ref['receipt_sha256'] and sha(ap/'queries.json') == ar['queries_sha256']
    queries = read(ap/'queries.json'); xy = np.array(arrays['xy']); records = []
    assert sha(ap/'arrays.npz') == ar['arrays_sha256']
    with np.load(ap/'arrays.npz') as audit_arrays:
        frozen_fixed = audit_arrays['grid_poses'][:, 2].copy()
    for first in range(0, len(queries), config['batch_queries']):
        last = min(first+config['batch_queries'], len(queries)); qq = queries[first:last]
        scores = np.array(arrays['scores'][first:last]); truth = np.array(arrays['truth'][first:last]); b = len(qq)
        ct = [costs_and_target(xy, g) for g in truth]
        costs, targets, good = [np.stack([v[k] for v in ct]) for k in range(3)]
        fixed_ids = np.argmax(.9*scores[:, 0].reshape(b, -1)+.1*scores[:, 1].reshape(b, -1), axis=1)
        protect = good[np.arange(b), fixed_ids]
        for q, idx in zip(qq, fixed_ids):
            q['fixed_id'] = int(idx)
        batch = PosteriorBatch(scores, costs, targets, good, protect)
        domains = [inset_intervals(q['success']['joint']) if ok else [[0., 1.]] for q, ok in zip(qq, protect)]
        # Full/acceptable max-line kinks cover every protection-penalty kink.
        scouts = []; coarse_mask = []
        for i, q in enumerate(qq):
            values = set(np.linspace(0, 1, 401).tolist()); extra = {.1}
            extra.update(x for lo, hi, _ in q['parts'] for x in (lo, hi))
            if protect[i]:
                mask = good[i]
                positive = envelope(scores[i, 0].ravel()[mask], scores[i, 1].ravel()[mask])
                extra.update(x for lo, hi, _ in positive for x in (lo, hi))
            for lo, hi in domains[i]:
                extra.update((lo, hi, (lo+hi)/2))
            values.update(extra); values = np.array(sorted(values)); scouts.append(values)
            coarse_mask.append(np.array([abs(w*200-round(w*200)) < 1e-9 or w in extra for w in values]))
        m = max(map(len, scouts)); w = np.array([np.pad(v, (0, m-len(v)), constant_values=1.) for v in scouts])
        z = batch.evaluate(w)
        weights = {'fixed_01': np.full(b, .1)}
        for name, temp in [('nll_t1', 1.), ('nll_t05', .5), ('nll_t025', .25)]:
            weights[name] = nll_weights(batch, temp)
        solver = {}
        for name, lam, domain in [('risk_t1', 0., None), ('risk_protect1', 1., None),
                                  ('risk_protect10', 10., None), ('risk_hard_bound', 0., domains)]:
            weights[name], minima = refine(batch, w, z['risk'], z['penalty'], lam, domain)
            # Numerically refine a coarser independent scout as a convergence check.
            wc = [s[mask] for s, mask in zip(scouts, coarse_mask)]; mc = max(map(len, wc))
            rr = [z['risk'][i, :len(s)][mask] for i, (s, mask) in enumerate(zip(scouts, coarse_mask))]
            hh = [z['penalty'][i, :len(s)][mask] for i, (s, mask) in enumerate(zip(scouts, coarse_mask))]
            wc = np.array([np.pad(v, (0, mc-len(v)), constant_values=1.) for v in wc])
            rc = np.array([np.pad(v, (0, mc-len(v)), mode='edge') for v in rr])
            hc = np.array([np.pad(v, (0, mc-len(v)), mode='edge') for v in hh])
            cw, cm = refine(batch, wc, rc, hc, lam, domain)
            # Keep the better objective, never choose based on Recall/GT MAP.
            convergence_gap = np.abs(cm-minima)
            better = cm < minima
            weights[name] = np.where(better, cw, weights[name]); minima = np.minimum(minima, cm)
            solver[name] = dict(refined_objective=minima, coarse_fine_gap=convergence_gap)
        weights['joint_success_oracle'] = np.array([oracle_weight(q, xy, g) for q, g in zip(qq, truth)])
        matrix = np.column_stack([weights[name] for name in METHODS]); final = batch.evaluate(matrix, poses=True)
        reproduced_fixed = np.array([pose(int(idx), xy) for idx in final['ids'][:, 0]])
        np.testing.assert_allclose(reproduced_fixed, frozen_fixed[first:last], atol=1e-7, rtol=0)
        # NLL residuals are checked at each one's own temperature.
        nll_checks = {}
        for name, temp in [('nll_t1', 1.), ('nll_t05', .5), ('nll_t025', .25)]:
            nn = batch.evaluate(weights[name], temp, gradient=True)
            nll_checks[name] = nn
        for i, (q, g) in enumerate(zip(qq, truth)):
            methods = {}
            for j, name in enumerate(METHODS):
                methods[name] = dict(weight=float(matrix[i, j]), pose=pose(int(final['ids'][i, j]), xy).tolist(),
                    expected_cost=float(final['risk'][i, j]), protection_gap=float(final['penalty'][i, j]), nll_t1=float(final['nll'][i, j]))
                if name in nll_checks:
                    methods[name].update(own_nll=float(nll_checks[name]['nll'][i, 0]), nll_derivative=float(nll_checks[name]['gradient'][i, 0]))
                if name in solver:
                    methods[name].update(objective=float(solver[name]['refined_objective'][i]), coarse_fine_gap=float(solver[name]['coarse_fine_gap'][i]))
            if protect[i]:
                hp = np.array(methods['risk_hard_bound']['pose'])
                if np.linalg.norm(hp[:2]-g[:2]) > 1 or yawdiff(hp[2], g[2]) > 30:
                    raise ValueError('Hard protection did not preserve baseline success')
            records.append(dict(scene_id=sid, split='train', sample_index=q['sample_index'], pose_id=q['pose_id'], truth=g.tolist(),
                baseline_joint_success=bool(protect[i]), success_possible=bool(q['success']['joint']), methods=methods))
        atomic_json(dest/'progress.json', dict(done=last, total=len(queries), test_accessed=False))
        atomic_json(out/'status.json', dict(status='optimizing_train', scene_id=sid, done=last, total=len(queries), test_accessed=False))
        if first % 64 == 0:
            print('optimize', sid, last, '/', len(queries), flush=True)
        del batch
    atomic_json(dest/'records.json', records)
    result = dict(status='complete', scene_id=sid, split='train', queries=len(records), test_accessed=False,
                  config_sha256=sha(out/'config.json'), data_sha256=sha(dest/'records.json'), source_receipt_sha256=ref['receipt_sha256'])
    atomic_json(dest/'complete.json', result)
    return result


def summarize(records):
    truth = np.array([r['truth'] for r in records]); fixed = np.array([r['methods']['fixed_01']['pose'] for r in records])
    possible = np.array([r['success_possible'] for r in records]); result = {}
    for name in METHODS:
        rows = [r['methods'][name] for r in records]; pp = np.array([r['pose'] for r in rows])
        m = recall(pp, truth, fixed)
        actual = (np.linalg.norm(pp[:, :2]-truth[:, :2], axis=1) <= 1) & (yawdiff(pp[:, 2], truth[:, 2]) <= 30)
        m.update(queries=len(records), mean_weight=float(np.mean([r['weight'] for r in rows])),
                 mean_expected_cost=float(np.mean([r['expected_cost'] for r in rows])),
                 mean_nll_t1=float(np.mean([r['nll_t1'] for r in rows])),
                 missed_recoverable=int((possible & ~actual).sum()),
                 baseline_joint_preserved=int((np.array([r['baseline_joint_success'] for r in records]) & actual).sum()))
        if 'coarse_fine_gap' in rows[0]:
            gaps = np.array([r['coarse_fine_gap'] for r in rows]); m.update(max_coarse_fine_objective_gap=float(gaps.max()), grid_disagreements_over1e6=int((gaps > 1e-6).sum()))
        if 'nll_derivative' in rows[0]:
            ww = np.array([r['weight'] for r in rows]); grad = np.array([r['nll_derivative'] for r in rows])
            violation = np.where(ww <= 1e-10, np.maximum(-grad, 0), np.where(ww >= 1-1e-10, np.maximum(grad, 0), abs(grad)))
            m['nll_max_kkt_violation'] = float(violation.max())
        result[name] = m
    return result


def main(args):
    root, ssd = args.root, args.ssd; out = root/REL; out.mkdir(parents=True, exist_ok=True)
    source = root/SOURCE; cfg = read(source/'config.json')
    # Filter before resolving/opening ANY score arrays, including Val.
    refs = [r for r in read(source/'scores_ready.json')['entries'] if r['split'] == 'train']
    assert {r['scene_id'] for r in refs} == set(cfg['selection']['train']) and len(refs) == 12
    config = dict(version='independent_pose_risk_v1', train_scenes=cfg['selection']['train'], train_queries=6387,
        source_config_sha256=sha(source/'config.json'), source_scores_sha256=sha(source/'scores_ready.json'),
        source_score_receipts={r['scene_id']:r['receipt_sha256'] for r in refs},
        score_transform='Existing frozen standardized scores, no recalibration', temperature=1.,
        cost=dict(kind='Huber delta1', xy_unit_m=1., yaw_unit_deg=30., yaw_weight=1., reduction='posterior expected candidate cost, NOT cost of mean pose'),
        protection='Only baseline w.1 joint-correct TRAIN queries; max_all_score minus max_GT_joint_region_score, score units, no forced positive margin',
        lambdas=[1., 10.], primary='risk_protect1',
        hard_bound='GT-dependent constrained numerical optimum on baseline-correct queries; NOT a deployable policy or standalone trainable loss',
        nll=dict(target_xy_sigma_m=.2, target_yaw_sigma_deg=10., temperatures=[1., .5, .25], solver='Convex derivative bisection40; endpoint/KKT checks'),
        risk_solver='401point scout + exact full/GT envelope kinks + safe-domain edges; all sampled local minima golden35; independent201point scout refinement check',
        precision='float64', batch_queries=args.batch, oracle='Keep w.1 if already joint-correct; else any continuous successful MAP interval (ties XY then yaw then closest to .1); no joint solution -> w.1',
        notice='GT per-query oracle diagnostic, not learned model or generalization result; risk numerical global-search evidence, not global proof',
        method_order=METHODS, labels=LABELS, val_accessed=False, test_accessed=False)
    immutable_json(out/'config.json', config)
    torch.set_num_threads(4)
    for ref in refs:
        info = process_scene(root, ssd, out, ref, config)
        print('SCENE COMPLETE', info['scene_id'], info['queries'], flush=True)
    records = [r for ref in refs for r in read(out/'scenes'/ref['scene_id']/'records.json')]
    assert len(records) == 6387 and {r['split'] for r in records} == {'train'}
    methods = summarize(records)
    for name, metric in methods.items():
        metric['per_scene'] = {ref['scene_id']:summarize([r for r in records if r['scene_id'] == ref['scene_id']])[name] for ref in refs}
    assert methods['joint_success_oracle']['recall_1m_30deg'] == sum(r['success_possible'] for r in records)/len(records)
    # Every fixed pose was reproduced from TRAIN-only frozen grid caches above.
    assert methods['fixed_01']['recall_1m_30deg'] == 4493/6387
    for name in ('risk_t1', 'risk_protect1', 'risk_protect10', 'risk_hard_bound'):
        assert methods[name]['max_coarse_fine_objective_gap'] < 1e-5
    for name in ('nll_t1', 'nll_t05', 'nll_t025'):
        assert methods[name]['nll_max_kkt_violation'] < 1e-6
    result = dict(status='complete', config=config, methods=methods, train_queries=len(records),
                  joint_recoverable=sum(r['success_possible'] for r in records), val_accessed=False, test_accessed=False,
                  verification=dict(baseline_reproduced=True, continuous_oracle_reproduced=True, nll_kkt_passed=True, risk_scout_refinement_passed=True))
    atomic_json(out/'results.json', result)
    atomic_json(out/'records.json', records)
    columns = ['method', *[k for k in methods['fixed_01'] if k != 'per_scene']]
    with (out/'metrics.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore'); w.writeheader()
        for name in METHODS:
            w.writerow(dict(method=LABELS[name], **methods[name]))
    atomic_json(out/'status.json', dict(status='complete', val_accessed=False, test_accessed=False))
    print('COMPLETE', out/'results.json', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, required=True); parser.add_argument('--ssd', type=Path, required=True)
    parser.add_argument('--batch', type=int, default=8); main(parser.parse_args())

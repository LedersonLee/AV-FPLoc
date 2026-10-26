"""TRAIN-only independent canonical-score replay of scalar objective results.

Replay with the exact historical expression (1-w)*v+w*a, not its algebraically
equivalent v+w*(a-v), to make deterministic argmax tie ownership explicit. Also
check oracle at all MAP-interval interiors AND boundary points/endpoints.
Preserves raw optimizer receipts; writes separate verified artifacts.
"""
import argparse
from pathlib import Path
import shutil
import csv

import numpy as np
import torch

from acoustic_branch.contracts import atomic_json, sha
from .independent_pose_risk import REL, METHODS, LABELS, costs_and_target, summarize, inset_intervals
from .input_supervision_audit import SOURCE, REL as AUDIT, read, resolve, pose, yawdiff


@torch.inference_mode()
def replay(scores, weights, cost, target, good):
    v = torch.tensor(scores[:, 0].reshape(len(scores), -1), dtype=torch.float64, device='cuda')
    a = torch.tensor(scores[:, 1].reshape(len(scores), -1), dtype=torch.float64, device='cuda')
    cost = torch.tensor(cost, dtype=torch.float64, device='cuda')
    target = torch.tensor(target, dtype=torch.float64, device='cuda')
    good = torch.tensor(good, dtype=torch.bool, device='cuda')
    outputs = {k: [] for k in ('ids', 'expected_cost', 'joint_mass_t1', 'gt_score_gap', 'map_probability_t1', 'nll_t1')}
    for first in range(0, weights.shape[1], 12):
        w = torch.tensor(weights[:, first:first+12], dtype=torch.float64, device='cuda')
        score = (1-w[:, :, None])*v[:, None]+w[:, :, None]*a[:, None]
        maximum, idx = score.max(2)
        positive = torch.where(good[:, None], score, -torch.inf).amax(2)
        expected = torch.einsum('bwn,bn->bw', score, target)
        score.sub_(maximum[:, :, None]).exp_(); z = score.sum(2)
        values = dict(ids=idx, expected_cost=torch.einsum('bwn,bn->bw', score, cost)/z,
            joint_mass_t1=torch.einsum('bwn,bn->bw', score, good.to(torch.float64))/z,
            gt_score_gap=maximum-positive, map_probability_t1=1/z,
            nll_t1=maximum+z.log()-expected)
        for k in outputs:
            outputs[k].append(values[k].cpu().numpy())
    return {k: np.concatenate(v, axis=1) for k, v in outputs.items()}


def run(args):
    root, ssd = args.root, args.ssd; out = root/REL
    raw = read(out/'results.json'); config = read(out/'config.json')
    refs = [r for r in read(root/SOURCE/'scores_ready.json')['entries'] if r['split'] == 'train']
    assert len(refs) == 12 and {r['scene_id'] for r in refs} == set(config['train_scenes'])
    records = []; differences = {n: 0 for n in METHODS}; max_cost_error = 0.; old_possible = 0
    risk_probes = [('risk_t1', 0.), ('risk_protect1', 1.), ('risk_protect10', 10.)]
    for ref in refs:
        sid = ref['scene_id']; folder = resolve(ref, ssd); dest = out/'scenes'/sid
        receipt = read(dest/'complete.json'); assert sha(dest/'records.json') == receipt['data_sha256']
        assert sha(folder/'receipt.json') == ref['receipt_sha256']
        rr = read(dest/'records.json'); audit = read(root/AUDIT/'scenes'/sid/'queries.json')
        s = np.load(folder/'scores.npy', mmap_mode='r'); xy = np.load(folder/'xy.npy'); verified = []
        for first in range(0, len(rr), 8):
            rows = rr[first:first+8]; aa = audit[first:first+8]; b = len(rows)
            costs = [costs_and_target(xy, np.array(r['truth'])) for r in rows]
            cost, target, good = [np.stack([c[k] for c in costs]) for k in range(3)]
            ww = np.array([[r['methods'][n]['weight'] for n in METHODS] for r in rows])
            oracle = []
            for r in aa:
                points = {0., .1, 1.}
                for lo, hi, idx in r['parts']:
                    points.update((lo, hi, (lo+hi)/2))
                oracle.append(sorted(points))
            count = max(map(len, oracle)); candidates = np.array([np.pad(v, (0, count-len(v)), constant_values=.1) for v in oracle])
            probes = np.zeros((b, len(risk_probes))); distances = np.zeros_like(probes)
            for i, (r, q) in enumerate(zip(rows, aa)):
                for j, (name, lam) in enumerate(risk_probes):
                    w = r['methods'][name]['weight']
                    if r['baseline_joint_success']:
                        interiors = inset_intervals(q['success']['joint'])
                        options = [float(np.clip(w, lo, hi)) for lo, hi in interiors]
                        probes[i,j] = min(options, key=lambda val: abs(val-w))
                        distances[i,j] = abs(probes[i,j]-w)
                    else:
                        probes[i,j] = w
            weights = np.column_stack((ww, candidates, probes)); values = replay(np.array(s[first:first+b]), weights, cost, target, good)
            for i, r in enumerate(rows):
                old_possible += r['success_possible']; gt = np.array(r['truth'])
                candidates_pose = np.array([pose(int(idx), xy) for idx in values['ids'][i, len(METHODS):len(METHODS)+len(oracle[i])]])
                dd = np.linalg.norm(candidates_pose[:, :2]-gt[:2], axis=1); angles = yawdiff(candidates_pose[:, 2], gt[2]); ok = (dd <= 1) & (angles <= 30)
                if r['success_possible'] and not ok.any():
                    raise ValueError('Stable oracle not reproduced')
                r['success_possible'] = bool(ok.any())
                oidx = len(METHODS)-1
                if r['baseline_joint_success']:
                    pick = oracle[i].index(.1)
                elif ok.any():
                    pick = min(np.flatnonzero(ok), key=lambda j:(dd[j], angles[j], abs(oracle[i][j]-.1)))
                else:
                    pick = oracle[i].index(.1)
                choices = list(range(len(METHODS))); choices[oidx] = len(METHODS)+int(pick)
                for j, name in enumerate(METHODS):
                    k = choices[j]; old = r['methods'][name]; pp = pose(int(values['ids'][i, k]), xy)
                    differences[name] += int(not np.allclose(pp, old['pose'], atol=1e-9, rtol=0))
                    if name != 'joint_success_oracle':
                        max_cost_error = max(max_cost_error, abs(float(values['expected_cost'][i,k])-old['expected_cost']))
                    old.update(pose=pp.tolist(), weight=float(weights[i,k]))
                    for field in ('expected_cost', 'joint_mass_t1', 'gt_score_gap', 'map_probability_t1', 'nll_t1'):
                        old[field] = float(values[field][i,k])
                    old['protection_gap'] = max(0., old['gt_score_gap']) if r['baseline_joint_success'] else 0.
                    old['map_pose_cost'] = float(cost[i, int(values['ids'][i,k])])
                for j, (name, lam) in enumerate(risk_probes):
                    k = len(METHODS)+count+j; old = r['methods'][name]
                    probe_pose = pose(int(values['ids'][i,k]), xy)
                    probe_good = np.linalg.norm(probe_pose[:2]-gt[:2]) <= 1 and yawdiff(probe_pose[2], gt[2]) <= 30
                    objective = old['expected_cost']+lam*old['protection_gap']
                    probe_objective = values['expected_cost'][i,k]+(lam*max(0., values['gt_score_gap'][i,k]) if r['baseline_joint_success'] else 0.)
                    old.update(safe_probe_weight=float(probes[i,j]), safe_probe_distance=float(distances[i,j]),
                               safe_probe_joint_success=bool(probe_good), safe_probe_objective_increase=float(probe_objective-objective))
                verified.append(r)
        atomic_json(dest/'verified_records.json', verified); records.extend(verified)
        print('VERIFIED', sid, len(verified), flush=True)
    assert len(records) == 6387 and max_cost_error < 1e-8
    methods = summarize(records)
    for name in METHODS:
        rows = [r['methods'][name] for r in records]
        for field in ('joint_mass_t1', 'gt_score_gap', 'map_probability_t1', 'map_pose_cost'):
            methods[name]['mean_'+field] = float(np.mean([r[field] for r in rows]))
        methods[name]['per_scene'] = {ref['scene_id']:summarize([r for r in records if r['scene_id'] == ref['scene_id']])[name] for ref in refs}
    for key in ('recall_0.1m', 'recall_0.5m', 'recall_1m', 'recall_1m_30deg'):
        assert methods['fixed_01'][key] == raw['methods']['fixed_01'][key]
    assert methods['risk_hard_bound']['harm_joint'] == 0
    assert methods['joint_success_oracle']['harm_joint'] == 0
    assert methods['joint_success_oracle']['recall_1m_30deg'] == sum(r['success_possible'] for r in records)/len(records)
    boundary = {}
    for name, lam in risk_probes:
        harmed = []
        for r in records:
            m = r['methods'][name]; gt = np.array(r['truth']); pp = np.array(m['pose'])
            if r['baseline_joint_success'] and not (np.linalg.norm(pp[:2]-gt[:2]) <= 1 and yawdiff(pp[2], gt[2]) <= 30):
                harmed.append(m)
        boundary[name] = dict(joint_harm=len(harmed), near_boundary_harm_under1e7=sum(m['safe_probe_distance'] < 1e-7 for m in harmed),
            equally_good_objective_safe_alternative=sum(m['safe_probe_joint_success'] and m['safe_probe_objective_increase'] <= 1e-6 for m in harmed),
            note='GT nearest-success-interior probe is diagnostic ONLY, not a new method or selected weight. Objective tolerance1e-6, weight boundary threshold1e-7.')
    verification = dict(status='passed', train_queries=6387, train_scenes=12, val_accessed=False, test_accessed=False,
        canonical_fusion='(1-w)*v+w*a in float64; direct all-candidate argmax', pose_changes_vs_rearranged_expression=differences,
        max_expected_cost_replay_error=max_cost_error, stable_joint_recoverable=old_possible,
        joint_recoverable_including_boundary_ties=sum(r['success_possible'] for r in records),
        nll_max_kkt=max(methods[n]['nll_max_kkt_violation'] for n in ('nll_t1', 'nll_t05', 'nll_t025')),
        risk_max_coarse_fine_gap=max(methods[n]['max_coarse_fine_objective_gap'] for n in ('risk_t1', 'risk_protect1', 'risk_protect10', 'risk_hard_bound')),
        source_code={n:sha(Path(__file__).with_name(n)) for n in ('independent_pose_risk.py', 'verify_pose_risk.py')})
    result = dict(status='complete', config=config, methods=methods, verification=verification, train_queries=6387,
        joint_recoverable=sum(r['success_possible'] for r in records), val_accessed=False, test_accessed=False,
        boundary_diagnostics=boundary,
        oracle_notice='GT-selected weights; exact scalar affine-MAP intervals plus canonical boundary checks. Hard-protection control uses GT, not learned inference.')
    atomic_json(out/'verified_results.json', result); atomic_json(out/'verified_records.json', records); atomic_json(out/'verification.json', verification)
    atomic_json(out/'boundary_diagnostics.json', boundary)
    columns = ['method', *[k for k in methods['fixed_01'] if k != 'per_scene']]
    with (out/'verified_metrics.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore'); writer.writeheader()
        for name in METHODS:
            writer.writerow(dict(method=LABELS[name], **methods[name]))
    if args.report:
        args.report.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out/'verified_results.json', args.report/'results.json')
        shutil.copy2(out/'verified_metrics.csv', args.report/'metrics.csv')
        shutil.copy2(out/'verification.json', args.report/'verification.json')
        shutil.copy2(Path(__file__).with_name('independent_pose_risk_report.html'), args.report/'index.html')
    print('VERIFICATION COMPLETE', out/'verified_results.json', flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True); p.add_argument('--ssd', type=Path, required=True)
    p.add_argument('--report', type=Path); run(p.parse_args())

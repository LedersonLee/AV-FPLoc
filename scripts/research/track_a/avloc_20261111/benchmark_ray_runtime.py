"""Stage-resolved batch-one frozen inference timing; never training or Test data.

Full mode samples one uniform query per Val scene, independently of errors.
Smoke mode checks integration only and must not support latency claims.
The existing prepare() processes both modalities even for single-expert arms;
this measures the current implementation, not an optimized deployment bound.
"""
import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import default_collate

from augment import prepare
from dataset import Observations
from evaluate_baseline_fields import sha
from matching import FOV, laplace_field, proposals
from model import AVLoc


def timed(fn, device):
    if device == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter_ns()
    result = fn()
    if device == 'cuda':
        torch.cuda.synchronize()
    return result, (time.perf_counter_ns() - start) / 1e6


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--arm', choices=['A0', 'A1', 'A2', 'A2_disabled', 'A3'], required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--grid-cm', type=int, choices=[10, 20], default=20)
    p.add_argument('--repeats', type=int, default=10)
    p.add_argument('--warmups', type=int, default=3)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    assert a.repeats >= 1 and a.warmups >= 1
    run = a.study / f'runs/{a.arm}_seed{a.seed}'
    assert (run / 'complete.json').is_file(), 'Only naturally completed runs'
    best = json.loads((run / 'best.json').read_text())
    checkpoint = run / best['path']
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    protocol = ck['protocol']
    assert protocol['arm'] == a.arm and protocol['seed'] == a.seed
    assert sha(a.study / 'data_v1/manifest.json') == protocol['input_manifest_sha256']
    for name in ['model.py', 'dataset.py', 'augment.py', 'audio.py']:
        assert sha(Path(__file__).with_name(name)) == protocol['source_sha256'][name]
    predroot = a.study / f'predictions/{a.arm}_seed{a.seed}'
    pm = json.loads((predroot / 'manifest.json').read_text())
    assert pm['complete'] and pm['queries'] == 9965 and pm['checkpoint_sha256'] == sha(checkpoint)
    bm = json.loads((a.study / 'banks_v1/manifest.json').read_text())
    assert bm['complete']
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if a.device == 'cuda':
        torch.cuda.set_per_process_memory_fraction(.25)
    model = AVLoc(a.arm).to(a.device).eval()
    model.load_state_dict(ck['model'], strict=True)
    del ck
    ds = Observations(a.study / 'data_v1', 'val')
    assert len(ds) == 9965 and len(ds.records) == 19
    head = {'A0': 'V', 'A1': 'A', 'A2': 'AV', 'A2_disabled': 'AV', 'A3': 'F'}[a.arm]
    directions = FOV if head == 'V' else None
    rng = np.random.default_rng(20261111)
    chosen = []
    offset = 0
    for rec in ds.records:
        chosen.append((rec['scene'], offset + int(rng.integers(rec['queries']))))
        offset += rec['queries']
    a.out.mkdir(parents=True, exist_ok=False)
    records = []
    with torch.inference_mode():
        for sid, index in (chosen[:1] if a.smoke else chosen):
            # File decoding, checkpoint/bank loading, and hashes are outside timings.
            batch = default_collate([ds[index]])
            qid = int(batch['query_id'][0])
            pr = next(r for r in pm['records'] if r['scene'] == sid)
            pp = predroot / pr['file']
            assert sha(pp) == pr['sha256']
            with np.load(pp) as pred:
                j = int(np.flatnonzero(pred['query_ids'] == qid)[0])
                expected = (pred[head + '_mu'][j].copy(), pred[head + '_scale'][j].copy())
            bp = a.study / f'banks_v1/{sid}/grid_{a.grid_cm}cm.npz'
            br = next(r for r in bm['records'] if r['scene'] == sid)
            assert sha(bp) == br['files'][bp.name]
            with np.load(bp) as bank:
                world, xy = bank['world_rays'].copy(), bank['xy'].copy()
            ref_field = laplace_field(world, *expected, directions)
            ref_ids = proposals(ref_field, xy)
            samples = []
            max_delta = 0.
            for rep in range(a.warmups + a.repeats):
                (rgb, spec, _), prep_ms = timed(lambda: prepare(batch, a.device, False), a.device)
                output, model_ms = timed(lambda: model(rgb, spec), a.device)
                arrays, transfer_ms = timed(lambda: tuple(t[0].cpu().numpy().copy() for t in output[head]), a.device)
                field, field_ms = timed(lambda: laplace_field(world, *arrays, directions), 'cpu')
                ids, nms_ms = timed(lambda: proposals(field, xy), 'cpu')
                # Batch-one/CPU kernels need not be bit-identical to batch-eight GPU prediction.
                for got, wanted in zip(arrays, expected):
                    np.testing.assert_allclose(got, wanted, rtol=2e-4, atol=2e-4)
                    max_delta = max(max_delta, float(np.max(abs(got - wanted))))
                np.testing.assert_array_equal(ids, ref_ids)
                assert int(np.argmax(field)) == int(np.argmax(ref_field))
                if rep >= a.warmups:
                    samples.append(dict(prepare_ms=prep_ms, model_ms=model_ms,
                                        output_transfer_ms=transfer_ms, field_ms=field_ms,
                                        nms_ms=nms_ms, stage_sum_ms=prep_ms + model_ms + transfer_ms + field_ms + nms_ms))
            records.append(dict(scene=sid, query_id=qid, xy_candidates=len(xy),
                                prediction_max_abs_delta=max_delta, candidate_replay_exact=True,
                                samples=samples, bank_sha256=sha(bp), prediction_sha256=pr['sha256']))
            print('CHECKED', sid, qid, flush=True)
    pooled = {k: [s[k] for r in records for s in r['samples']] for k in records[0]['samples'][0]}
    result = dict(complete=True, smoke=a.smoke, arm=a.arm, seed=a.seed, grid_cm=a.grid_cm,
                  scope='integration check only' if a.smoke else 'uniform one query per Val19 scene; warm batch-one stage timings',
                  exclusions='RGB/RIR disk decode, bank/checkpoint load, initialization, posterior diagnostics. No FP-RIR baseline encoder or GEOQ time included.',
                  limitation='Shared host; isolation is not established by this script. prepare computes both modalities for every arm. Stage sum includes synchronization boundaries; not streaming throughput.',
                  source_sha256={n: sha(Path(__file__).with_name(n)) for n in ['benchmark_ray_runtime.py', 'matching.py', 'model.py', 'dataset.py', 'augment.py', 'audio.py']},
                  checkpoint_sha256=sha(checkpoint), training_code_commit=protocol['code_commit'],
                  prediction_manifest_sha256=sha(predroot / 'manifest.json'),
                  device=a.device, platform=platform.platform(), torch_version=torch.__version__,
                  numpy_version=np.__version__, torch_threads=torch.get_num_threads(),
                  gpu=torch.cuda.get_device_name() if a.device == 'cuda' else None,
                  cpu_affinity=sorted(os.sched_getaffinity(0)), load_average=list(os.getloadavg()),
                  threads_environment={k: os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']},
                  warmups=a.warmups, repeats=a.repeats, queries=len(records), records=records,
                  pooled_ms={k: dict(median=float(np.median(v)), p90=float(np.percentile(v, 90))) for k, v in pooled.items()},
                  test_accessed=False, training=False)
    (a.out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()

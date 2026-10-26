"""Train/val-only acoustic diagnostics. Never opens test data or changes splits."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from .contracts import atomic_json, sha
from .data import read, load_manifest, check_cache
from .train import AcousticEncoder, PairDataset, contrastive_loss


def pose_partition(queries):
    """Keep every yaw at one physical position on the same side."""
    keys = sorted({tuple(q['observed']['xyz']) for q in queries})
    held = set(keys[::5])
    return ([q for q in queries if tuple(q['observed']['xyz']) not in held],
            [q for q in queries if tuple(q['observed']['xyz']) in held])


def tensors(dataset, indices, device):
    rows = [dataset[i] for i in indices]
    return [torch.stack([r[k] for r in rows]).to(device) for k in range(4)]


def scores(model, batch):
    oe, oc, ce, cc = batch
    b, n = ce.shape[:2]
    return model(oe, oc), model(ce.flatten(0, 1), cc.flatten(0, 1)).reshape(b, n, -1)


def metrics(logits):
    return dict(top1=float((logits.argmax(-1) == 0).float().mean()),
                loss=float(F.cross_entropy(logits, torch.zeros(len(logits), dtype=torch.long, device=logits.device))))


def run(root, config_path, output):
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    config = read(config_path)
    result = dict(protocol='diagnostic_v1', test_accessed=False, config_sha256=sha(config_path),
                  baseline_note='cached normalized log-STFT energy L1 is NOT legacy raw STFT-L1',
                  baseline={}, experiments=[])
    datasets = {}
    for split in ('train', 'val'):
        atomic_json(output/'status.json', dict(status='running', stage='audit_'+split, updated_unix=time.time()))
        queries, digest = load_manifest(root, split, config)
        locations, cache_sha = check_cache(root, config, split, digest, full=True)
        datasets[split] = PairDataset(queries, locations)
        totals = {k: [] for k in ('energy_l1', 'coherence_l2')}
        ties = {k: 0 for k in totals}
        for start in range(0, len(queries), 32):
            oe, oc, ce, cc = tensors(datasets[split], range(start, min(start+32, len(queries))), 'cpu')
            logits = dict(energy_l1=-(ce-oe[:, None]).abs().flatten(2).mean(-1),
                          coherence_l2=-(cc-oc[:, None]).square().flatten(2).mean(-1))
            for key, value in logits.items():
                # Ties receive fractional credit, never optimistic slot-zero argmax.
                maximum = value.max(-1, keepdim=True).values
                winners = value == maximum
                totals[key].extend((winners[:, 0]/winners.sum(-1)).tolist())
                ties[key] += int((winners.sum(-1) > 1).sum())
        result['baseline'][split] = dict(queries=len(queries), cache_sha256=cache_sha,
            **{k: dict(top1=float(np.mean(v)), tied_queries=ties[k]) for k, v in totals.items()})
        atomic_json(output/'results.json', result)
    # One TRAIN scene only; held physical positions remain diagnostic TRAIN data,
    # never relabelled as official val/test. Reuse fixed seven-negative pools.
    scene = sorted({q['scene_id'] for q in datasets['train'].queries})[0]
    selected = [q for q in datasets['train'].queries if q['scene_id'] == scene]
    fit, held = pose_partition(selected)
    fit, held = fit[:64], held[:64]
    if not fit or not held:
        raise ValueError('Insufficient pose-disjoint diagnostic data')
    device = torch.device('cuda')
    fit_ds = PairDataset(fit, datasets['train'].store.locations)
    held_ds = PairDataset(held, datasets['train'].store.locations)
    fit_batch = tensors(fit_ds, range(len(fit)), device)
    held_batch = tensors(held_ds, range(len(held)), device)
    result['partition'] = dict(scene=scene, fit_sample_indices=[q['sample_index'] for q in fit],
        held_sample_indices=[q['sample_index'] for q in held],
        scope='train-scene diagnostic only; no official split reassignment')
    for mode in ('energy', 'coherence', 'both'):
        torch.manual_seed(42)
        model = AcousticEncoder(features=mode).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        history = []
        for step in range(301):
            model.eval()
            if step % 10 == 0:
                with torch.no_grad():
                    def evaluate(batch):
                        o, c = scores(model, batch)
                        return metrics(torch.einsum('bd,bkd->bk', o, c)/.07)
                    row = dict(step=step, fit=evaluate(fit_batch), held=evaluate(held_batch))
                history.append(row)
                print(json.dumps(dict(mode=mode, **row)), flush=True)
                atomic_json(output/'status.json', dict(status='running', stage='overfit_'+mode,
                    step=step, metrics=row, updated_unix=time.time()))
            if step == 300:
                break
            model.train()
            optimizer.zero_grad(set_to_none=True)
            o, c = scores(model, fit_batch)
            loss, _ = contrastive_loss(o, c)
            if not torch.isfinite(loss):
                raise FloatingPointError('Diagnostic nonfinite loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
            optimizer.step()
        torch.save(dict(model=model.state_dict(), mode=mode, diagnostic_only=True), output/f'{mode}.pt')
        result['experiments'].append(dict(mode=mode, history=history))
        atomic_json(output/'results.json', result)
        del model, optimizer
        torch.cuda.empty_cache()
    atomic_json(output/'status.json', dict(status='complete', test_accessed=False, updated_unix=time.time()))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    try:
        run(a.root, a.config, a.output)
    except Exception as e:
        atomic_json(a.output/'status.json', dict(status='failed', error=repr(e), updated_unix=time.time()))
        raise

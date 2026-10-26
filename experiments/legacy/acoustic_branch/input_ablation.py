"""Local, split-frozen input ablation with a persistent sequential GPU queue.

E, E+C, residual envelope, E+C+envelope. Selection uses held-out val loss,
never test. The extra input branch changes parameter count (reported explicitly).
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import fcntl
import html
import json
from pathlib import Path
import time

import torch

from .contracts import atomic_json, immutable_json, sha
from .data import read, load_manifest, check_cache, preflight
from .envelope_cache import build, check
from .train import AcousticEncoder, PairDataset, train, retrieval, requires_envelope

PILOT = Path('AcousticContrastive/pilots/scene_subset_v1')
REL = PILOT/'input_ablation_v1'
MODES = ('energy', 'both', 'envelope', 'all')
LABELS = dict(energy='STFT energy', both='STFT energy + complex coherence',
              envelope='Mic-mean-removed envelope', all='Energy + coherence + envelope')


def publish(out, report):
    summary = read(out/'summary.json')
    report.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in summary['runs']:
        metrics = row.get('best_val_captured_bank', {})
        def value(key):
            return f"{100*metrics[key]:.2f}" if key in metrics else '—'
        rows.append('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in
            (LABELS[row['mode']], row['status'], row['parameters'], row.get('best_epoch', '—'),
             f"{row['best_val_loss']:.4f}" if 'best_val_loss' in row else '—',
             value('recall_0.1m'), value('recall_0.5m'), value('recall_1m'), value('recall_1m_30deg')) )+'</tr>')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>HSSD acoustic input ablation</title><style>body{font:16px system-ui;margin:32px;max-width:1400px;color:#172a42}
table{border-collapse:collapse;width:100%}td,th{padding:12px;border:1px solid #cad5e4;text-align:left}th{background:#edf2fa}
.note{background:#fff4d9;padding:16px;line-height:1.6}a{color:#1264ab}</style><h1>HSSD acoustic input ablation</h1>
<p>Status: STATUS. Refresh this page for progress.</p><div class="note">Train: 12 scenes / 6,387 queries. Val: 3 held-out scenes / 1,704 queries.
<strong>TEST IS UNOPENED.</strong> These recalls use all captured FP poses/yaws per val scene, NOT the dense grid; do not compare them directly to dense-test recalls.
Same queries, 1 positive + 7 negatives, seed 42, AdamW, temperature 0.07, batch 64, max 40 epochs / patience 8.
Best is selected by val eight-candidate loss, not this retrieval table. All embeddings have 128 dimensions.
Extra branches increase parameter counts; this is not a parameter-matched architecture ablation.</div>
<table><thead><tr><th>Inputs</th><th>Status</th><th>Parameters</th><th>Best epoch</th><th>Val loss</th>
<th>R@0.1m %</th><th>R@0.5m %</th><th>R@1m %</th><th>R@1m,30° %</th></tr></thead><tbody>ROWS</tbody></table>
<h2>Definitions</h2><ul><li>Energy: 500–4000 Hz STFT power, fixed-clock 2–128 ms, globally normalized/log-scaled, [6,19,95].
Not the legacy three-band handcrafted descriptor.</li><li>Coherence: adjacent mic complex coherency, early/full windows, [2,6,2,19].</li>
<li>Envelope: subtract instantaneous six-mic mean, bandpass 500–1500 Hz, Hilbert envelope, smooth/bin to 0.5 ms, per-mic max normalize, [6,252].
This does not semantically isolate ceiling echoes.</li></ul><p>Each run saves best/last full-state checkpoints, per-epoch metrics and W&amp;B logs.
<a href="summary.json">Machine-readable results</a></p></html>'''
    temporary = report/'index.pending.html'
    temporary.write_text(page.replace('STATUS', html.escape(summary['status'])).replace('ROWS', ''.join(rows)))
    temporary.replace(report/'index.html')
    atomic_json(report/'summary.json', summary)


def update(out, report, **fields):
    summary = read(out/'summary.json')
    summary.update(fields, updated_unix=time.time())
    atomic_json(out/'summary.json', summary)
    publish(out, report)


def evaluate_best_last(root, config, run):
    queries, digest = load_manifest(root, 'val', config)
    locations, _ = check_cache(root, config, 'val', digest, full=False)
    envelope = check(root, config, 'val')[0] if requires_envelope(config['model']['features']) else None
    dataset = PairDataset(queries, locations, envelope)
    results = {}
    try:
        for name in ('best', 'last'):
            checkpoint = torch.load(run/f'{name}.pt', map_location='cpu', weights_only=False)
            model = AcousticEncoder(**config['model']).cuda()
            model.load_state_dict(checkpoint['model'])
            metrics, examples = retrieval(model, dataset, config, torch.device('cuda'))
            results[name] = dict(epoch=checkpoint['epoch'], metrics=metrics, examples=examples)
            del model
        atomic_json(run/'final_val_retrieval.json', dict(results=results, test_accessed=False,
            scope='captured_pose_bank; not dense localization', checkpoint_selection='val eight-candidate loss'))
        return results
    finally:
        dataset.store.close()


def pipeline(root, report):
    root = Path(root)
    out = root/REL
    out.mkdir(parents=True, exist_ok=True)
    lock = (out/'pipeline.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    fusion = read(root/'ReliabilityFusion/pilots/v1/pipeline_status.json')
    if fusion.get('status') != 'complete':
        raise RuntimeError('GPU reserved for reliability-fusion pipeline until it completes')
    base = read(root/PILOT/'corrected_yup_v2/config.json')
    if base['subset_counts'] != dict(train=12, val=3, test=3):
        raise ValueError('Frozen subset changed')
    ready = preflight(root, base)
    if ready['status'] != 'ready':
        raise ValueError('Original E/C preflight failed: '+json.dumps(ready))
    configs = {}
    for mode in MODES:
        config = copy.deepcopy(base)
        config['model']['features'] = mode
        configs[mode] = config
        immutable_json(out/'configs'/f'{mode}.json', config)
    if not (out/'summary.json').exists():
        immutable_json(out/'summary.json', dict(status='running', started_unix=time.time(), test_accessed=False,
            data=ready, runs=[dict(mode=m, status='pending',
                parameters=sum(p.numel() for p in AcousticEncoder(**configs[m]['model']).parameters())) for m in MODES]))
    publish(out, report)
    def envelope_work():
        for split in ('train', 'val'):
            build(root, base, split, workers=2)
            check(root, base, split)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(envelope_work)
            for index, mode in enumerate(MODES):
                if mode in ('envelope', 'all'):
                    update(out, report, status='awaiting verified envelope cache')
                    future.result()
                run = out/'runs'/mode
                if (run/'status.json').exists() and read(run/'status.json')['status'] == 'complete':
                    pass
                else:
                    update(out, report, status='training '+mode)
                    rows = read(out/'summary.json')['runs']
                    rows[index]['status'] = 'training'
                    update(out, report, runs=rows)
                    resume = run/'last.pt' if (run/'last.pt').exists() else None
                    train(root, configs[mode], run, resume)
                metrics = evaluate_best_last(root, configs[mode], run)
                rows = read(out/'summary.json')['runs']
                rows[index].update(status='complete', best_epoch=metrics['best']['epoch'],
                    last_epoch=metrics['last']['epoch'], best_val_loss=read(run/'status.json')['best_val_loss'],
                    best_val_captured_bank=metrics['best']['metrics'], last_val_captured_bank=metrics['last']['metrics'])
                update(out, report, runs=rows)
            future.result()
        update(out, report, status='complete')
    except Exception as error:
        update(out, report, status='failed', error=repr(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    pipeline(args.root, args.report)


if __name__ == '__main__':
    main()

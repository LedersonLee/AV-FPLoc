"""Isolate CPU/CUDA preprocessing versus model differences on one fixed Val query.
No tuning, training, full-cohort accuracy or timing conclusions.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import default_collate
from augment import prepare
from audio import chirp, echo_wave, spectrogram
from dataset import Observations
from evaluate_baseline_fields import sha
from matching import laplace_field, proposals
from model import AVLoc


def delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    return dict(max_abs=float(abs(a-b).max()), mean_abs=float(abs(a-b).mean()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    run = a.study/'runs/A2_seed42'
    assert (run/'complete.json').is_file()
    best = json.loads((run/'best.json').read_text())
    cp = run/best['path']
    ck = torch.load(cp, map_location='cpu', weights_only=False)
    for name in ['model.py', 'dataset.py', 'augment.py', 'audio.py']:
        assert sha(Path(__file__).with_name(name)) == ck['protocol']['source_sha256'][name]
    assert sha(a.study/'data_v1/manifest.json') == ck['protocol']['input_manifest_sha256']
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(.25)
    ds = Observations(a.study/'data_v1', 'val')
    assert len(ds.records) == 19 and len(ds) == 9965
    index = int(np.random.default_rng(20261111).integers(ds.records[0]['queries']))
    batch = default_collate([ds[index]])
    sid, qid = batch['scene'][0], int(batch['query_id'][0])
    pm = json.loads((a.study/'predictions/A2_seed42/manifest.json').read_text())
    assert pm['complete'] and pm['checkpoint_sha256'] == sha(cp)
    pr = next(r for r in pm['records'] if r['scene'] == sid)
    pp = a.study/'predictions/A2_seed42'/pr['file']
    assert sha(pp) == pr['sha256']
    with np.load(pp) as z:
        i = int(np.flatnonzero(z['query_ids'] == qid)[0])
        reference = (z['AV_mu'][i].copy(), z['AV_scale'][i].copy())
    bm = json.loads((a.study/'banks_v1/manifest.json').read_text())
    bp = a.study/f'banks_v1/{sid}/grid_20cm.npz'
    assert sha(bp) == next(r for r in bm['records'] if r['scene'] == sid)['files'][bp.name]
    with np.load(bp) as z:
        xy, world = z['xy'].copy(), z['world_rays'].copy()
    field = laplace_field(world, *reference)
    refids, reftop = proposals(field, xy), int(np.argmax(field))
    models = {d: AVLoc('A2').to(d).eval() for d in ['cpu', 'cuda']}
    for model in models.values():
        model.load_state_dict(ck['model'], strict=True)
    del ck
    outputs, rows = {}, []
    with torch.inference_mode():
        prepared = {d: prepare(batch, d, False)[:2] for d in models}
        for prepdevice in models:
            for modeldevice, model in models.items():
                rgb, spec = (t.to(modeldevice) for t in prepared[prepdevice])
                prediction = tuple(t[0].cpu().numpy().copy() for t in model(rgb, spec)['AV'])
                outputs[prepdevice, modeldevice] = prediction
                field = laplace_field(world, *prediction)
                ids = proposals(field, xy)
                rows.append(dict(prepare_device=prepdevice, model_device=modeldevice,
                                 mu_difference_from_saved=delta(prediction[0], reference[0]),
                                 scale_difference_from_saved=delta(prediction[1], reference[1]),
                                 top1_matches_saved=int(np.argmax(field)) == reftop,
                                 top20_ids_equal=bool(np.array_equal(ids, refids))))
        arrays = {d: [t.cpu().numpy() for t in prepared[d]] for d in models}
        waves = {d: echo_wave(batch['rir'].to(d)) for d in models}
        # Same GPU-generated echo fed to both spectrogram backends isolates STFT/phase code.
        same_echo_specs = {d: spectrogram(waves['cuda'].to(d)).cpu().numpy() for d in models}
        audio_checks = dict(chirp=delta(chirp('cpu', torch.float32).numpy(), chirp('cuda', torch.float32).cpu().numpy()),
                            echo=delta(waves['cpu'].numpy(), waves['cuda'].cpu().numpy()),
                            spectrogram_same_echo=delta(same_echo_specs['cpu'], same_echo_specs['cuda']))
    contrasts = {}
    for name, left, right in [('preprocessing_at_fixed_cuda_model', ('cpu','cuda'), ('cuda','cuda')),
                              ('model_at_fixed_cuda_preprocessing', ('cuda','cpu'), ('cuda','cuda'))]:
        contrasts[name] = dict(mu=delta(outputs[left][0], outputs[right][0]),
                               scale=delta(outputs[left][1], outputs[right][1]))
    result = dict(complete=True, scene=sid, query_id=qid, rows=rows, contrasts=contrasts,
                  rgb_difference=delta(arrays['cpu'][0], arrays['cuda'][0]),
                  spectrogram_difference=delta(arrays['cpu'][1], arrays['cuda'][1]), audio_checks=audio_checks,
                  checkpoint_sha256=sha(cp), prediction_sha256=sha(pp), bank_sha256=sha(bp),
                  source_sha256={n:sha(Path(__file__).with_name(n)) for n in ['audit_runtime_backend.py','audio.py','model.py','augment.py','dataset.py','matching.py']},
                  torch_version=torch.__version__, test_accessed=False, training=False,
                  scope='One predetermined Val query; backend portability diagnostic only. No tolerance changes or benchmark revisions.')
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(rows=rows, contrasts=contrasts, audio_checks=audio_checks)), flush=True)


if __name__ == '__main__':
    main()

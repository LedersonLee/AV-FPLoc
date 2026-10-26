#!/usr/bin/env python3
"""Fresh shared observed/FP encoder, query-specific hard-negative InfoNCE.

No UnLoc weights/RGB/visual likelihood enters this model. Mining only supplies
train candidate identities. Training cannot start until BOTH raw-data and
feature-cache verification pass, including genuinely held-out val scenes.
"""
import argparse
from collections import OrderedDict, defaultdict
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import uuid

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .data import (read, preflight, load_manifest, check_cache, unique_refs)
from .features import roll_features
from .contracts import atomic_json, immutable_json, sha

FEATURE_STREAMS = dict(energy=('energy',), coherence=('coherence',),
    envelope=('envelope',), both=('energy','coherence'),
    energy_envelope=('energy','envelope'), coherence_envelope=('coherence','envelope'),
    all=('energy','coherence','envelope'))


def requires_envelope(features):
    return 'envelope' in FEATURE_STREAMS[features]


class AcousticEncoder(nn.Module):
    """One weight-shared encoder for both domains; no mic-order invariance.

    GroupNorm avoids per-domain running statistics. Coarse time bins are kept
    in the energy branch (not globally pooled away). No learned yaw prior.
    """
    def __init__(self, features='both', width=32, embedding_dim=128):
        super().__init__()
        if features not in FEATURE_STREAMS or width < 8 or width % 8:
            raise ValueError('Invalid architecture/width')
        self.mode = features
        streams = FEATURE_STREAMS[features]
        self.energy = nn.Sequential(
            nn.Conv2d(6, width, 3, padding=1), nn.GroupNorm(8, width), nn.GELU(),
            nn.Conv2d(width, width*2, 3, stride=2, padding=1), nn.GroupNorm(8, width*2), nn.GELU(),
            nn.Conv2d(width*2, width*3, 3, stride=2, padding=1), nn.GroupNorm(8, width*3), nn.GELU(),
            nn.AdaptiveAvgPool2d((2, 16)), nn.Flatten(), nn.Linear(width*3*32, embedding_dim), nn.GELU()
        ) if 'energy' in streams else None
        self.coherence = nn.Sequential(
            nn.Conv1d(24, width, 3, padding=1), nn.GroupNorm(8, width), nn.GELU(),
            nn.Conv1d(width, width*2, 3, padding=1), nn.GroupNorm(8, width*2), nn.GELU(),
            nn.AdaptiveAvgPool1d(4), nn.Flatten(), nn.Linear(width*2*4, embedding_dim), nn.GELU()
        ) if 'coherence' in streams else None
        self.envelope = nn.Sequential(
            nn.Conv1d(6, width, 5, padding=2), nn.GroupNorm(8, width), nn.GELU(),
            nn.Conv1d(width, width*2, 5, stride=2, padding=2), nn.GroupNorm(8, width*2), nn.GELU(),
            nn.AdaptiveAvgPool1d(16), nn.Flatten(), nn.Linear(width*2*16, embedding_dim), nn.GELU()
        ) if 'envelope' in streams else None
        count = len(streams)
        self.projection = nn.Sequential(nn.Linear(embedding_dim*count, embedding_dim), nn.LayerNorm(embedding_dim))

    def forward(self, energy, coherence, envelope=None):
        values = []
        if self.energy is not None:
            values.append(self.energy(energy))
        if self.coherence is not None:
            values.append(self.coherence(coherence.flatten(1, 3)))
        if self.envelope is not None:
            if envelope is None:
                raise ValueError('Envelope model requires verified envelope inputs')
            values.append(self.envelope(envelope))
        return F.normalize(self.projection(torch.cat(values, dim=-1)), dim=-1)


def contrastive_loss(observed, candidates, temperature=.07):
    """L=-log(exp(sim(o,p+)/tau)/sum_{j=0..7} exp(sim(o,pj)/tau)).

    No between-scene shortcuts or unlabeled/nearby in-batch false negatives.
    Correct FP candidate is slot 0; orientation negatives keep channel order.
    """
    if temperature <= 0 or candidates.ndim != 3 or candidates.shape[1] != 8:
        raise ValueError('Expected eight candidates and positive temperature')
    logits = torch.einsum('bd,bkd->bk', observed.float(), candidates.float()) / temperature
    return F.cross_entropy(logits, torch.zeros(len(logits), dtype=torch.long, device=logits.device)), logits


class FeatureStore:
    """Bounded mmap handles; each worker opens its own cache lazily."""
    def __init__(self, locations, max_open=16, envelope_locations=None):
        self.locations = locations
        self.max_open = max_open
        self.maps = OrderedDict()
        self.envelope_locations = envelope_locations

    def close(self):
        for pair in self.maps.values():
            for array in pair:
                array._mmap.close()
        self.maps.clear()

    def __getstate__(self):
        return dict(locations=self.locations, max_open=self.max_open, maps=OrderedDict(),
                    envelope_locations=self.envelope_locations)

    def get(self, r):
        directory, index = self.locations[r['path']]
        if directory not in self.maps:
            self.maps[directory] = tuple(np.load(Path(directory)/name, mmap_mode='r', allow_pickle=False)
                                         for name in ('energy.npy', 'coherence.npy'))
            while len(self.maps) > self.max_open:
                _, pair = self.maps.popitem(last=False)
                for array in pair:
                    array._mmap.close()
        self.maps.move_to_end(directory)
        e, c = (np.array(a[index], dtype=np.float32) for a in self.maps[directory])
        e, c = roll_features(e, c, r['roll'])
        if not np.isfinite(e).all() or not np.isfinite(c).all():
            raise ValueError('Nonfinite cached feature')
        values = (torch.from_numpy(e), torch.from_numpy(c))
        if self.envelope_locations is not None:
            envelope_path, envelope_index = self.envelope_locations[r['path']]
            if envelope_path not in self.maps:
                self.maps[envelope_path] = (np.load(envelope_path, mmap_mode='r', allow_pickle=False),)
                while len(self.maps) > self.max_open:
                    _, pair = self.maps.popitem(last=False)
                    for array in pair:
                        array._mmap.close()
            self.maps.move_to_end(envelope_path)
            envelope = np.roll(np.array(self.maps[envelope_path][0][envelope_index], dtype=np.float32),
                               int(r['roll']), axis=0)
            if not np.isfinite(envelope).all():
                raise ValueError('Nonfinite envelope feature')
            values += (torch.from_numpy(envelope),)
        return values


class PairDataset(Dataset):
    def __init__(self, queries, locations, envelope_locations=None):
        self.queries = queries
        self.store = FeatureStore(locations, envelope_locations=envelope_locations)

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, index):
        q = self.queries[index]
        observed = self.store.get(q['observed'])
        candidates = [self.store.get(r) for r in q['candidates']]
        return (*observed, *(torch.stack([v[i] for v in candidates]) for i in range(len(observed))))


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)


def loader(dataset, config, epoch, training):
    settings = config['training']
    generator = torch.Generator().manual_seed(config['seed'] + epoch)
    return DataLoader(dataset, batch_size=settings['batch_size'], shuffle=training, generator=generator,
        num_workers=settings['workers'], persistent_workers=False, worker_init_fn=seed_worker,
        pin_memory=settings['device'].startswith('cuda'), drop_last=False)


def pass_epoch(model, batches, device, temperature, optimizer=None, scaler=None, grad_clip=1, amp=False):
    training = optimizer is not None
    model.train(training)
    totals = np.zeros(4, dtype=np.float64)
    with torch.set_grad_enabled(training):
        for tensors in batches:
            if len(tensors) not in (4, 6):
                raise ValueError('Expected two or three verified feature streams')
            tensors = [v.to(device, non_blocking=True) for v in tensors]
            streams = len(tensors)//2
            observed_inputs, candidate_inputs = tensors[:streams], tensors[streams:]
            batch, count = candidate_inputs[0].shape[:2]
            if training:
                optimizer.zero_grad(set_to_none=True)
            amp_dtype = torch.bfloat16 if device.type == 'cuda' and torch.cuda.is_bf16_supported() else torch.float16
            with torch.autocast(device_type=device.type, enabled=amp, dtype=amp_dtype):
                observed = model(*observed_inputs)
                candidate = model(*(v.flatten(0, 1) for v in candidate_inputs)).reshape(batch, count, -1)
                loss, logits = contrastive_loss(observed, candidate, temperature)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss; checkpoint not advanced')
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                # FP16 overflow is handled by GradScaler skipping the step.
                # BF16/FP32 have no scaling and must fail on a nonfinite grad.
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=not scaler.is_enabled())
                scaler.step(optimizer)
                scaler.update()
            order = logits.argsort(dim=-1, descending=True)
            rank = (order == 0).nonzero(as_tuple=False)[:, 1] + 1
            totals += [float(loss.detach()) * batch, float((rank == 1).sum()), float((rank <= 5).sum()), float((1/rank.float()).sum())]
    size = len(batches.dataset)
    return dict(loss=totals[0]/size, sampled_pool_top1=totals[1]/size,
                sampled_pool_top5=totals[2]/size, sampled_pool_mrr=totals[3]/size)


@torch.no_grad()
def encode_refs(model, store, refs, device, batch_size):
    values = []
    for offset in range(0, len(refs), batch_size):
        batch = [store.get(r) for r in refs[offset:offset+batch_size]]
        inputs = [torch.stack([v[i] for v in batch]).to(device) for i in range(len(batch[0]))]
        values.append(model(*inputs).cpu())
    return torch.cat(values)


@torch.no_grad()
def retrieval(model, dataset, config, device):
    """All captured FP pose/yaw views WITHIN each val scene, not a dense grid.

    Never report seven-negative-pool recall as global acoustic localization.
    """
    model.eval()
    scenes = defaultdict(list)
    for q in dataset.queries:
        scenes[q['scene_id']].append(q)
    xy_errors, yaw_errors, ranks, examples = [], [], [], []
    for scene, queries in sorted(scenes.items()):
        bank = [q['candidates'][0] for q in queries]
        identities = [(tuple(r['xyz']), r['yaw'] % 360) for r in bank]
        if len(identities) != len(set(identities)):
            raise ValueError('Duplicate captured-pose bank identity')
        candidate = encode_refs(model, dataset.store, bank, device, config['training']['batch_size'])
        observed = encode_refs(model, dataset.store, [q['observed'] for q in queries], device, config['training']['batch_size'])
        scores = (observed @ candidate.T).numpy()
        winner = scores.argmax(axis=1)
        gt_xyz = np.array([r['xyz'] for r in bank])
        gt_yaw = np.array([r['yaw'] for r in bank])
        xy = np.linalg.norm((gt_xyz[winner] - gt_xyz)[:, [0, 2]], axis=1)
        yaw = abs((gt_yaw[winner] - gt_yaw + 180) % 360 - 180)
        # Exact ties use average rank for MRR (no optimistic hidden tie break).
        positive = scores.diagonal()[:, None]
        rank = 1 + (scores > positive).sum(axis=1) + .5*((scores == positive).sum(axis=1)-1)
        xy_errors.extend(xy.tolist())
        yaw_errors.extend(yaw.tolist())
        ranks.extend(rank.tolist())
        for i in (0, len(queries)//2):
            examples.append(dict(scene_id=scene, sample_index=queries[i]['sample_index'], image_path=queries[i]['image_path'],
                gt_xyz=gt_xyz[i].tolist(), gt_yaw=float(gt_yaw[i]), predicted_xyz=gt_xyz[winner[i]].tolist(),
                predicted_yaw=float(gt_yaw[winner[i]]), xy_error_m=float(xy[i]), yaw_error_deg=float(yaw[i]),
                candidate_xz=gt_xyz[:, [0, 2]].tolist(), scores=scores[i].tolist()))
    xy, yaw = np.asarray(xy_errors), np.asarray(yaw_errors)
    metrics = {f'recall_{threshold:g}m': float((xy <= threshold).mean()) for threshold in (.1, .5, 1)}
    metrics.update(recall_1m_30deg=float(((xy <= 1) & (yaw <= 30)).mean()),
        median_xy_error_m=float(np.median(xy)), median_yaw_error_deg=float(np.median(yaw)),
        mrr=float(np.mean(1/np.asarray(ranks))))
    return metrics, examples


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best, bad_epochs, config, contract):
    value = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
        scaler=scaler.state_dict(), epoch=epoch, best=best, bad_epochs=bad_epochs, config=config, contract=contract,
        rng_python=random.getstate(), rng_numpy=np.random.get_state(), rng_torch=torch.get_rng_state(),
        rng_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    path = Path(path)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def restore_checkpoint(path, model, optimizer, scheduler, scaler, config, contract):
    # Only trust locally generated full-state checkpoints (contains Python RNG).
    value = torch.load(path, map_location='cpu', weights_only=False)
    if value['config'] != config or value['contract'] != contract:
        raise ValueError('Resume config/data/code contract changed; new run required')
    model.load_state_dict(value['model'])
    optimizer.load_state_dict(value['optimizer'])
    scheduler.load_state_dict(value['scheduler'])
    scaler.load_state_dict(value['scaler'])
    random.setstate(value['rng_python'])
    np.random.set_state(value['rng_numpy'])
    torch.set_rng_state(value['rng_torch'])
    if value['rng_cuda'] is not None:
        torch.cuda.set_rng_state_all(value['rng_cuda'])
    return value['epoch'], value['best'], value['bad_epochs']


def figures(root, examples):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    from .data import safe_path
    plots = []
    for e in examples[:6]:
        scene = e['scene_id']
        meta = read(root/'2DFP'/scene/'floorplan_metadata.json')
        image = np.asarray(Image.open(root/'2DFP'/scene/'map.png').convert('L'))
        s = meta['map_spacing_m']
        bounds = meta['world_bounds_m']
        extent = [bounds['x_min']-.5*s, bounds['x_min']+(image.shape[1]-.5)*s,
                  bounds['z_min']+(image.shape[0]-.5)*s, bounds['z_min']-.5*s]
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        axes[0].imshow(Image.open(safe_path(root, e['image_path'])))
        axes[0].axis('off')
        axes[1].imshow(image, cmap='gray', vmin=0, vmax=255, extent=extent)
        points = np.asarray(e['candidate_xz'])
        # Max over bank yaws at each captured XY; NOT a dense XY likelihood.
        xy_scores = {}
        for p, score in zip(points, e['scores']):
            xy_scores[tuple(p)] = max(xy_scores.get(tuple(p), -np.inf), score)
        xy = np.array(list(xy_scores))
        score = np.array(list(xy_scores.values()))
        color = (score-score.min()) / max(float(np.ptp(score)), 1e-9)
        axes[1].scatter(xy[:, 0], xy[:, 1], c=color, cmap='coolwarm', vmin=0, vmax=1, s=10)
        for xyz, yaw, label, color in [(e['gt_xyz'], e['gt_yaw'], 'GT', 'dodgerblue'),
                                       (e['predicted_xyz'], e['predicted_yaw'], 'Acoustic MAP', 'limegreen')]:
            radians = math.radians(yaw)
            axes[1].arrow(xyz[0], xyz[2], .6*math.cos(radians), .6*math.sin(radians),
                          color=color, width=.04, head_width=.2, length_includes_head=True, label=label)
        axes[1].legend()
        axes[1].set_title('Captured-pose bank scores · +Z downward')
        fig.suptitle(f'{scene} / Q{e["sample_index"]} · XY {e["xy_error_m"]:.2f}m / yaw {e["yaw_error_deg"]:.0f}°')
        fig.tight_layout()
        plots.append(fig)
    return plots


def validate_training_config(config):
    t = config['training']
    if config['loss']['negative_count'] != 7 or config['loss']['temperature'] <= 0:
        raise ValueError('Invalid loss contract')
    if min(t['epochs'], t['batch_size'], t['cpu_threads'], t['early_stop_patience'], t['evaluate_every']) < 1 or t['workers'] < 0:
        raise ValueError('Invalid training schedule')
    if t['lr'] <= 0 or t['weight_decay'] < 0 or t['grad_clip'] <= 0:
        raise ValueError('Invalid optimizer settings')
    if config['wandb']['mode'] not in ('online', 'offline', 'disabled'):
        raise ValueError('Invalid W&B mode')


def train(root, config, run_dir, resume=None):
    validate_training_config(config)
    report = preflight(root, config)
    if report['status'] != 'ready':
        raise RuntimeError('Training blocked: ' + json.dumps(report['blockers']))
    root, run_dir = Path(root), Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    lock = (run_dir/'train.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if resume is None and (run_dir/'last.pt').exists():
        raise FileExistsError('Existing run: use --resume; never overwrite a trained model')
    # Feature/cache and code identity are part of exact epoch-boundary resume.
    code = {name: sha(Path(__file__).with_name(name)) for name in
            ('train.py', 'data.py', 'features.py', 'contracts.py')}
    envelope_contract = {}
    if requires_envelope(config['model']['features']):
        from .envelope_cache import check as check_envelope
        code['envelope_cache.py'] = sha(Path(__file__).with_name('envelope_cache.py'))
        for split in ('train', 'val'):
            _, envelope_contract[split] = check_envelope(root, config, split)
    bf16 = bool(config['training']['amp'] and torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    contract = dict(data=report, code=code, torch_version=str(torch.__version__),
                    precision='bf16' if bf16 else 'fp16' if config['training']['amp'] and torch.cuda.is_available() else 'fp32',
        selection='val eight-candidate loss; test unopened', retrieval_scope='captured_pose_bank',
        envelope=envelope_contract)
    immutable_json(run_dir/'contract.json', contract)
    immutable_json(run_dir/'config.json', config)
    torch.set_num_threads(config['training']['cpu_threads'])
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(config['training']['device'])
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; no silent CPU full training')
    model = AcousticEncoder(**config['model']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['lr'], weight_decay=config['training']['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['training']['epochs'])
    amp = bool(config['training']['amp'] and device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=amp and not bf16)
    datasets = {}
    for split in ('train', 'val'):
        queries, digest = load_manifest(root, split, config)
        locations, _ = check_cache(root, config, split, digest, full=False)
        envelope_locations = check_envelope(root, config, split)[0] if envelope_contract else None
        datasets[split] = PairDataset(queries, locations, envelope_locations)
    epoch, best, bad_epochs = 0, float('inf'), 0
    if resume is not None:
        epoch, best, bad_epochs = restore_checkpoint(resume, model, optimizer, scheduler, scaler, config, contract)
    wandb_run = None
    if config['wandb']['mode'] != 'disabled':
        import wandb
        id_path = run_dir/'wandb_id.json'
        if id_path.exists():
            wandb_id = read(id_path)['id']
        else:
            wandb_id = wandb.util.generate_id()
            immutable_json(id_path, dict(id=wandb_id))
        wandb_run = wandb.init(project=config['wandb']['project'], entity=config['wandb']['entity'],
            mode=config['wandb']['mode'], id=wandb_id, resume='allow', dir=str(run_dir), config=config,
            name=config['wandb'].get('name'), group=config['wandb'].get('group'))
    try:
        for epoch in range(epoch+1, config['training']['epochs']+1):
            train_metrics = pass_epoch(model, loader(datasets['train'], config, epoch, True), device,
                config['loss']['temperature'], optimizer, scaler, config['training']['grad_clip'], amp)
            val_metrics = pass_epoch(model, loader(datasets['val'], config, epoch, False), device, config['loss']['temperature'])
            metrics = {f'train/{k}': v for k, v in train_metrics.items()}
            metrics.update({f'val/{k}': v for k, v in val_metrics.items()})
            metrics.update(epoch=epoch, lr=optimizer.param_groups[0]['lr'])
            improved = val_metrics['loss'] < best-config['training']['min_delta']
            best, bad_epochs = (val_metrics['loss'], 0) if improved else (best, bad_epochs+1)
            scheduler.step()
            for name in ['last.pt'] + (['best.pt'] if improved else []):
                save_checkpoint(run_dir/name, model, optimizer, scheduler, scaler, epoch, best, bad_epochs, config, contract)
            log = dict(metrics)
            if epoch % config['training']['evaluate_every'] == 0:
                bank_metrics, examples = retrieval(model, datasets['val'], config, device)
                metrics.update({f'val_captured_bank/{k}': v for k, v in bank_metrics.items()})
                log.update(metrics)
                atomic_json(run_dir/f'val_epoch{epoch:03d}.json', dict(metrics=metrics, examples=examples,
                    scope='all captured FP views per held-out val scene; NOT dense floorplan likelihood'))
                save_checkpoint(run_dir/f'epoch{epoch:03d}.pt', model, optimizer, scheduler, scaler, epoch, best, bad_epochs, config, contract)
                if wandb_run is not None:
                    import matplotlib.pyplot as plt
                    plots = figures(root, examples)
                    log['val_captured_bank/examples'] = [wandb.Image(fig) for fig in plots]
                    for fig in plots:
                        plt.close(fig)
            atomic_json(run_dir/'status.json', dict(status='running', epoch=epoch, best_val_loss=best,
                bad_epochs=bad_epochs, metrics=metrics, test_accessed=False))
            atomic_json(run_dir/f'metrics_epoch{epoch:03d}.json', metrics)
            if wandb_run is not None:
                wandb_run.log(log, step=epoch)
            print(json.dumps(metrics), flush=True)
            if bad_epochs >= config['training']['early_stop_patience']:
                break
        atomic_json(run_dir/'status.json', dict(status='complete', epoch=epoch, best_val_loss=best,
            stopped_early=epoch < config['training']['epochs'], test_accessed=False))
    finally:
        for dataset in datasets.values():
            dataset.store.close()
        if wandb_run is not None:
            wandb_run.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    train(args.root, read(args.config), args.run_dir, args.resume)


if __name__ == '__main__':
    main()

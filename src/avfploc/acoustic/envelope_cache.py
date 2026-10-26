"""Verified additive residual-envelope cache; never modifies the E/C cache.

Matches planar_probe.arrival_features on the six-mic mean-subtracted RIR.
No test records are loaded here. Roll channels only AFTER feature extraction.
"""
import argparse
import fcntl
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy import ndimage, signal

from .contracts import atomic_json, immutable_json, sha
from .data import read, load_manifest, unique_refs, checked_rir, check_cache

DEFINITION = dict(version='mic_mean_removed_envelope_v1', sample_rate_hz=48000,
    direct_sample=7, start_ms=2, end_ms=128, band_hz=[500, 1500], filter_order=3,
    padding_samples=1024, gaussian_sigma_samples=12, bin_samples=24,
    normalization='per-mic maximum; zero below 1e-15', dtype='float16')


def extract(rir):
    rir = np.asarray(rir, dtype=np.float64)
    if rir.ndim != 2 or rir.shape[0] != 6 or not np.isfinite(rir).all():
        raise ValueError('Expected finite six-channel RIR')
    residual = rir - rir.mean(axis=0, dtype=np.float64)
    first, last = 103, 6151  # direct sample 7 + [2ms, 128ms] at 48kHz
    x = np.zeros((6, last), dtype=np.float64)
    n = min(last, rir.shape[-1])
    if n <= first:
        raise ValueError('RIR does not cover reflection window')
    x[:, first:n] = residual[:, first:n]
    x = np.pad(x, ((0, 0), (1024, 1024)))
    sos = signal.butter(3, [500, 1500], btype='bandpass', fs=48000, output='sos')
    x = signal.sosfiltfilt(sos, x, axis=-1)
    envelope = abs(signal.hilbert(x, axis=-1))[:, 1024+first:1024+last]
    envelope = ndimage.gaussian_filter1d(envelope, 12, axis=-1)
    envelope = envelope.reshape(6, 252, 24).mean(-1)
    maximum = envelope.max(-1, keepdims=True)
    return np.divide(envelope, maximum, out=np.zeros_like(envelope), where=maximum > 1e-15).astype(np.float32)


def directory(root, config, split):
    if split not in ('train', 'val'):
        raise ValueError('Ablation cache cannot access test')
    identity = dict(definition=DEFINITION, subset_sha256=config['scene_subset_sha256'],
                    corrected=config['corrected_extrusion'], split_sha256=config['split_sha256'])
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    return Path(root)/'AcousticContrastive/features'/key/'residual_envelope'/split


def build(root, config, split, workers=3):
    queries, digest = load_manifest(root, split, config)
    # Require the original verified cache/manifest lineage before adding a stream.
    locations, source_sha = check_cache(root, config, split, digest)
    refs = unique_refs(queries)
    if set(refs) != set(locations):
        raise ValueError('Original cache reference coverage differs')
    dest = directory(root, config, split)
    dest.mkdir(parents=True, exist_ok=True)
    lock = (dest/'build.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source_records = {}
    for folder, _ in set(locations.values()):
        if folder not in source_records:
            source_records[folder] = read(Path(folder)/'receipt.json')
    expected_raw = {p: source_records[folder]['records'][i]['raw_sha256']
                    for p, (folder, i) in locations.items()}
    contract = dict(definition=DEFINITION, manifest_sha256=digest, source_index_sha256=source_sha,
                    split=split, split_sha256=config['split_sha256'], test_accessed=False)
    immutable_json(dest/'contract.json', contract)
    items = sorted(refs.items())
    shards = []
    def work(item):
        path, ref = item
        rir, receipt = checked_rir(root, path, ref, config)
        if receipt['raw_sha256'] != expected_raw[path]:
            raise ValueError('Envelope and E/C features refer to different raw arrays')
        return extract(rir), receipt
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for offset in range(0, len(items), 256):
            chunk = items[offset:offset+256]
            folder = dest/f'shard{offset//256:05d}'
            folder.mkdir(exist_ok=True)
            receipt_path = folder/'receipt.json'
            paths = [p for p, _ in chunk]
            if receipt_path.exists():
                receipt = read(receipt_path)
                if (receipt['paths'] != paths or receipt['contract_sha256'] != sha(dest/'contract.json')
                    or sha(folder/'envelope.npy') != receipt['sha256']
                    or any(r['raw_sha256'] != expected_raw[p] for p, r in zip(paths, receipt['raw_receipts']))):
                    raise ValueError('Committed envelope shard changed')
            else:
                values = list(pool.map(work, chunk))
                temporary = folder/'envelope.pending.npy'
                np.save(temporary, np.stack([v[0] for v in values]).astype(np.float16), allow_pickle=False)
                temporary.replace(folder/'envelope.npy')
                receipt = dict(paths=paths, raw_receipts=[v[1] for v in values],
                    contract_sha256=sha(dest/'contract.json'), sha256=sha(folder/'envelope.npy'))
                immutable_json(receipt_path, receipt)
            shards.append(dict(path=str(receipt_path.relative_to(root)), sha256=sha(receipt_path)))
            atomic_json(dest/'status.json', dict(status='running', arrays=min(offset+256, len(items)),
                expected=len(items), split=split, workers=workers, updated_unix=time.time()))
    immutable_json(dest/'index.json', dict(contract=contract, shards=shards, arrays=len(items), status='complete'))
    atomic_json(dest/'status.json', dict(status='complete', arrays=len(items), split=split, updated_unix=time.time()))


def check(root, config, split):
    queries, digest = load_manifest(root, split, config)
    source_locations, source_sha = check_cache(root, config, split, digest, full=False)
    if config.get('cache_envelope'):
        # Full-scene extraction computes all three inputs in the same raw read.
        # Each joint shard pins a single raw receipt for all three modalities.
        locations = {}
        for folder in sorted({p for p, _ in source_locations.values()}):
            receipt = read(Path(folder)/'receipt.json')
            path = Path(folder)/'envelope.npy'
            if receipt.get('envelope_definition') != DEFINITION or sha(path) != receipt['files'].get('envelope.npy'):
                raise ValueError('Joint envelope definition/checksum changed')
            array = np.load(path, mmap_mode='r', allow_pickle=False)
            if array.shape != (len(receipt['paths']), 6, 252) or array.dtype != np.float16:
                raise ValueError('Joint envelope tensor shape/dtype changed')
            array._mmap.close()
            for i, relative in enumerate(receipt['paths']):
                if relative in locations:
                    raise ValueError('Duplicate joint envelope reference')
                locations[relative] = (str(path), i)
        if set(locations) != set(unique_refs(queries)):
            raise ValueError('Joint envelope coverage mismatch')
        return locations, source_sha
    dest = directory(root, config, split)
    index = read(dest/'index.json')
    contract = dict(definition=DEFINITION, manifest_sha256=digest, source_index_sha256=source_sha,
                    split=split, split_sha256=config['split_sha256'], test_accessed=False)
    if index['status'] != 'complete' or index['contract'] != contract or read(dest/'contract.json') != contract:
        raise ValueError('Stale/incomplete envelope cache')
    locations = {}
    for entry in index['shards']:
        path = Path(root)/entry['path']
        if sha(path) != entry['sha256']:
            raise ValueError('Envelope receipt changed')
        receipt = read(path)
        if receipt['contract_sha256'] != sha(dest/'contract.json') or sha(path.parent/'envelope.npy') != receipt['sha256']:
            raise ValueError('Envelope data changed')
        array = np.load(path.parent/'envelope.npy', mmap_mode='r', allow_pickle=False)
        if array.shape != (len(receipt['paths']), 6, 252) or array.dtype != np.float16:
            raise ValueError('Envelope tensor shape/dtype changed')
        array._mmap.close()
        for i, relative in enumerate(receipt['paths']):
            if relative in locations:
                raise ValueError('Duplicate envelope reference')
            locations[relative] = (str(path.parent/'envelope.npy'), i)
    if set(locations) != set(unique_refs(queries)) or len(locations) != index['arrays']:
        raise ValueError('Envelope coverage mismatch')
    return locations, sha(dest/'index.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--split', choices=['train', 'val'], required=True)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    build(args.root, read(args.config), args.split, args.workers)
    print(dict(status='verified', split=args.split, arrays=len(check(args.root, read(args.config), args.split)[0])), flush=True)


if __name__ == '__main__':
    main()

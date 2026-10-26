#!/usr/bin/env python3
"""Split-locked manifests, independent val preparation, verified feature shards.

Commands never start model training. Existing train RIR indices are read-only.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import time
import uuid

import numpy as np

from .contracts import atomic_json, immutable_json, sha, recipe, PROTOCOL, RING
from .features import extract, feature_id, validate_config

TRAIN_SOURCE = 'RIR/manifests/hard_negatives_epoch020_v1'
DEST = 'AcousticContrastive/manifests/v1'


def read(path):
    return json.loads(Path(path).read_text())


def safe_path(root, relative):
    root = Path(root).resolve()
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Unsafe dataset-relative path')
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Dataset symlink escapes root')
    return path


def splits(root, config):
    path = Path(root) / 'manifests/scene_splits_v1.json'
    data = read(path)
    if sha(path) != config['split_sha256']:
        raise ValueError('Frozen scene split hash changed')
    all_ids = [s for k in ('train', 'val', 'test') for s in data[k]]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError('Scene split leakage')
    if {k: len(data[k]) for k in ('train', 'val', 'test')} != config['expected_scenes']:
        raise ValueError('Scene split coverage changed')
    if config.get('scene_subset_path'):
        subset_path = safe_path(root, config['scene_subset_path'])
        if sha(subset_path) != config['scene_subset_sha256']:
            raise ValueError('Frozen subset hash changed')
        subset = read(subset_path)
        if subset['parent_split_sha256'] != config['split_sha256']:
            raise ValueError('Subset parent split changed')
        selected = subset['scenes']
        for split in ('train', 'val', 'test'):
            if not selected[split] or len(selected[split]) != len(set(selected[split])) or not set(selected[split]) <= set(data[split]):
                raise ValueError('Subset scene leakage/duplicate/empty split')
        if {k: len(selected[k]) for k in selected} != config['subset_counts']:
            raise ValueError('Subset counts changed')
        return selected
    return data


def ref(path, roll, xyz, yaw, kind='positive'):
    return dict(path=path, roll=int(roll), xyz=list(xyz), yaw=float(yaw), kind=kind)


def validate_query(q, split, allowed):
    scene = q['scene_id']
    if q['split'] != split or scene not in allowed:
        raise ValueError('Cross-scene/split leakage in query')
    if len(q['candidates']) != 8:
        raise ValueError('Expected one positive plus seven negatives')
    seen = set()
    for role, r in [('Observed', q['observed'])] + [('FP', r) for r in q['candidates']]:
        path = Path(r['path'])
        if path.is_absolute() or '..' in path.parts or tuple(path.parts[:4]) != ('RIR', role, split, scene):
            raise ValueError('Cross-scene/split or unsafe RIR path')
        base, roll = recipe(r['yaw'])
        if r['roll'] != roll or path.name != f'base_acoustic_yaw_{base:03d}.npy':
            raise ValueError('Wrong microphone order/yaw recipe')
        if len(r['xyz']) != 3 or not np.isfinite(r['xyz']).all() or not np.isfinite(r['yaw']):
            raise ValueError('Invalid pose')
        if role == 'FP':
            identity = (tuple(r['xyz']), r['yaw'] % 360)
            if identity in seen:
                raise ValueError('Duplicate pose candidate (path alone is NOT pose identity)')
            seen.add(identity)
    pos = q['candidates'][0]
    if pos['xyz'] != q['observed']['xyz'] or pos['yaw'] != q['observed']['yaw']:
        raise ValueError('Observed/positive pose mismatch')
    kinds = Counter(r['kind'] for r in q['candidates'][1:])
    if kinds != Counter(spatial_hard=4, orientation_hard=2, spatial_random=1) and split == 'train':
        raise ValueError('Hard-negative mix changed')
    for r in q['candidates'][1:]:
        distance = np.linalg.norm(np.asarray(r['xyz'])[[0, 2]] - np.asarray(pos['xyz'])[[0, 2]])
        yaw_error = abs((r['yaw'] - pos['yaw'] + 180) % 360 - 180)
        if r['kind'] == 'orientation_hard':
            if distance > 1e-6 or yaw_error < 60:
                raise ValueError('Invalid orientation negative')
        # Mining and validation may run against different NumPy/BLAS builds.
        # Existing immutable pairs include 1.0000000000000002 m boundaries
        # that round to 1.0 m on node2. Accept only a 1-nanometre numerical
        # tolerance, not genuinely close/ambiguous positive candidates.
        elif distance < (2 if r['kind'] == 'spatial_random' else 1) - 1e-9:
            raise ValueError('False negative inside GT exclusion radius')


def compile_split(root, split, queries_by_scene, sources, config):
    allowed = set(splits(root, config)[split])
    if set(queries_by_scene) != allowed:
        raise ValueError(f'{split}: incomplete scene coverage')
    entries = []
    for scene in sorted(queries_by_scene):
        queries = sorted(queries_by_scene[scene], key=lambda q: q['sample_index'])
        if not queries or len({q['sample_index'] for q in queries}) != len(queries):
            raise ValueError('Empty scene/duplicate sample')
        for q in queries:
            validate_query(q, split, allowed)
        path = Path(root) / DEST / split / f'{scene}.json'
        immutable_json(path, dict(scene_id=scene, split=split, queries=queries))
        entries.append(dict(scene_id=scene, path=str(path.relative_to(root)), sha256=sha(path), queries=len(queries)))
    index = dict(version='contrastive_manifest_v1', split=split, split_sha256=config['split_sha256'],
        negative_count=7, scenes=entries, queries=sum(e['queries'] for e in entries), sources=sources)
    path = Path(root) / DEST / f'{split}.json'
    immutable_json(path, index)
    return index


def prepare_train(root, config):
    root = Path(root)
    source = root / TRAIN_SOURCE
    index = read(source / 'index.json')
    if index['split_sha256'] != config['split_sha256'] or index['split'] != 'train':
        raise ValueError('Mining split changed')
    grouped = {}
    sources = {str((source / 'index.json').relative_to(root)): sha(source / 'index.json')}
    for e in index['scenes']:
        schedule_path = safe_path(root, e['schedule_path'])
        if sha(schedule_path) != e['schedule_sha256']:
            raise ValueError('Mining schedule changed')
        schedule = read(schedule_path)
        pair_path = safe_path(root, schedule['pair_path'])
        if sha(pair_path) != schedule['pair_sha256']:
            raise ValueError('Mining pairs changed')
        positive_path = root / 'RIR/manifests/positive_pairs_v1/schedules/train' / f'{e["scene_id"]}.json'
        if sha(positive_path) != schedule['positive_schedule_sha256']:
            raise ValueError('Positive schedule changed')
        for p in (schedule_path, pair_path, positive_path):
            sources[str(p.relative_to(root))] = sha(p)
        rows = defaultdict(list)
        for p in read(pair_path)['pairs']:
            if p['scene_id'] != e['scene_id'] or p['split'] != 'train':
                raise ValueError('Pair split leakage')
            rows[p['sample_index']].append(p)
        positive = {q['sample_index']: q for q in read(positive_path)['queries']}
        if set(rows) != set(positive) or len(rows) != e['queries']:
            raise ValueError('Missing training queries')
        queries = []
        for sample, pairs in rows.items():
            p = pairs[0]
            q = positive[sample]
            for row in pairs:
                if (row['observed_rir_path'] != q['observed_rir_path'] or row['positive_fp_rir_path'] != q['fp_rir_path']
                    or row['gt_centre_xyz_m'] != q['centre_xyz_m'] or row['gt_world_yaw_deg'] != q['world_yaw_deg']
                    or row['observed_channel_roll'] != q['channel_roll'] or row['positive_channel_roll'] != q['channel_roll']):
                    raise ValueError('Pair positive reference changed')
            obs = ref(p['observed_rir_path'], p['observed_channel_roll'], p['gt_centre_xyz_m'], p['gt_world_yaw_deg'])
            candidates = [ref(p['positive_fp_rir_path'], p['positive_channel_roll'], p['gt_centre_xyz_m'], p['gt_world_yaw_deg'])]
            candidates += [ref(x['negative_fp_rir_path'], x['negative_channel_roll'], x['negative_centre_xyz_m'],
                               x['negative_world_yaw_deg'], x['kind']) for x in pairs]
            queries.append(dict(scene_id=e['scene_id'], split='train', sample_index=sample, observed=obs,
                candidates=candidates, image_path=q['image_path']))
        grouped[e['scene_id']] = queries
    result = compile_split(root, 'train', grouped, sources, config)
    if result['queries'] != index['queries']:
        raise ValueError('Train total changed')
    return result


def val_queries(scene, schedule, seed):
    """Frozen held-out within-scene pool, independent of any test/mining model.

    Only existing val positive FP views are used: no extra dense val RIR bank.
    """
    queries = schedule['queries']
    by_xyz = defaultdict(list)
    for q in queries:
        by_xyz[tuple(q['centre_xyz_m'])].append(q)
    output = []
    for q in queries:
        rng = np.random.default_rng(int(hashlib.sha256(f'{seed}:{scene}:{q["sample_index"]}'.encode()).hexdigest()[:16], 16))
        xyz = np.asarray(q['centre_xyz_m'])
        distant = [key for key in sorted(by_xyz) if np.linalg.norm((np.asarray(key) - xyz)[[0, 2]]) > 2]
        orientation = [a for a in by_xyz[tuple(xyz)] if abs((a['world_yaw_deg'] - q['world_yaw_deg'] + 180) % 360 - 180) >= 60]
        rng.shuffle(orientation)
        yaw_selected = []
        for a in orientation:
            if all(abs((a['world_yaw_deg'] - b['world_yaw_deg'] + 180) % 360 - 180) >= 60 for b in yaw_selected):
                yaw_selected.append(a)
            if len(yaw_selected) == 2:
                break
        # Furnished/no-hit QA may retain only a subset of eight headings at a
        # captured centre. Keep every accepted query; never invent an unseen
        # yaw or discard the scene just to match a negative sampling quota.
        spatial_count = 7-len(yaw_selected)
        if len(distant) < spatial_count:
            raise ValueError(f'{scene}: too few independent val positions; cannot silently alter pool')
        selected = [by_xyz[distant[i]][int(rng.integers(len(by_xyz[distant[i]])))]
                    for i in rng.choice(len(distant), size=spatial_count, replace=False)]
        def candidate(a, kind):
            return ref(a['fp_rir_path'], a['channel_roll'], a['centre_xyz_m'], a['world_yaw_deg'], kind)
        candidates = [candidate(q, 'positive')]
        candidates += [candidate(a, 'spatial_validation') for a in selected[:-1]]
        candidates += [candidate(a, 'orientation_hard') for a in yaw_selected]
        candidates += [candidate(selected[-1], 'spatial_random')]
        output.append(dict(scene_id=scene, split='val', sample_index=q['sample_index'], image_path=q['image_path'],
            observed=ref(q['observed_rir_path'], q['channel_roll'], q['centre_xyz_m'], q['world_yaw_deg']), candidates=candidates))
    return output


def prepare_val(root, config):
    from .render_adapter import positive_tools
    prepare_scene = positive_tools().prepare_scene
    root = Path(root)
    membership = splits(root, config)
    adapter = root / 'UnLoc/frame_adapter_v4_furnished_qa/manifests/val_scenes.json'
    entries = read(adapter)['scenes']
    if {e['scene_id'] for e in entries} != set(membership['val']):
        raise ValueError('Val adapter membership changed')
    sources = {str(adapter.relative_to(root)): sha(adapter)}
    scenes, grouped = [], {}
    for e in sorted(entries, key=lambda v: v['scene_id']):
        # Separate index: do NOT mutate the index pinned by running train renders.
        scene = prepare_scene(root, root / 'source/hssd-hab', 'val', e, config['split_sha256'], 'area_weighted_mean')
        p = safe_path(root, scene['schedule_path'])
        sources[scene['schedule_path']] = sha(p)
        grouped[scene['scene_id']] = val_queries(scene['scene_id'], read(p), config['seed'])
        scenes.append(dict(**scene, schedule_sha256=sha(p)))
        print('[val prepared]', scene['scene_id'], flush=True)
    immutable_json(root / DEST / 'val_rir_index.json', dict(split='val', split_sha256=config['split_sha256'], scenes=scenes))
    return compile_split(root, 'val', grouped, sources, config)


def load_manifest(root, split, config):
    if split not in ('train', 'val'):
        raise ValueError('Training/selection never opens test data')
    root = Path(root)
    membership = splits(root, config)
    path = root / DEST / f'{split}.json'
    index = read(path)
    if index['split'] != split or index['split_sha256'] != config['split_sha256']:
        raise ValueError('Manifest split changed')
    indexed = {e['scene_id'] for e in index['scenes']}
    parent = read(root/'manifests/scene_splits_v1.json')
    if indexed != set(parent[split]):
        raise ValueError('Manifest scene coverage changed')
    for p, digest in index['sources'].items():
        if sha(safe_path(root, p)) != digest:
            raise ValueError(f'Pinned upstream changed: {p}')
    queries = []
    for e in index['scenes']:
        if e['scene_id'] not in membership[split]:
            continue
        p = safe_path(root, e['path'])
        if sha(p) != e['sha256']:
            raise ValueError('Compiled scene manifest changed')
        document = read(p)
        if document['scene_id'] != e['scene_id'] or document['split'] != split or len(document['queries']) != e['queries']:
            raise ValueError('Compiled scene identity/count changed')
        for q in document['queries']:
            validate_query(q, split, membership[split])
        queries.extend(document['queries'])
    expected = sum(e['queries'] for e in index['scenes'] if e['scene_id'] in membership[split])
    if len(queries) != expected or len({(q['scene_id'], q['sample_index']) for q in queries}) != len(queries):
        raise ValueError('Duplicate/missing compiled queries')
    return queries, sha(path)


def train_verification(root, config):
    root = Path(root)
    source = root / TRAIN_SOURCE
    report = read(source / 'distributed/verification.json')
    index = read(source / 'index.json')
    expected = dict(status='complete', plan_sha256=sha(source / 'distributed/plan.json'),
        index_sha256=sha(source / 'index.json'), split_sha256=config['split_sha256'],
        train_scenes=config['expected_scenes']['train'], queries=index['queries'], verified_pair_links=index['pairs'],
        verified_negative_arrays=index['new_fp_records'], duplicates=0, missing=0, checksum_failures=0)
    if any(report.get(k) != v for k, v in expected.items()):
        raise ValueError('Train final verification missing/stale/failed')
    return sha(source / 'distributed/verification.json')


def unique_refs(queries):
    refs = {}
    for q in queries:
        for r in [q['observed']] + q['candidates']:
            previous = refs.setdefault(r['path'], r)
            if previous['xyz'] != r['xyz'] or recipe(previous['yaw'])[0] != recipe(r['yaw'])[0]:
                raise ValueError('One raw array referenced with conflicting physical pose')
    return refs


@lru_cache(maxsize=512)
def schedule_sha(path):
    # Schedules are immutable and pinned by load_manifest at command startup.
    # Do not reread multi-MB schedule JSON for each of 388k physical RIRs.
    return sha(path)


@lru_cache(maxsize=8)
def corrected_plan(path, digest):
    if sha(path) != digest:
        raise ValueError('Corrected render plan changed')
    return read(path)


@lru_cache(maxsize=8)
def physical_fp_prefix(root, legacy_prefix):
    layout=Path(root)/'RIR/manifests/storage_layout_v1.json'
    if layout.exists():
        receipt=read(layout)
        if receipt.get('status')!='complete' or receipt.get('fp_prefix')!='RIR/FP':
            raise ValueError('Canonical RIR storage migration incomplete')
        return 'RIR/FP'
    return legacy_prefix


def checked_rir(root, relative, r, config, single_read=False):
    parts = Path(relative).parts
    corrected = config.get('corrected_extrusion')
    actual = relative
    if corrected and parts[1] == 'FP':
        if corrected['prefix'] not in ('RIR/FP_corrected_yup_v2','RIR/FP'):
            raise ValueError('Unknown corrected FP namespace')
        prefix=physical_fp_prefix(str(root),corrected['prefix'])
        actual = str(Path(prefix).joinpath(*parts[2:]))
    path = safe_path(root, actual)
    metadata_bytes = path.with_suffix('.json').read_bytes() if single_read else None
    meta = json.loads(metadata_bytes) if single_read else read(path.with_suffix('.json'))
    native = config.get('native_frame')
    if native and parts[1] == 'FP':
        plan = corrected_plan(str(safe_path(root, native['plan_path'])), native['plan_sha256'])
        stage = plan['stages'][parts[3]]
        frame = meta.get('loaded_frame') or {}
        bounds = np.asarray(frame.get('loaded_bounds', []))
        if (not frame.get('verified') or frame.get('up') != [0, 1, 0]
            or frame.get('front') != [0, 0, -1] or bounds.shape != (2, 3)
            or not np.allclose(bounds, stage['expected_bounds'], rtol=0, atol=1e-4)
            or meta.get('geometry_sha256') != stage['mesh_sha256']):
            raise ValueError('Native FP lacks verified Y-up geometry/bounds lineage')
    if corrected and parts[1] == 'FP':
        frame = meta.get('loaded_frame') or {}
        if (not frame.get('verified') or frame.get('up') != [0,1,0]
            or frame.get('front') != [0,0,-1]
            or meta.get('corrected_plan_sha256') != corrected['plan_sha256']):
            raise ValueError('Corrected FP lacks verified loaded-frame lineage')
        plan = corrected_plan(str(safe_path(root, corrected['plan_path'])), corrected['plan_sha256'])
        expected = plan['stages'][parts[3]]['expected_bounds']
        if not np.allclose(frame.get('loaded_bounds', []), expected, rtol=0, atol=1e-4):
            raise ValueError('Corrected FP runtime bounds mismatch')
    angles = np.asarray(meta.get('microphone_angles_rad', []))
    if (meta.get('role') != parts[1] or meta.get('split') != parts[2] or meta.get('scene_id') != parts[3]
        or meta.get('centre_xyz_m') != r['xyz'] or meta.get('source_xyz_m') != r['xyz']
        or meta.get('base_acoustic_yaw_deg') != recipe(r['yaw'])[0]
        or meta.get('acoustics') != PROTOCOL
        or angles.shape != (6,) or not np.allclose(angles,
                           RING + np.deg2rad(recipe(r['yaw'])[0]), rtol=0, atol=1e-12)):
        raise ValueError(f'Raw RIR contract mismatch: {relative}')
    schedule = (Path(root) / TRAIN_SOURCE / 'schedules/train' / f'{parts[3]}.json') if parts[4] == 'negative_records' else (
        Path(root) / 'RIR/manifests/positive_pairs_v1/schedules' / parts[2] / f'{parts[3]}.json')
    payload = path.read_bytes() if single_read else None
    raw_sha = hashlib.sha256(payload).hexdigest() if single_read else sha(path)
    if meta.get('schedule_sha256') != schedule_sha(str(schedule)) or meta.get('rir_sha256') != raw_sha:
        raise ValueError(f'Raw checksum/schedule mismatch: {relative}')
    array = np.load(io.BytesIO(payload) if single_read else path, allow_pickle=False)
    if (array.ndim != 2 or array.shape[0] != 6 or not np.isfinite(array).all()
        or not np.all(np.any(array != 0, axis=-1)) or list(array.shape) != meta['rir_shape']):
        raise ValueError(f'Invalid raw RIR: {relative}')
    metadata_sha = hashlib.sha256(metadata_bytes).hexdigest() if single_read else sha(path.with_suffix('.json'))
    return array, dict(raw_sha256=meta['rir_sha256'], metadata_sha256=metadata_sha,
                      raw_samples=int(array.shape[-1]))


def verify_val(root, config, workers):
    queries, manifest_sha = load_manifest(root, 'val', config)
    refs = unique_refs(queries)
    def check(item):
        p, r = item
        _, receipt = checked_rir(root, p, r, config)
        return p, receipt
    with ThreadPoolExecutor(max_workers=workers) as pool:
        receipts = dict(pool.map(check, refs.items()))
    report = dict(status='complete', split='val', manifest_sha256=manifest_sha,
        split_sha256=config['split_sha256'], queries=len(queries), arrays=len(refs), records=receipts)
    atomic_json(Path(root) / DEST / 'val_verification.json', report)
    return report


def verification(root, split, config, manifest_sha):
    if config.get('native_frame'):
        native = config['native_frame']
        corrected_plan(str(safe_path(root, native['plan_path'])), native['plan_sha256'])
        path = safe_path(root, native['verification_directory'])/f'{split}_verification.json'
        report = read(path)
        if (report.get('status') != 'complete' or report.get('split') != split
            or report.get('manifest_sha256') != manifest_sha
            or report.get('plan_sha256') != native['plan_sha256']
            or report.get('split_sha256') != config['split_sha256']
            or report.get('test_accessed') is not False):
            raise ValueError('Native-frame raw verification missing/stale')
        return sha(path)
    if config.get('corrected_extrusion'):
        corrected = config['corrected_extrusion']
        plan_path = safe_path(root, corrected['plan_path'])
        if sha(plan_path) != corrected['plan_sha256']:
            raise ValueError('Corrected render plan changed')
        path = plan_path.parent/f'{split}_verification.json'
        report = read(path)
        if (report.get('status') != 'complete' or report.get('manifest_sha256') != manifest_sha
            or report.get('plan_sha256') != corrected['plan_sha256']
            or report.get('split_sha256') != config['split_sha256']
            or report.get('split') != split or report.get('test_accessed') is not False):
            raise ValueError('Corrected raw verification missing/stale')
        return sha(path)
    if split == 'train':
        return train_verification(root, config)
    path = Path(root) / DEST / 'val_verification.json'
    report = read(path)
    if report.get('status') != 'complete' or report.get('split') != 'val' or report.get('manifest_sha256') != manifest_sha or report.get('split_sha256') != config['split_sha256']:
        raise ValueError('Val raw verification missing/stale')
    return sha(path)


def cache_directory(root, config, split):
    directory = Path(root) / 'AcousticContrastive/features' / feature_id(config['features'])
    if config.get('scene_subset_sha256'):
        directory = directory / 'subsets' / config['scene_subset_sha256'][:16]
    if config.get('corrected_extrusion'):
        directory = directory / 'corrected_yup_v2' / config['corrected_extrusion']['plan_sha256'][:16]
    if config.get('native_frame'):
        directory = directory / 'native_yup' / config['native_frame']['plan_sha256'][:16]
    return directory / split


def build_cache(root, config, split, workers=4, shard_size=256):
    """Bounded CPU, one extraction per physical array, roll at load time.

    Immutable shard receipts permit interruption/resume without trusting partial
    files. Train requires the distributed full-RIR verifier to finish first.
    """
    queries, manifest_sha = load_manifest(root, split, config)
    verified_sha = verification(root, split, config, manifest_sha)
    refs = unique_refs(queries)
    dest = cache_directory(root, config, split)
    dest.mkdir(parents=True, exist_ok=True)
    lock = (dest / 'build.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    groups = defaultdict(list)
    for p in sorted(refs):
        groups['/'.join(Path(p).parts[1:4])].append(p)
    shards = []
    completed = 0
    def work(p):
        array, receipt = checked_rir(root, p, refs[p], config)
        energy, coherence = extract(array, config['features'])
        receipt['zero_tail_padding'] = bool(config['features']['version'] == 'energy_complex_zero_tail_v2'
            and array.shape[-1] < 7 + round(config['features']['end_ms']*48))
        if config.get('cache_envelope'):
            from .envelope_cache import extract as extract_envelope
            return energy, coherence, receipt, extract_envelope(array)
        return energy, coherence, receipt
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for group, paths in sorted(groups.items()):
            for offset in range(0, len(paths), shard_size):
                selected = paths[offset:offset + shard_size]
                directory = dest / group / f'{offset // shard_size:05d}'
                directory.mkdir(parents=True, exist_ok=True)
                receipt_path = directory / 'receipt.json'
                if receipt_path.exists():
                    receipt = read(receipt_path)
                    if receipt['paths'] != selected or receipt['manifest_sha256'] != manifest_sha or receipt['features'] != config['features']:
                        raise ValueError('Stale feature shard')
                    for name, digest in receipt['files'].items():
                        if sha(directory / name) != digest:
                            raise ValueError('Corrupt committed feature shard')
                    if config.get('cache_envelope'):
                        from .envelope_cache import DEFINITION
                        if receipt.get('envelope_definition') != DEFINITION or 'envelope.npy' not in receipt['files']:
                            raise ValueError('Joint envelope definition/coverage changed')
                else:
                    values = list(pool.map(work, selected))
                    files = {}
                    streams = [('energy.npy', [v[0] for v in values]), ('coherence.npy', [v[1] for v in values])]
                    if config.get('cache_envelope'):
                        streams.append(('envelope.npy', [v[3] for v in values]))
                    for name, arrays in streams:
                        path = directory / name
                        temporary = path.with_name(name + '.' + uuid.uuid4().hex + '.tmp')
                        with temporary.open('wb') as stream:
                            np.save(stream, np.asarray(arrays, dtype=np.float16), allow_pickle=False)
                        temporary.replace(path)
                        files[name] = sha(path)
                    receipt = dict(paths=selected, records=[v[2] for v in values], files=files,
                        manifest_sha256=manifest_sha, features=config['features'])
                    if config.get('cache_envelope'):
                        from .envelope_cache import DEFINITION
                        receipt['envelope_definition'] = DEFINITION
                    atomic_json(receipt_path, receipt)
                shards.append(dict(path=str(receipt_path.relative_to(root)), sha256=sha(receipt_path)))
                completed += len(selected)
                atomic_json(dest / 'progress.json', dict(status='running', completed_arrays=completed,
                    total_arrays=len(refs), workers=workers, updated_unix=time.time()))
                if len(shards) % 25 == 0:
                    print(f'[{split} features] {offset + len(selected)}/{len(paths)} in {group}; {len(shards)} shards', flush=True)
    result = dict(status='complete', split=split, split_sha256=config['split_sha256'],
        manifest_sha256=manifest_sha, verification_sha256=verified_sha, features=config['features'],
        arrays=len(refs), shards=shards)
    atomic_json(dest / 'index.json', result)
    atomic_json(dest / 'progress.json', dict(status='complete', completed_arrays=completed,
        total_arrays=len(refs), workers=workers, updated_unix=time.time()))
    return result


def check_cache(root, config, split, manifest_sha, full=True):
    dest = cache_directory(root, config, split)
    index = read(dest / 'index.json')
    if (index.get('status') != 'complete' or index['split'] != split or index['features'] != config['features']
        or index['manifest_sha256'] != manifest_sha or index['split_sha256'] != config['split_sha256']
        or index['verification_sha256'] != verification(root, split, config, manifest_sha)):
        raise ValueError('Feature cache incomplete/stale')
    locations = {}
    for shard in index['shards']:
        path = safe_path(root, shard['path'])
        if sha(path) != shard['sha256']:
            raise ValueError('Feature receipt changed')
        receipt = read(path)
        if receipt['manifest_sha256'] != manifest_sha or receipt['features'] != config['features']:
            raise ValueError('Feature shard identity changed')
        if config.get('cache_envelope'):
            from .envelope_cache import DEFINITION
            if receipt.get('envelope_definition') != DEFINITION or 'envelope.npy' not in receipt['files']:
                raise ValueError('Joint envelope cache missing or changed')
        for name, digest in receipt['files'].items():
            if full and sha(path.parent / name) != digest:
                raise ValueError('Feature data checksum mismatch')
        for i, p in enumerate(receipt['paths']):
            if p in locations:
                raise ValueError('Duplicate feature array')
            locations[p] = (str(path.parent), i)
    if len(locations) != index['arrays']:
        raise ValueError('Feature array coverage mismatch')
    return locations, sha(dest / 'index.json')


def extrusion_frame_blocker(root):
    audit=Path(root)/'RIR/manifests/extrusion_frame_audit_v1.json'
    if audit.exists() and read(audit).get('status')=='legacy_invalid':
        return 'Legacy canonical-extrusion FP RIRs have an unvalidated/rotated loader frame. Regenerate and verify loaded bounds before training; see '+str(audit)
    return None


def preflight(root, config):
    validate_config(config['features'])
    membership = splits(root, config)
    blockers, details = [], {}
    frame_error=extrusion_frame_blocker(root)
    if frame_error and not (config.get('corrected_extrusion') or config.get('native_frame')):
        return dict(status='blocked',split_counts={k:len(v) for k,v in membership.items()},
            split_sha256=config['split_sha256'],datasets={},blockers=[dict(split='FP',reason=frame_error)],test_accessed=False)
    for split in ('train', 'val'):
        try:
            queries, digest = load_manifest(root, split, config)
            verification(root, split, config, digest)
            locations, cache_sha = check_cache(root, config, split, digest)
            if set(locations) != set(unique_refs(queries)):
                raise ValueError('Feature cache does not cover exact query references')
            details[split] = dict(queries=len(queries), arrays=len(locations), manifest_sha256=digest, cache_sha256=cache_sha)
        except (FileNotFoundError, ValueError, KeyError) as error:
            blockers.append(dict(split=split, reason=str(error)))
    return dict(status='ready' if not blockers else 'blocked', split_counts={k: len(v) for k, v in membership.items()},
        split_sha256=config['split_sha256'], datasets=details, blockers=blockers, test_accessed=False)


def render_val(root, config, renderer_dir, workers):
    """Explicit future CPU render command; refuses to overlap current RIR writers."""
    from .render_adapter import supervisor_tools
    supervise_tasks = supervisor_tools().supervise_tasks
    for key in ('CUDA_VISIBLE_DEVICES', 'NVIDIA_VISIBLE_DEVICES'):
        if os.environ.get(key) not in ('-1', 'void', 'none', ''):
            raise ValueError('Val RIR rendering must be CPU-only')
    root = Path(root)
    load_manifest(root, 'val', config)
    index = read(root / DEST / 'val_rir_index.json')
    if index['split'] != 'val' or index['split_sha256'] != config['split_sha256']:
        raise ValueError('Val render split mismatch')
    lock = (root / 'RIR/manifests/positive_pairs_v1/render.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    tasks = []
    for e in index['scenes']:
        path = safe_path(root, e['schedule_path'])
        if sha(path) != e['schedule_sha256'] or e['split'] != 'val':
            raise ValueError('Val schedule changed')
        ids = [p['pose_id'] for p in read(path)['positions']]
        for role in ('FP', 'Observed'):
            for offset in range(0, len(ids), 12):
                tasks.append((str(root), str(root / 'source/hssd-hab'), str(renderer_dir), e, role, ids[offset:offset+12], 2))
    supervise_tasks(tasks, root / DEST / 'val_render', workers,
        expected_records=sum(2 * e['physical_records_per_role'] for e in index['scenes']))
    return verify_val(root, config, workers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['prepare-train', 'prepare-val', 'render-val', 'verify-val', 'cache', 'preflight'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--split', choices=['train', 'val'], default='train')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--renderer-dir', type=Path)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('workers must be positive')
    config = read(args.config)
    if args.phase == 'prepare-train': result = prepare_train(args.root, config)
    elif args.phase == 'prepare-val': result = prepare_val(args.root, config)
    elif args.phase == 'verify-val': result = verify_val(args.root, config, args.workers)
    elif args.phase == 'render-val':
        if args.renderer_dir is None: parser.error('--renderer-dir required')
        result = render_val(args.root, config, args.renderer_dir, args.workers)
    elif args.phase == 'cache': result = build_cache(args.root, config, args.split, args.workers)
    else: result = preflight(args.root, config)
    print(json.dumps({k: v for k, v in result.items() if k not in ('scenes', 'shards', 'records', 'sources')}, indent=2))
    if args.phase == 'preflight' and result['status'] != 'ready':
        raise SystemExit(2)


if __name__ == '__main__':
    main()

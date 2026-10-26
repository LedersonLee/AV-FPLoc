"""Freeze a small scene-disjoint diagnostic experiment, without score selection."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from .contracts import sha, immutable_json
from .data import read, splits, load_manifest, unique_refs, TRAIN_SOURCE

SCENE5 = '104862687_172226883'


def choose(parent, counts, seed=42):
    result = {}
    for split in ('train', 'val', 'test'):
        ordered = sorted(parent[split], key=lambda scene: hashlib.sha256(
            f'hssd-acoustic-pilot-v1:{seed}:{split}:{scene}'.encode()).hexdigest())
        if split == 'test':
            ordered = [SCENE5] + [s for s in ordered if s != SCENE5]
        result[split] = ordered[:counts[split]]
        if len(result[split]) != counts[split] or not set(result[split]) <= set(parent[split]):
            raise ValueError('Invalid pilot membership')
    return result


def prepare(root, base_config):
    root = Path(root)
    parent = splits(root, base_config)
    counts = dict(train=12, val=3, test=3)
    selected = choose(parent, counts)
    destination = root/'AcousticContrastive/pilots/scene_subset_v1'
    destination.mkdir(parents=True, exist_ok=True)
    subset_path = destination/'scene_splits.json'
    immutable_json(subset_path, dict(parent_split_sha256=base_config['split_sha256'], scenes=selected,
        selection='SHA256(seed42, split, scene), no performance/GT-error input; scene5 forced as historical reference',
        seed=42, purpose='exploratory multi-scene diagnostic, not a final benchmark'))
    config = copy.deepcopy(base_config)
    config.update(scene_subset_path=str(subset_path.relative_to(root)), scene_subset_sha256=sha(subset_path), subset_counts=counts)
    config['features']['version'] = 'energy_complex_zero_tail_v2'
    config['training'].update(epochs=40, early_stop_patience=8)
    immutable_json(destination/'config.json', config)
    queries, arrays = {}, {}
    for split in ('train', 'val'):
        records, _ = load_manifest(root, split, config)
        queries[split] = len(records)
        arrays[split] = len(unique_refs(records))
    protocol = dict(version='small_multiscene_diagnostic_v1', scenes=selected,
        config_sha256=sha(destination/'config.json'), queries=queries, physical_training_arrays=arrays,
        selection_checkpoint='best held-out val loss; max40 epochs/patience8; never choose on test',
        vision=dict(checkpoint='UnLoc/evaluation_checkpoints/epoch020/model.ckpt', frozen=True,
            caveat='Vision previously trained on116 scenes; acoustic on12. Pilot is not equal-data backbone benchmarking.'),
        evaluation=dict(methods=['vision_only', 'acoustic_only', 'equal_weight_fusion', 'visual_topk_acoustic_rerank'],
            candidate_scope='identical world XY/yaw bank for all four methods; no GT-injected candidates',
            dense_grid_spacing_m=.2, yaw_step_deg=10, query_scope='all accepted v4 RGB query poses in selected test scenes',
            primary_k=50, diagnostic_ks=[5,10,20],
            equal_weight='0.5*zscore(visual score)+0.5*zscore(acoustic cosine); per query on same valid full bank; no test-fitted weight',
            topk='top-K XY by max-yaw vision, acoustic chooses XY by max-yaw audio, report vision yaw of selected XY',
            metrics=['recall_0.1m','recall_0.5m','recall_1m','recall_1m_30deg','median_xy_m','median_yaw_deg'],
            diagnostics=['topK spatial oracle coverage','topK joint-pose oracle coverage',
                'repair/harm/preserve versus vision','acoustic rank of GT-nearest candidate',
                'yaw-only versus position failures','geometry/observation mismatch audit'],
            restricted_bank_caveat='If only sampled-pose RIRs exist, label results restricted-bank, not global localization; never compare to dense vision as same task.'),
        raw_data_policy='reuse existing train/val RIRs; do not delete or resplit parent data',
        hypotheses=['test whether acoustic learns pose-discriminative information under clutter',
            'test whether topK is beneficial, not assume it is',
            'separate visual candidate exclusion from acoustic candidate-selection failure'],
        stage='subset prepared; acoustic training and new multi-scene evaluation not yet run')
    immutable_json(destination/'protocol.json', protocol)
    print(json.dumps(dict(scenes=selected, queries=queries, arrays=arrays, output=str(destination)), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.root, read(args.config))


if __name__ == '__main__':
    main()

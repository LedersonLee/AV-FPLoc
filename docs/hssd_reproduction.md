# HSSD two-branch reproduction

The paper name (AV-FLoc / AV-Loc) is not fixed. Repository `AV-FPLoc` and
Python namespace `avfploc` are infrastructure names, not a publication claim.

## Scope

Independent frozen UnLoc and contrastive acoustic branches, full aligned XY/yaw
score banks, fixed or query-adaptive scalar fusion. No spatial-interaction or
direct floorplan/acoustic cross-attention experiments are enabled by this package.
Adaptive fusion is an experimental method, not a proven improvement over fixed
acoustic weight 0.1. Both weights sum to one: `(1-w)*vision + w*acoustic`.

## Installation

```bash
python3.10 -m venv .venv
. .venv/bin/activate
pip install pip==25.2
pip install torch==2.4.1+cu124 torchvision==0.19.1+cu124 --extra-index-url https://download.pytorch.org/whl/cu124
pip install -r environments/requirements-cu124.txt
pip install -e .
pytest tests/hssd
export AVFPLOC_DATA_ROOT=/file2/jeongeon/AV-FPLoc/datasets/HSSD
```

Alternatively `docker build -f environments/Dockerfile -t avfploc:cu124 .`.
The Docker recipe uses a public base, not an image existing only on the author's
PC. No credentials, data, checkpoint, or external clone is bundled in Git.
Install W&B credentials yourself; offline/disabled logging is supported.

## External Vision dependency

```bash
python scripts/setup_unloc.py --destination external/checkouts/UnLoc
```

The script pins UnLoc and DepthAnythingV2 commits and applies our tracked patch.
Place `depth_anything_v2_metric_hypersim_vitl.pth` at
`external/checkouts/UnLoc/depth_anything_v2/checkpoints/` (a shared-file symlink
is fine). Use the original model's license/download terms. Never overwrite an
existing local checkout. `external/UNLOC_LICENSE` records upstream terms.

## Portable bank contract

`vision.npz` and `acoustic.npz` use:

* `scores`: float `[query, candidate_xy, 36]`, higher is better;
* `xy`: `[candidate_xy,2]` world X,Z in metres;
* `yaw_deg`: exactly `[0,10,...,350]`, world +X=0, +Z=90;
* `query_ids`: unique string IDs, same ordering in both banks;
* Vision additionally `depth[query,40]`, `ray_bank[candidate_xy,36]` for adaptive
  features (DESDF spacing 0.1m, structural cap 10m in the historical protocol).

Candidate sampling was 0.2m on both axes, not 0.2 square metres. Both branches
must use the same candidate list; fusion refuses silent reordering/interpolation.
NPZ is convenient for scene-level/small exports; do not combine all scenes into
one huge archive. Core inference APIs can be called in query chunks for large banks.

Vision input NPZ: `images` (paths relative to NPZ), `query_ids`, `xy`, `ray_bank`.
Acoustic input NPZ: `observed_energy`, `observed_coherence`, `observed_envelope`,
`candidate_energy`, `candidate_coherence`, `candidate_envelope`, `xy`, `query_ids`.
Envelope is required by E+C+N. Candidate features are cell-major/yaw-minor
with all 36 world headings already expanded. Raw six physical base headings
must use `avfploc.acoustic.contracts.recipe` and microphone rolls before encoding;
do not repeat a single yaw or rotate embedded vectors. E/C use 0.5–4kHz;
common-removed envelope uses 0.5–1.5kHz and training-compatible float16 quantization.

```bash
avfploc vision --inputs vision_inputs.npz --checkpoint "$AVFPLOC_DATA_ROOT/UnLoc/evaluation_checkpoints/epoch020/model.ckpt" --unloc-root external/checkouts/UnLoc --output vision.npz
avfploc acoustic --inputs acoustic_inputs.npz --checkpoint "$AVFPLOC_DATA_ROOT/AcousticContrastive/pilots/scene_subset_v1/input_ablation_v1/runs/all/best.pt" --output acoustic.npz
avfploc fuse --vision vision.npz --acoustic acoustic.npz --weight 0.5 --output equal.npz
avfploc fuse --vision vision.npz --acoustic acoustic.npz --weight 0.1 --output fixed01.npz
avfploc fuse --vision vision.npz --acoustic acoustic.npz --checkpoint gate.pt --schema processed58 --output adaptive.npz
avfploc evaluate --predictions adaptive.npz --truth truth.npz
```

The two inference commands can run independently or concurrently if memory permits;
fusion runs only after both complete. `truth.npz` has `poses[Q,3]` and `query_ids`.
GT is never an argument to fusion inference. Per-query z-score normalization is
over the whole XY/yaw bank. These are relative scores, not calibrated likelihoods.

## Weighted fusion training

`avfploc train-fusion --manifest train_val.json --output runs/new_gate --schema processed58`.
Manifest has **only** `train` and `val` lists, each with `scene_id`, `vision`,
`acoustic`, `truth` paths relative to the manifest. Training refuses overlapping
scene IDs/files and fits feature normalization on Train only. It saves epoch0,
best/last checkpoints and Train/Val Recall, Repair/Harm versus fixed0.1.

This portable trainer uses per-query updates and Val scene-macro joint Recall
selection. It is **not** a byte-for-byte reproduction of historical batch16,
scene-balanced, logit-regression/success-set schedules. Those are retained under
`experiments/legacy/reliability_fusion/`; use their archived configs for historical
comparisons. No new localization performance is claimed from migration tests.

Supported checkpoint schemas are ordered core14, processed58, legacy17. Schema,
Train mean/std and MLP widths are checked at inference. Old checkpoints lacking
feature names/normalization must be explicitly converted with their original
metadata, never guessed. Only load trusted torch checkpoints (pickle is enabled).

## Other training and data

`python -m avfploc.acoustic.train --help` and
`python -m avfploc.vision.train --help` expose the migrated original trainers.
The acoustic default config is a historical initial schema, **not** a replacement
for a corrected E+C+N run config. Use the config saved with the selected run.
Training requires checksum-verified prepared features/manifests; this release
does not launch node1/node2 or regenerate RIRs. Renderer integration remains an
optional legacy adapter, not part of the portable inference dependency chain.

Use explicit split manifests: parent HSSD116/19/17 and subset12/3/3 are different
scopes. Do not resplit randomly or select checkpoints/weights using Test.
Dataset assets remain on shared File2; local SSD caches are not automatically
available to colleagues. See the release artifact manifest for checked files.

## Shared three-query smoke example (Train only)

```bash
python scripts/verify_artifacts.py --root "$AVFPLOC_DATA_ROOT" --full-hash
SMOKE="$AVFPLOC_DATA_ROOT/releases/two_branch_fusion_v1/smoke_train"
avfploc fuse --vision "$SMOKE/vision.npz" --acoustic "$SMOKE/acoustic.npz" --weight 0.1 --output /tmp/avfploc-fixed01.npz
avfploc fuse --vision "$SMOKE/vision.npz" --acoustic "$SMOKE/acoustic.npz" --checkpoint "$AVFPLOC_DATA_ROOT/ReliabilityFusion/pilots/capacity_probe_v1/runs/128_64_seed42/best.pt" --schema processed58 --output /tmp/avfploc-adaptive.npz
avfploc evaluate --predictions /tmp/avfploc-adaptive.npz --truth "$SMOKE/truth.npz"
```

Use new output names on subsequent runs; overwrites are refused. This is an
execution/compatibility test, not a representative performance evaluation.
For other scenes use `scripts/export_score_bank.py --help` with explicit cache,
depth and DESDF paths. The existing complete score caches can remain on local SSD;
the portable smoke example and required model weights are on File2.
The DA file for the external checkout is shared at
`$AVFPLOC_DATA_ROOT/UnLoc/pretrained/depth_anything_v2_metric_hypersim_vitl.pth`.

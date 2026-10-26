# HSSD preprocessing

## Scope

- Main code: `src/avfploc/preprocessing/hssd/`.
- Original files remain unchanged under the workspace's `experiments/exp4_acoustic_xy_observability/scripts/`.
- Internal imports are package-relative; commands no longer require that experiment directory on `PYTHONPATH`.
- Source hashes: [migration provenance](hssd_preprocessing_sources.json).
- Migration does not regenerate data, change scene membership, or launch remote workers.

## Dependencies

```bash
pip install -e '.[preprocess,test]'
pytest tests/preprocessing
python -m avfploc.preprocessing.hssd.build_hssd_unloc_dataset --help
```

- Geometry/QA helpers: NumPy, SciPy, Pillow, PyYAML.
- RGB/navmesh/depth: a compatible Habitat-Sim environment and licensed HSSD assets.
- RIR: audio-enabled Habitat/SoundSpaces plus an explicit `--renderer-dir` containing `hssd_cpu_rir_contract`, `render_hssd_paired_rirs`, and `render_hssd_exp4_ring_rirs` and their dependencies.
- The training Docker image is not an audio-rendering environment.
- RIR generation retains CPU-only device checks; moving code does not authorize GPU use on shared nodes.

## Roots and split safety

- `--hssd-root`: check each command's help; rendering commands use raw HSSD assets, while some historical audit/export commands use the prepared dataset root.
- Prepared data: `/file2/jeongeon/AV-FPLoc/datasets/HSSD` or its SSD mirror.
- Existing split: `manifests/scene_splits_v1.json` (full 116/19/17); use the separate 12/3/3 manifest for subset work.
- Geometry builder requires `--split-manifest`; it validates duplicate/overlapping IDs and preserves ordering.
- `--legacy-create-split` explicitly reproduces the initial 160-scene split. Do not use it for the curated release.
- Geometry builder requires an empty output directory. Other migrated tools retain their existing write/resume behavior: run in a staging copy, not directly over a released dataset.
- Audit/replenishment tools read the existing v2 adapter split manifests; prepare/copy these before running them. They do not infer splits from directory names.

## Processing order

| Step | Module | Result / constraint |
|---|---|---|
| 1 | `build_hssd_unloc_dataset` | Initial floorplans and capture schedules; `--scene` explicitly selects maps to build |
| 2 | `reconcile_hssd_unloc_connectivity` | Connectivity checks and repairs; review before acceptance |
| 3 | `build_hssd_unloc_curation_manifest` | Accepted/excluded scenes with explicit manual decisions |
| 4 | `build_hssd_unloc_semantic_maps` | Separate geometry and door/window maps |
| 5 | `render_hssd_unloc_rgb_pilot` | Furnished RGB and capture metadata; default is a pilot, not a full release |
| 6 | `export_hssd_unloc_frames` | Structural ray targets and frame records |
| 7 | `audit_hssd_unloc_furnished_depth` | Furnished-depth checks; retain natural ambiguity |
| 8 | `reclassify_hssd_furnished_nohit` | Reclassify existing audit results using the v4 no-hit policy |
| 9 | `replenish_hssd_unloc_furnished_views` | Replacement views from the same scene/split |
| 10 | `export_hssd_unloc_furnished_adapter` | Final retained/replacement query adapter |
| 11 | `prepare_hssd_unloc_structural_targets`, `validate_hssd_nohit_dataset` | Target parity and release checks |

- This is a dependency guide, not an automatic runner. Intermediate manifests and manually reviewed curation inputs are required.
- `build_hssd_*_review` modules produce HTML for QA.
- Canonical storage remains separate: `2DFP/`, `2DFP_w_semantic/`, `UnLoc/`, and `RIR/`.

## Preserved contracts

- World Y-up; raster +X right, +Z down; pose/yaw conversions unchanged.
- Geometry is binary: white indoor free, black wall/exterior. Semantics are a separate artifact.
- Wall-core filling, portal handling, and vertical-clearance checks retain their original algorithms.
- Furnished v4 no-hit-only policy retains natural false-closed/false-open ambiguity; missing/corrupt evidence still fails validation.
- Preserve the historical ray ordering, depth convention, and target cap; do not infer a new cap from scene size during migration.

## Acoustic preparation

- Positive/negative scheduling: `python -m avfploc.preprocessing.hssd.hssd_positive_pairs --help` and `hssd_negative_rirs --help`.
- Resumable CPU scheduler: `hssd_rir_supervisor`; worker subprocesses use the package module path.
- Corrected FP frame verification and schedule/geometry hashes remain enforced.
- Feature extraction and verification: `python -m avfploc.acoustic.data --help` and `python -m avfploc.acoustic.envelope_cache --help`.
- Distributed node launchers remain historical operational scripts; no node jobs are migrated or restarted automatically.

## Verification limits

- Synthetic regression tests cover geometry, semantic masks, yaw, clearance, no-hit policy, replacement bookkeeping, and split isolation.
- CLI help checks do not render scenes or prove full-release bitwise reproduction.
- Before a new bulk run, validate one scene against the released maps/RGB/targets using the original simulator environment and manifests.

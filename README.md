# AV-FPLoc

- **Goal:** estimate floorplan position and yaw from visual and acoustic observations.
- **Vision:** UnLoc RGB-to-structural-depth branch.
- **Acoustic:** contrastive encoder with STFT energy, complex coherence, and common-removed envelope.
- **Fusion:** aligned candidate scores with fixed or learned query-level scalar weights.
- **Status:** adaptive fusion is experimental; improvement over fixed weights is not established.
- **Naming:** repository/package names are provisional; the paper title is undecided.

## Quick start

- [Installation, checkpoints, inference, and training](docs/hssd_reproduction.md)
- [Migration checks and numerical validation](docs/migration_validation.md)
- [HSSD preprocessing: maps, RGB, QA, and RIR schedules](docs/hssd_preprocessing.md)
- Run both branches independently, then fuse their aligned scores.
- Shared HSSD data: `/file2/jeongeon/AV-FPLoc/datasets/HSSD`.

## Repository structure

```text
src/avfploc/   Main library: vision, acoustic, fusion, data, geometry, evaluation
configs/       Dataset and experiment settings
experiments/   Historical experiments, separate from the main library
tests/         Unit, portability, and numerical-contract tests
environments/  Installation and Docker recipes
external/      Upstream pins, licenses, and patches
scripts/       Setup, verification, export, and legacy Gibson utilities
docs/          Reproduction instructions and research records
```

## Reproduction rules

- Keep Train/Val/Test scenes separate; select models and weights without Test.
- Distinguish full HSSD splits (116/19/17) from the subset (12/3/3).
- Match query IDs, candidate positions, and yaw ordering before fusion.
- Keep data, checkpoints, generated outputs, and credentials outside Git.
- Historical experiment scripts may require their original caches and environment.

## Legacy Gibson support

- Path helpers: `src/avfploc/data/gibson_paths.py`.
- [Gibson RIR workflow](docs/collaborator_gibson_rir.md).
- Gibson audit tools require a configuration; the default `configs/datasets/gibson_av.yaml` is not bundled.
- Earlier design documents are historical records, not the current package layout.

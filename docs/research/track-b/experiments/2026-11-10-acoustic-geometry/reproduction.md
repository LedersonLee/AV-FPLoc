# Reproduction and execution provenance

Repository: `AV-FPLoc`, branch `refactor/hssd-two-branch-fusion`.
External study root: `/home/jeongeonlee/track1/experiments/acoustic_geometry_20261110`.
Inputs are the approved HSSD dataset and Study27's frozen cohort, not included in Git. Clean-clone reproduction requires those assets. The two pre-existing unrelated dirty Gibson runtime files are excluded from all commits and executions here.

Container mounts: repository→`/work` read-only; HSSD→`/data` read-only; Study27 `cache_v1`→`/cohort` read-only; new study root→`/results` read/write. `PYTHONPATH=/work`, network disabled, host UID/GID,3 CPU limit. Geometry uses `echoloc/floorplan-geometry:py311`,10GB limit, `OPENBLAS_NUM_THREADS=1`; features use `avfploc:cu124-clean`,8GB, `OMP_NUM_THREADS=2`. Training/smoke adds GPU0 and retains other processes. Image identities: geometry `sha256:2c4e632c38280b5ac6c787d74bccdc80236ebf689f407ee390297cc369055a9a`; training `sha256:e51cb712100f25d0018aa65e371e63deaa51d12601c465ce698c0c446800c318`.

Executed target generation:

```bash
python -m track_b.acoustic_geometry.prepare_targets --root /data --cohort /cohort/protocol.json --out /results/targets_v3 --code-sha f9d0f2e92a8d45cb750d65f38f4f3bed14859a42
```

Executed feature preparation (completion recorded by its `COMPLETE` file):

```bash
python -m track_b.acoustic_geometry.prepare_features --root /data --targets /results/targets_v2 --out /results/features_v1 --code-sha bd05ef297da92b085975e9c53d4a2c08b73a6f0b
```

Features reuse the earlier target query manifest; final training replaces its labels with independently verified v3 targets. This avoids silently editing the already hashed feature packages.

Executed GPU smoke:

```bash
python -m track_b.acoustic_geometry.smoke --features /results/features_v1 --targets /results/targets_v3 --out /results/smoke_v2 --code-sha 0bd7faa8280862a43e80ca1baaac341c3d623f2f
python -m track_b.acoustic_geometry.qa --root /data --targets /results/targets_v3 --out /results/qa_v2
python -m pytest /work/track_b/tests/test_acoustic_geometry_models.py /work/track_b/tests/test_acoustic_geometry_targets.py -q -p no:cacheprovider
```

Completed fixed-budget primary training, after complete inputs and runner smoke:

```bash
python -m track_b.acoustic_geometry.train --features /results/features_v1 --targets /results/targets_v3 --out /results/representation_v1 --steps 1500 --seeds 42 43 44 --code-sha 13f18b028b35d694c8e0adee4d138d46fecf9d00
```

All output names must be new. Do not overwrite failed attempts. Fixed checkpoint selection uses final scheduled updates, with no evaluation on historical Val19 yet. Configuration, per-step training curve, per-scene metrics, per-query distributions and checkpoint checksums are written by the runner. Actual run status supersedes this prepared command and must be verified before claiming results.

## Completed follow-up executions

The same training command with `--out /results/phase_zero_v1 --architectures spatial --phase-zero` completed6 additional fits using the same training SHA. `runner_smoke_v1` ran2 steps, seed42, `--smoke-scenes 1`, before the primary run; it is an execution check only.

```bash
python -m track_b.acoustic_geometry.summarize --run /results/representation_v1 --out /results/summary_v1 --code-sha 6be2af1a690e039d1453e653d573607aade2cc4c
python -m track_b.acoustic_geometry.summarize --run /results/phase_zero_v1 --out /results/phase_zero_summary_v1 --code-sha 6be2af1a690e039d1453e653d573607aade2cc4c
python -m track_b.acoustic_geometry.directional_diagnostic --run /results/representation_v1 --out /results/directional_v1 --code-sha 774e7908f4f60677f33cb2476a6f75f64feb1831
python -m track_b.acoustic_geometry.figures --root /data --targets /results/targets_v3 --run /results/representation_v1 --out /results/figures_v1
```

Figures source `ed5c17a4a85a14fc20c480411838d7568339a87a`. Summaries/diagnostic used host Python3 and NumPy; rendered figures used the training image. Primary and phase-zero `COMPLETE` markers and registries confirm all18 fits. Failure directories are preserved. No19-scene holdout waveform feature cache has been prepared yet.

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

All output names must be new. Do not overwrite failed attempts. Fixed checkpoint selection uses final scheduled updates, with no historical Val19 evaluation during training or selection. Configuration, per-step training curve, per-scene metrics, per-query distributions and checkpoint checksums are written by the runner. Actual run status supersedes this prepared command and must be verified before claiming results.

## Completed follow-up executions

The same training command with `--out /results/phase_zero_v1 --architectures spatial --phase-zero` completed6 additional fits using the same training SHA. `runner_smoke_v1` ran2 steps, seed42, `--smoke-scenes 1`, before the primary run; it is an execution check only.

```bash
python -m track_b.acoustic_geometry.summarize --run /results/representation_v1 --out /results/summary_v1 --code-sha 6be2af1a690e039d1453e653d573607aade2cc4c
python -m track_b.acoustic_geometry.summarize --run /results/phase_zero_v1 --out /results/phase_zero_summary_v1 --code-sha 6be2af1a690e039d1453e653d573607aade2cc4c
python -m track_b.acoustic_geometry.directional_diagnostic --run /results/representation_v1 --out /results/directional_v1 --code-sha 774e7908f4f60677f33cb2476a6f75f64feb1831
python -m track_b.acoustic_geometry.figures --root /data --targets /results/targets_v3 --run /results/representation_v1 --out /results/figures_v1
```

Figures source `ed5c17a4a85a14fc20c480411838d7568339a87a`. Summaries/diagnostic used host Python3 and NumPy; rendered figures used the training image. Primary and phase-zero `COMPLETE` markers and registries confirm all18 fits. Failure directories are preserved. This was the status at the primary-report checkpoint; subsequent frozen holdout preparation is recorded below.

## Transfer and physical follow-up executions

The `/scale` mount is the read-only `track_b_scale_20261109` output. The `/results` mount is this study output. Use fresh output names; these commands document executed runs, not permission to overwrite them.

```bash
python -m track_b.acoustic_geometry.transfer --study /results --scale /scale --targets /results/targets_v3 --out /results/transfer_v1 --code-sha f6e9d922df2b5df31c976a6da363fb1193ee4920
python -m track_b.acoustic_geometry.summarize_transfer --run /results/transfer_v1 --out /results/transfer_summary_v1 --code-sha 334695c1d7787e64c28fb475de9659c499dedf96
python -m track_b.acoustic_geometry.render_pairs --root /data --targets /results/targets_v3 --out /results/pairs_v1 --code-sha 16ec5114d323dbd56decc5e7687da639ecaa470c
python -m track_b.acoustic_geometry.render_pairs --root /data --targets /results/targets_v3 --out /results/opening_pairs_v1 --queries 8 --opening-preserved-only --code-sha 8ee29be225f83f9a0786e08201fefa9379eeda98
python -m track_b.acoustic_geometry.evaluate_pairs --root /data --study /results --pairs /results/pairs_v1 --opening-pairs /results/opening_pairs_v1 --out /results/pair_evaluation_openings_v1 --code-sha 4d36bc4
python -m track_b.acoustic_geometry.heldout_cache --root /data --study /results --scale /scale --targets /results/targets_v3 --freeze /results/method_freeze_v1.json --out /results/development_heldout_cache_v1 --part development_holdout --code-sha 45e4ce0
python -m track_b.acoustic_geometry.evaluate_heldout --study /results --cache /results/development_heldout_cache_v1 --freeze /results/method_freeze_v1.json --out /results/development_heldout_v1 --code-sha 6907b72
```

CPU native rendering uses `track1-soundspaces-echoloc:latest`, image digest `sha256:c2907a52a8d425fc853beaa8c2fb202b577538acaee7f512b9cee9a5b8d26fc1`, explicit `--entrypoint /usr/bin/nice`, arguments `-n 15 /opt/conda/envs/soundspaces/bin/python`, `--runtime runc --network none --cpus 4 --memory 8g --user <uid>:<gid>`. Environment: `PYTHONPATH=/work`, `CUDA_VISIBLE_DEVICES=-1`, `NVIDIA_VISIBLE_DEVICES=void`, `EGL_PLATFORM=surfaceless`, `LIBGL_ALWAYS_SOFTWARE=1`, `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6`, thread limits `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=LP_NUM_THREADS=1`, `MESA_GLTHREAD=false`, `PYTHONDONTWRITEBYTECODE=1`, `MESA_SHADER_CACHE_DISABLE=true`, `HABITAT_SIM_LOG=MAGNUM_LOG=quiet`. Repository and dataset mounts are read-only; only new external study output is writable. Protocols hash every imported rendering helper. Declared source commit identifies relevant imported code even when later unrelated research commits are present.

Training and frozen GPU evaluation use `avfploc:cu124-clean`, device0, read-only repo and dataset, existing GPU services preserved. Frozen inference has no optimizer. Original and fresh inputs keep their distinct condition labels.

## Completed official Test evaluation

The same geometry image and CPU renderer launcher described above were used. `/cohort/protocol.json` is the earlier scale `cache_v1/protocol.json`; `/results/method_freeze_v1.json` is the frozen contract copied into Docs. Official raw generation completed17 scenes ×(64 furnished+64 stage-only+1 fresh repeat)=2,193 records. No model fitting followed the freeze.

```bash
python -m track_b.acoustic_geometry.prepare_targets --root /data --cohort /cohort/protocol.json --out /results/official_targets_v1 --official-test --method-freeze /results/method_freeze_v1.json --code-sha 689915c776f3e136c87031bec8dc45725b190655
python -m track_b.acoustic_geometry.render_pairs --root /data --targets /results/official_targets_v1 --out /results/official_pairs_v1 --part official_test --queries 64 --code-sha 689915c776f3e136c87031bec8dc45725b190655
python -m track_b.acoustic_geometry.heldout_cache --root /data --study /results --scale /scale --targets /results/official_targets_v1 --freeze /results/method_freeze_v1.json --out /results/official_cache_v1 --part official_test --pairs /results/official_pairs_v1 --source /work/external/checkouts/UnLoc/depth_anything_v2 --checkpoint /data/UnLoc/pretrained/depth_anything_v2_metric_hypersim_vitl.pth --code-sha 45e4ce0
python -m track_b.acoustic_geometry.evaluate_heldout --study /results --cache /results/official_cache_v1 --freeze /results/method_freeze_v1.json --out /results/official_evaluation_v1 --code-sha 6907b72
python -m track_b.acoustic_geometry.verify_heldout --study /results --cache /results/official_cache_v1 --run /results/official_evaluation_v1 --freeze /results/method_freeze_v1.json --out /results/official_verification_v1 --code-sha 1e9b5b9ecf6f185f483486a26f7e339809d5a8bd
python -m track_b.acoustic_geometry.report_heldout --run /results/official_evaluation_v1 --out /results/official_report_v1 --code-sha 3ee938b
```

Full resolved source SHAs and preparation/evaluation protocols are in [execution record](official_execution.json). Renderer source689915c was loaded before the later opening-control code edit: every official manifest record uses the earlier schema without `render_geometry_sha256`, as checked during execution; raw stage/config/registry hashes are retained. Existing unrelated dirty Gibson files are not part of these imported modules. Python source/externally installed renderer helpers, dataset assets and model weights must be provided for reproduction; a clean repository clone alone is insufficient. Original output directories must never be overwritten.

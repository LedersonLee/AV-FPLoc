# Reproduction

Working directory `/home/jeongeonlee/track1/AV-FPLoc`. Python is provided by existing Docker image `avfploc:cu124-clean`, pinned SHA in provenance. PyTorch2.4.1+cu124, NumPy1.24.4, RTX4090. No packages downloaded for runs. Use fresh output paths; training scripts refuse existing directories. Source commits differ by run and are recorded in registry. Replaying the latest script does not reconstruct the original diagnostic bug.

Executed container pattern (replace OUTPUT with a fresh directory for reruns):

```bash
docker run --rm --network none --gpus 'device=0' \
  --user "$(id -u):$(id -g)" -e OMP_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2 \
  -v /home/jeongeonlee/track1/AV-FPLoc:/work:ro \
  -v /home/jeongeonlee/track1/experiments/track_b_structure_20261109:/out \
  -v /file2/jeongeon/AV-FPLoc/datasets/HSSD:/data:ro \
  -w /work --entrypoint python avfploc:cu124-clean \
  -m track_b.scripts.train_structure --cache /out/cache --out /out/OUTPUT
```

Module arguments for each run:

```text
supervised_v1: --modes visual acoustic fusion
shuffled_v1: --modes fusion --arm shuffled
dense_only_v1: --modes fusion --arm dense_only
frozen_v1: --modes fusion --regime frozen
partial_v1: --modes fusion --regime partial
matched_frozen_v1: --modes fusion --regime frozen --aux-steps 0 --dense-steps 1000 --pretrained-dir /out/frozen_v1 --skip-sampling-steps 1500
matched_partial_v1: same matched arguments, --regime partial
matched_joint_v1: same matched arguments, --regime joint
```

All use seeds42/43/44. Original five families use default1,500aux+1,000dense; dense_only treats all2,500 as dense. Matched families reuse frozen_v1 archived pretraining; SHA256 recorded per seed. This resets AdamW for all three conditions and matches sampling order, but does not promise bitwise reproducibility of CUDA training. Old frozen and partial warmup checkpoints were not bitwise equal at any seed; do not replace the strict matched study with those earlier runs.

Other executed modules (same mounts, additional audit mount for cache builder):

```text
track_b.scripts.structure_cache --root /data --audit /audit --out /out/cache
track_b.scripts.profile_structure --cache /out/cache --out /out/profile.json
track_b.scripts.extended_structure --root /data --cache /out/cache --run /out/supervised_v1 --out /out/extended_v1
track_b.scripts.correct_structure_diagnostics --cache /out/cache --run /out/supervised_v1 --out /out/corrected_ray_diagnostics.json
```

Prior audit is `/home/jeongeonlee/track1/experiments/track_b_20261109/audit_v1` (verify retained audit directory against previous report before rebuilding). Existing audited cache manifest pins each serialized scene and original schedule/map hashes. `*_inputs.json` outside Git pins each RGB/RIR. Baseline `track_b.scripts.unloc_reference --help` exposes paths for original checkpoint and external source; provenance stores hashes. Mount dataset also at its original absolute path because external DepthAnything symlink points there. Baseline assets are not supplied by a clean clone.

Validation executed inside the image:

```bash
python -m pytest -q -p no:cacheprovider track_b/tests
```

Aggregate small results, bootstrap and hashes on host:

```bash
python3 -m track_b.scripts.summarize_structure \
  --root /home/jeongeonlee/track1/experiments/track_b_structure_20261109 \
  --out docs/research/track-b/experiments/2026-11-09-structure-supervision
python3 docs/research/tools/validate_archive.py
```

Original logs and checkpoints remain external. First aggregation attempt failed because early protocols predated the encoder_regime field; fixed aggregation uses explicit joint default for these legacy runs. No model rerun or result overwrite was needed. Auxiliary diagnostics correction replays rather than overwrites original reports.

Baseline inputs: `/work/external/checkouts/UnLoc`, `/data/UnLoc/evaluation_checkpoints/epoch020/model.ckpt`; external checkout base `3c7a301aa08049f6b1bb86f110e8e6e675e2300b`, preexisting dirty depth modules and DepthAnything submodule. The historical checkpoint is used only in isolated baseline evaluation.


Independent fusion executes at code commit `99caa09` (full SHA in provenance), using the same Docker mounts:

```text
python -m track_b.scripts.independent_fusion --cache /out/cache --run /out/supervised_v1 --out /out/independent_fusion_v1
```

This does no network training. It sweeps15 weight/temperature combinations, selects calibration parameters without the target scene, and re-scores full banks for fixed/calibrated/shuffled conditions. External `sweep.json` and `predictions.json` are retained, not uploaded. No Test access. Fine accuracy is positional only, as in earlier experiments.

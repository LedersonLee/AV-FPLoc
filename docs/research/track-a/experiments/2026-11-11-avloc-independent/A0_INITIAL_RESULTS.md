# A0 seed42 completed training and initial evaluation

Date: 2026-11-12 KST. Track A. Evidence: Exploratory, reused Val19 development and checkpoint selection. Full research remains incomplete.

## Observed results

A0 completed 100 epochs (max-epochs criterion). Best clean-Val ray MAE selected epoch99, 0.1795708863 m. Frozen FP32 prediction covers all19 Val scenes and9,965 queries. Checkpoint SHA256: `03942eba07916c8da6d494319b484c830e74a5740793765e864433ce58b091f7`. Training source: `62c58da20ed26ba95863638bbfd4926829999eec`.

On the shared20cm XY/10° yaw grid: R@1m50.9483%, Joint@1m&30°50.1054%, R@0.1m22.0271%, NMS Top5 coverage70.9784%, NMS Top20 coverage88.7506%. Scene-macro Joint49.6699%. FP32 ray MAE0.1795708835 m replays the selected validation metric within numerical precision. Native normalized entropy0.04297 and ECE0.39337 show that concentration alone must not be called calibrated confidence.

[Machine-readable aggregates and provenance](a0_seed42_initial_results.json). These are single-seed descriptive outcomes; paired scene intervals and matched baseline integration remain pending. No independent generalization or causal interaction conclusion follows. A1/A2/A3, multiple seeds, fusion/interaction ablations and10cm/calibration analyses remain required. Test17 was not used.

## Failure and correction

The original evaluation queue successfully completed prediction and20cm matching, then stopped in post-hoc ray analysis: NumPy `ptp` cannot subtract boolean localization outcomes. This was an analysis-type error, not a model or matching failure. Convert correlation inputs to floating point before finite filtering/range checks. Three regression tests cover varying boolean outcomes, constant boolean outcomes and NaN filtering; all passed. Corrected full19-scene ray analysis also passed all query-ID, source-hash and per-query MAE replay checks.

Correction/recovery source: `5f8fdc4e8fdd22a17f1a243646c263fcd23e9ae1`. The failed analysis directory and logs are preserved. Recovery copies the immutable original evaluation source snapshot, changes only `analyze_rays.py`, verifies receipts for the two completed stages, writes20cm analysis to a new `_v2` directory, then continues10cm evaluation/analysis and both calibration sweeps. All recovery stages stop on error and reject existing destinations. No retraining or checkpoint change was made. A1 continues independently.

## Reproduction

Set WORKSPACE to the workspace root and STUDY to the local large-artifact root. The prior queue snapshot and its receipts are required; this is not a clean-clone-only data release.

```bash
python3 AV-FPLoc/scripts/research/track_a/avloc_20261111/test_ray_correlation.py
python3 AV-FPLoc/scripts/research/track_a/avloc_20261111/recover_a0_evaluation.py --previous "$WORKSPACE/experiments/avloc_20261111/a0_evaluation_queue_v1" --out "$WORKSPACE/experiments/avloc_20261111/a0_evaluation_recovery_v2" --study "$STUDY"
```

Executed with the original pinned Docker image recorded in the recovery plan. Existing outputs intentionally cause refusal; never overwrite them to rerun. Inference49.06s was measured concurrently with A1 training and is not an isolated runtime benchmark. Reports contain aggregates only; checkpoints, query arrays and logs stay local.

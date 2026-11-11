# A1 seed42 training completion and evaluation launch

2026-11-12 KST. Track A. Exploratory: reused Val19 development. Full benchmark incomplete.

A1 acoustic-only36-ray model completed70 training epochs on112Train/58,894 queries; selected epoch50 by clean Val19 ray MAE1.4785059889m. Twenty subsequent completed epochs did not improve the selection metric. This is not yet a localization or acoustic-observability conclusion: FP matching, direction/distance errors and uncertainty analyses remain required. Do not compare the36-ray MAE directly to A0's11 in-FoV rays as evidence for modality superiority.

## Receipt discrepancy and correction

The original complete.json says last_epoch71. Inspection of train.py shows the for-loop assigns71 before checking the exhausted patience and breaking. The last executed epoch file is epoch_070.json with patience_used20; no71 epoch metrics were written. Thus the precise statement is70 completed epochs and a summary-counter off-by-one. Preserve the original receipt and checkpoints. Training, stopping decisions and selected best checkpoint are unaffected. Live/queued immutable training source has not been altered; downstream reporting must use completed epoch files. [Original receipt, correction and hashes](a1_training_completion.json).

Training source `62c58da20ed26ba95863638bbfd4926829999eec`. A2seed42 has automatically started. A1 evaluation runner `8e77076c4f9c0f7cb0db32fceb2b230a3f16f31b` verifies completed-run status, exact training source hashes and fresh destinations, then freezes a source snapshot and uses the pinned image. It runs FP32 prediction,20cm evaluation/ray analysis,10cm evaluation/ray analysis and both temperature sweeps, with all9,965 queries required per stage and stop-on-error. The boolean-correlation correction is included. No Test17 access; no retraining or new checkpoint selection.

```bash
python3 AV-FPLoc/scripts/research/track_a/avloc_20261111/evaluate_completed_expert.py --workspace "$WORKSPACE" --study "$STUDY" --out "$WORKSPACE/experiments/avloc_20261111/a1_evaluation_queue_v1" --arm A1 --seed 42
```

Requires existing local assets and completed training artifacts; outputs must not exist. Plan and logs remain local under a1_evaluation_queue_v1. Runtime overlaps A2 training and is not an isolated cost measurement. Quantitative matching, controls, uncertainty intervals and interpretation will be appended after completion; this report is not a research stopping condition.

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


## Initial20cm matching: negative result

All19 Val scenes/9,965 queries completed prediction,20cm matching and post-hoc ray analysis. R@1m9.2524%, Joint2.4285%, R@0.1m0.1505%, NMS Top5 coverage9.9147%, Top20 coverage22.6894%. The same-cohort FP-RIR cosine baseline is30.376% R@1m and21.365% Joint. [Matched table](A0_A1_MATCHED_COMPARISON.md), [aggregation provenance](a0_a1_matched_comparison.json). Current A1 does not support the hypothesis of strong acoustic-only localization. No positive cases were selected to override this aggregate failure.

Ray MAE1.478506m reproduces selected-checkpoint validation. In-FoV1.491381m, outside1.472841m. Pooled scale/absolute-error Spearman0.374921 is descriptive and does not establish calibrated uncertainty. [Direction/distance/interval diagnostics](a1_ray_diagnostics_20cm.json).

Competing explanations include insufficient learned directional information, adaptation/optimization limitations and remaining input/coordinate/model errors. [Prediction variability diagnostic](a1_prediction_variability.json) rules out literally identical outputs across queries: equal-scene mean of per-direction query std is1.0444m versus target2.1992m. Within-query directional std is only0.4132m versus target1.9611m; mean predicted distance2.3552m versus target2.9543m. These observations are consistent with directional smoothing and distance underprediction, but do not identify a causal mechanism or prove an input error. Full distribution/perturbation controls and independent-fusion/interaction-disabled comparisons remain required. No architecture change or new checkpoint selection has been made from these results.

Variability is an ad-hoc read-only NumPy analysis, reproduced by the formulas and source manifest hash in its JSON; it has no separate historical experiment-code commit. Source context is the committed prediction/analysis pipeline, training SHA above.10cm and temperature analyses remain in progress; A2 is still training. Do not interpret this single-seed result as failure of the original paper or physical non-observability.

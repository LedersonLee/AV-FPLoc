# A2 seed42 completed training; matched evaluation pending

2026-11-12 KST, Track A, Exploratory/reused Val19. Full research remains incomplete.

The joint visual/audio36-ray model completed99 epochs. Best clean-Val19 Ray MAE is0.5608134521m at epoch79;20 subsequent completed epochs did not improve. The unchanged training implementation records last_epoch100 because it assigns the next loop index before checking exhausted patience. Actual epoch_099.json and patience_used20 establish99 executed epochs. This is the same reporting-only counter discrepancy documented for A1; original receipts and weights are preserved.

[Completion, selected checkpoint and provenance hashes](a2_training_completion.json). Training source `62c58da20ed26ba95863638bbfd4926829999eec`,112Train/58,894queries,Val19/9,965queries. No Test17 access.

Observed: A2's36-ray validation MAE is lower than A1's1.478506m. Interpretation is limited: this alone does not show localization superiority, causal cross-modal interaction or improved calibration. A0 uses only11FoV rays, so its overall MAE is not directly comparable. Compare in/outFoV errors and matched localization after frozen evaluation. Independent equal fusion and interaction-disabled controls, plus seed43, remain necessary.

A0seed43 training started automatically. A2 evaluation was launched with `evaluate_completed_expert.py` source `8e77076c4f9c0f7cb0db32fceb2b230a3f16f31b`: frozen FP32 predictions,20cm and10cm matching, ray diagnostics and temperature sweeps. It verifies training-source hashes and completed status, creates an immutable source snapshot, refuses existing outputs and stops on failure. Expected19scenes/9,965queries each stage. Runtime overlaps seed43 training and is not isolated cost evidence.

```bash
python3 AV-FPLoc/scripts/research/track_a/avloc_20261111/evaluate_completed_expert.py --workspace "$WORKSPACE" --study "$STUDY" --out "$WORKSPACE/experiments/avloc_20261111/a2_evaluation_queue_v1" --arm A2 --seed 42
```

Local plan/logs: a2_evaluation_queue_v1; large outputs under the configured study root. Model fitting is not rerun by evaluation. Quantitative matching/ablation conclusions and A3 gate review remain pending; training completion is not benchmark completion.


## Completed20cm evaluation: accuracy tradeoff, not uniform superiority

All19scenes/9,965queries completed frozen prediction,20cm matching and ray analysis. R@1m65.7200%,Joint64.9272%,fine30.4666%,NMS Top5coverage85.5093%,Top20coverage94.1796%. The prediction ray MAE0.560813455m replays the selected-checkpoint value. FoV MAE0.177372m,outsideFoV0.729528m. [Ray/uncertainty aggregates](a2_ray_diagnostics_20cm.json). Native entropy0.009699 and nearest-grid NLL154.848 show that sharpness and accurate confidence remain separate questions; temperature analysis is still pending.

[Full matched comparison](A2_MATCHED_COMPARISON.md), [source hashes](a2_matched_comparison.json), [paired contrasts](a2_paired_contrasts.json). Contrast code `48882b376ec50fffdaab385ec6e94a0ccbf02e3f`; exact scene/query alignment and saved-source hashes,10,000paired scene bootstrap draws, seed20261111.

| A2 minus comparator | Micro Joint delta | Macro Joint delta [95% scene CI] | Micro fine delta | Macro fine delta [95% scene CI] |
|---|---:|---|---:|---|
|A0|+14.822pp|+14.852[12.304,17.218]pp|+8.440pp|+8.525[6.826,10.229]pp|
|Independent equal fusion|+55.374pp|+54.929[49.963,59.189]pp|+29.985pp|+29.765[26.383,33.196]pp|
|Preserved visualFoV+audiooutside|+12.624pp|+12.568[10.424,14.727]pp|+7.928pp|+8.004[6.488,9.533]pp|
|B2+GEOQ seed42/43 mean|−8.144pp|−8.211[−10.045,−6.477]pp|+7.265pp|+7.089[5.418,8.704]pp|

A2 repairs2,172 A0 joint failures and harms695 successes. Against B2+GEOQ, mean paired repairs967/harms1,778.5 for joint, and1,641.5/917.5 for fine. Fractions arise from averaging two baseline seeds, not fractional queries. A2 currently has one training seed, so these intervals do not estimate training-seed variability.

Observed: joint training beats the tested independent frozen-distribution controls and improves fine localization relative to the existing GEOQ pipeline, while GEOQ retains higher1m/yaw success. Interpretation: the tradeoff depends on the metric; neither method dominates. The differences do not isolate cross-modal attention causality: A2_disabled is still pending, training data/teacher histories differ from Track A, and all checkpoints/protocols reused Val19.10cm results must remain separate. No pristine-independent or original-paper replication claim follows from this6ch adaptation.

Next: complete10cm/ray/calibration review, then audit the A0/A1/A2 expert gate for A3. Seed43 and interaction-disabled controls remain required regardless of this positive A2 result. The report is not a stopping condition.

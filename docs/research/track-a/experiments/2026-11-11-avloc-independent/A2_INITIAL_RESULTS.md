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

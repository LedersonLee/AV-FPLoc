# Frozen-score interventions: completed results

2026-11-09. All9 frozen models×6 reused Val scenes×64 queries×4 conditions completed,13,824 records. Evaluator and independent registration exited0. All3,456 normal and3,456 both-versus-initial metric replay checks passed. Evidence remains Exploratory; no retraining, selected hyperparameter or independent Test evaluation.

| Architecture | Intervention | Joint % | Fine % | Raw20 % | NLL |
|---|---|---:|---:|---:|---:|
| attention | normal | 16.753 | 0.781 | 50.087 | 9.232 |
| attention | initial_scales | 19.358 | 0.781 | 53.993 | 9.337 |
| attention | zero_interaction | 22.309 | 1.562 | 58.594 | 9.475 |
| attention | both | 23.872 | 2.170 | 61.892 | 9.580 |
| concat | normal | 17.101 | 0.868 | 50.434 | 9.249 |
| concat | initial_scales | 19.444 | 0.868 | 53.559 | 9.357 |
| concat | zero_interaction | 22.309 | 1.562 | 58.594 | 9.475 |
| concat | both | 23.872 | 2.170 | 61.892 | 9.580 |
| separable | normal | 16.319 | 0.694 | 50.000 | 9.245 |
| separable | initial_scales | 19.097 | 0.694 | 53.385 | 9.353 |
| separable | zero_interaction | 22.309 | 1.562 | 58.594 | 9.475 |
| separable | both | 23.872 | 2.170 | 61.892 | 9.580 |

## Within-model intervention evidence

With final interaction retained, restoring original scales improves Joint by2.604pp attention[1.649,3.385],2.344pp concat[0.868,3.646],2.778pp separable[1.649,3.993]. With final scales retained, removing learned gate/residual improves Joint by5.556pp attention[3.299,7.813],5.208pp concat[3.125,7.465],5.990pp separable[4.427,7.726]. These are95% paired scene bootstrap intervals, seeds averaged within scene; six scenes and uncorrected multiple contrasts limit inference.

At original scales, removing learned gate/residual still improves Joint by4.514pp attention[1.997,7.118],4.427pp concat[2.257,6.597],4.774pp separable[3.038,6.597]. With interaction absent, restoring scales adds1.563pp[0.521,2.517] for each architecture (reported means coincide). Thus neither scale drift alone nor encoder drift alone explains the loss; the learned gate/residual also carries a negative contribution under these fixed checkpoints. The effects are conditional, not additive percentages of a uniquely identified cause.

Restoring both recovers F1 Joint23.872%, fine2.170%, raw20 61.892%, NLL9.580. Normal frozen fields achieve lower NLL around9.23–9.25 while worse Joint/fine/raw coverage. This is direct evidence of a ranking/precision versus nearest-grid NLL tradeoff under this training schedule, not proof that NLL is intrinsically unsuitable or that calibration improves. The both control is an expected algebraic recovery of the initialization because precursor descriptors/maps were frozen; it is a consistency check, not a new method.

The recorded scale tensors are finite and inside the0…8 clamp. For attention seeds42/43/44, acoustic effective scale rises from about69.3 to94.8–95.0 while visual scale changes only about61.0–61.6 to61.3–63.0. This changes relative modality weighting even though nominal.85/.15 weights are fixed. It is an observed parameter change; interpreting it as over-trust of acoustics remains a hypothesis. The scale-only intervention supports a contribution to the measured regression, but gate and residual are removed together and are not separately identified. Full raw/effective values for all9 models are in completed_scales.json.

## What this does and does not establish

The inference-time interventions isolate two changed parameter groups in a system with independently verified fixed encoders/maps. They support the conclusion that both groups contribute to the regression on this reused cohort. They do not establish physical acoustic reasoning, a learning-trajectory mechanism, superiority of attention over concat, or out-of-domain generalization. Disabling the learned interaction restores a strong baseline but removes the intended interaction contribution; it is not a successful M1 result. Future retraining with constrained scales or an alternative ranking objective would be a new experiment, not implied by these counterfactual inference results. Current112-scene factorial baseline study continues unchanged.

Fine and raw20 gains/losses are retained in completed_summary/paired.json. No output filtering, GT injection, candidate pruning or validation-based parameter selection. Full posterior confidence/cost and severe shift conclusions are not newly established by this study. Repeated six-scene evaluation and upstream/historical exposure remain limitations. Overall goal remains incomplete.

## Validation and reproduction

All condition/model/seed/query keys exactly match the initial F1 cohort. Source checkpoint hashes and training protocol hash match the declared frozen models; feature/cache protocols also match. Normal and both replays verify Joint/R@1m/fine/raw20/NLL within declared tolerance, not bitwise equality of every score or selected index. Every map cache was recomputed after parameter intervention. No model weights or original prediction files were overwritten. No failed full invocation; smoke32 records remain separately archived and were not treated as full evidence.

Evaluator source `d8ca9d8b836f1c7178ce07056faf3b9f96af9c2f`; registrar source `a5a0e48a0fa012f895dfefaf43cf2b6521fe3d1f`; frozen training source062860067f82a5f6a0e6ce8411b5c43702ae0ab5. Commands and environment are in README.md. Register with `python3 -m track_b.scripts.register_frozen_score_controls --run <frozen_score_controls_v1> --training <foundation_frozen_v1> --reference <foundation/independent_fusion_anchor_v1> --out <fresh-registration>`. Host NumPy registrar, GPU evaluator avfploc:cu124-clean; fixed inputs,2 CPU threads. Tensor diagnostic reads the same checkpoint hashes and reports raw and exp(clamp(value,0,8)) for log_scale/acoustic_log_scale before/after training. External outputs frozen_score_controls_v1 and frozen_score_registered_v1; raw predictions/weights remain outside Git.

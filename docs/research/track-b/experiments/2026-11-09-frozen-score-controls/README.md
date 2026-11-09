# Frozen-score scale × interaction intervention

Study28,2026-11-09,Track B,Exploratory. Status: full nine-model evaluation and registration completed. [Results](completed_results.md): both scale drift and learned gate/residual contribute to the regression. Source `d8ca9d8b836f1c7178ce07056faf3b9f96af9c2f`. This is a post-hoc mechanistic diagnostic motivated by the completed27-fit encoder-boundary failure in study24; it is not independent validation or a newly trained method.

## Question and competing explanations

Frozen precursor encoders still lose6.77–7.55pp Joint relative to initial F1. Candidate explanations are changed visual/acoustic score scales, the learned acoustic gate/additive interaction residual, or their combination. Because frozen descriptor/map tensors are already verified unchanged, restoring selected inference parameters can localize which changed components carry the observed score-field regression. It cannot prove a counterfactual learning trajectory, physical mechanism, or generalization.

## Fixed four conditions

Apply to all9 frozen checkpoints (attention/concat/separable×42/43/44),6 existing Val scenes,64 observations/scene, same.25m/36yaw full bank. No training or Test access.

1. normal: unmodified final frozen checkpoint.
2. initial_scales: restore both log_scale and acoustic_log_scale from that checkpoint's exact archived initial state; all other parameters stay final.
3. zero_interaction: retain final scales; set final gate and joint-head Linear weights/bias to zero. The explicit formula then yields acoustic gate1 and additive residual0, preserving fixed.85/.15 independent matching.
4. both: combine scale restoration with gate/residual removal; should reproduce the original independent F1 field metrics because precursor encoders/maps stayed fixed.

Map caches are recomputed after each state intervention. Observations, bank, labels and metrics do not change. GT appears only after full-field inference in evaluation; no GT candidate insertion or filtering. Head inputs/retrieval projections remain final but their final zero outputs cannot influence condition3/4 scores. Nominal capacity is unchanged; the controls deliberately alter effective inference computation. They are not matched retraining ablations.

Planned13,824 records,3,456 normal replay checks and3,456 both-versus-initial metric checks. Candidate/query keys, checkpoint/cache/precursor hashes are fixed. The evaluator preserves mismatch outputs and rejects any mismatch rather than silently claiming a successful replay. Primary Joint, secondary fine/raw20/NLL with paired scenes and seeds averaged; retain all extended metrics. Final contrasts include scale restoration with interaction present(initial_scales−normal) and absent(both−zero_interaction), removal with final scales(zero_interaction−normal) and initial scales(both−initial_scales). Full coverage and complete registries are required before interpretation; uncorrected secondary intervals are descriptive. No condition is selected or retuned from the smoke.

## Executed validation and scope

The smoke used first mode/seed/scene and first8 query IDs:32 condition records. Original final-model8 metrics and both-versus-initial8 metrics replayed with zero mismatches at declared floating tolerance; exit0. Source hashes are in smoke_protocol.json. No smoke bootstrap intervals are computed. Its numeric outcomes are runtime evidence only, not a scientific test. Four-arm design was encoded before the smoke; full execution uses it unchanged. No failure in the smoke invocation.

Full command: `python -m track_b.scripts.evaluate_frozen_score_controls --run <interaction/foundation_frozen_v1> --cache <structure/cache> --features <foundation/features_v1> --reference <foundation/independent_fusion_anchor_v1> --out <fresh-output> --code-sha d8ca9d8b836f1c7178ce07056faf3b9f96af9c2f`. Smoke adds `--smoke`. GPU Dockeravfploc:cu124-clean,2 CPU threads, non-root/no network, inputs/code read-only. External full output /home/jeongeonlee/track1/experiments/track_b_interaction_20261109/frozen_score_controls_v1, session16172. Raw predictions/checkpoints/logs remain external; original models are never overwritten.

Checkpoint training source062860067f82a5f6a0e6ce8411b5c43702ae0ab5; initial F1 provenance in study19/20 and study24. Full-code reproduction requires those recorded external artifacts; a clean clone alone is insufficient. Reused six-scene Val, historical/upstream exposure, multiple diagnostics and limited interventions remain limitations. No new accuracy claim, final model selection or overall completion is implied. Full results, uncertainty, failure evidence and next scientific decision will be appended after completion.

## Completion

Full13,824 records,3,456 normal and3,456 initial metric replays passed. Completed results and all four conditional contrasts are archived; no overall scientific success or interaction benefit is claimed. Prior planned/runtime descriptions are retained as provenance.

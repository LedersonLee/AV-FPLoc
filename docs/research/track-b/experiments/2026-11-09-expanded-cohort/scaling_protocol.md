# Frozen2×2 scaling comparison — running

2026-11-09. Plan frozen before expanded-cohort model outcomes. Structural and visual-feature verification completed before launch. All131 scenes/8,384 queries passed; all22 overlapping pilot feature tensors matched exactly. This is an independent V/A/F1 baseline scaling study, not a substitute for the proposed M1 architecture, and not an independent Test result.

## Question and competing explanations

More scene diversity may improve held-out development performance. Alternatively apparent scaling gains may come from extra optimization; apparent losses may come from fewer updates per training query. A2×2 design varies Train scene count and update budget independently, holding the full19-scene development set fixed. The first16 SHA256-ordered Train scenes exactly match the old pilot. All112 scenes are the already audited eligible Train pool. Val19, input representation, optimizer, losses, seeds and scheduled final-checkpoint rule are unchanged between cells.

| Train scenes | Budget | Auxiliary steps | Dense steps | Expected aux draws/query | Expected dense draws/query |
|---:|---:|---:|---:|---:|---:|
|16|1|1,500|1,000|11.71875|7.8125|
|112|1|1,500|1,000|1.67411|1.11607|
|16|7|10,500|7,000|82.03125|54.6875|
|112|7|10,500|7,000|11.71875|7.8125|

Batch8 with uniform scene/query sampling and64 queries per scene. “Draws” are expected sampling exposure, not unique examples or complete epochs. Budget scales auxiliary and dense learning together, so it does not isolate dense-only optimization. Compare112vs16 at each fixed update budget; budget7vs1 at each fixed scene count; and112/budget7vs16/budget1 at matched expected draws/query. Wall-time/host-transfer cost is not guaranteed equal. No expanded outcomes have selected these contrasts.

## Fits and fixed settings

Four cells × seeds42/43/44 × three stages =36 stage fits, giving24 final modality models plus12 auxiliary visual fits. Every cell uses CPU-resident scene transfer consistently. Each stage saves the exact initial checkpoint and records its hash and actual source SHA. Fresh initial visual/acoustic model states can therefore be checked across scene counts and budgets; dense visual initializations intentionally come from their own cell's auxiliary fit and need not match across cells.

1. Fresh foundation visual adapter/FP model, auxiliary-only budget above. Frozen, separately sourced upstream encoder; no HSSD-trained head.
2. Continue that cell/seed's visual auxiliary checkpoint with a fresh AdamW, dense budget above, visual auxiliary1000 and raw-geometry FP anchor1000. Skip the corresponding auxiliary sampling draws so continuation follows the declared sample order.
3. Fresh acoustic/FP model, auxiliary and dense budgets above, acoustic auxiliary10.

All lr3e-4, weight decay1e-4, batch8. Inference combines independently trained visual/acoustic fields with fixed.85/.15 weights. Full19 Val is used in every cell, with original6 and added13 reported as predefined secondary strata. No epoch/model/grid/calibration tuning from expanded Val. Data size does not change the definition of the validation cohort. Final evaluations and exact input-cohort compatibility checks remain required; no fitted/calibrated result is implied by this protocol file.

## Implementation safeguards and executed checks

`--train-scene-count` restricts only the frozen Train prefix and preserves every Val scene. It rejects invalid counts and cannot be combined with the separate OOF-fold option. Pretrained continuation checks the effective Train/Val lists for equality, preventing accidental reuse of a head trained on another cell. Independent fusion also checks effective visual/acoustic cohort and seeds, not just a shared larger cache hash. New classes keep scene/query IDs outside model inputs.

43 tests passed. On existing cached data, an8-Train/6-Val one-step auxiliary fit and its one-step dense continuation both completed, exit0, verifying the subset path and same-cohort continuation. A separate one-aux/one-dense recording smoke completed, exit0; its recorded source and initial checkpoint SHA match the actual saved artifact. These three probes are runtime checks, not accuracy comparisons or expanded training. The earlier residency bitwise failure remains preserved with its repeat diagnostic; this design does not claim that short-run numerical equality proves long-run determinism.

## Start gate and remaining work

Before any scaled fit, require all131 expanded visual-feature artifacts, hashes/query alignment and exact comparison to the original22 pilot feature tensors. The verifier checks the upstream checkpoint, source files, preprocessing and normal-augmentation settings; the older normal-cache schema's absent brightness/occlusion fields are explicitly interpreted as1/0, consistent with its original unmodified-image construction. It does not silently accept a shifted cache.

`scaling_plan.json` contains all cells, stage settings, cohorts, source cache/verification hashes and contrasts. Full feature validation passed, exit0. The guarded runner started n16_budget1_visual_aux; subsequent completion, initial-state comparisons, paired scene uncertainty, raw-input controls, final costs and scientific decisions remain pending. The full19 Val set has historical upstream/research exposure; “added13” does not mean independent validation. Test is not opened during this design.

## Provenance and reproduction

Subset/feature-verifier source `e7d571e63d578f498ec3d6c457c7cc4ddeff2620`; actual subset smokes use that source. Initial-state recording and frozen-plan source `412e93458c9dcf04ff15b731c45323ff715765a7`; the recording smoke uses this source. Independent fusion cohort guard source713de85 (full identity in provenance). Shared checkout also contains concurrent Track A commits; this report identifies the relevant code content and never substitutes a Docs commit for experiment code.

Freeze plan with `python -m track_b.scripts.freeze_scaling_plan --cache <expanded-cache> --pilot <old-cache> --validation <completed-cache-validation> --out <fresh-plan-directory>`. Verify features with `python -m track_b.scripts.verify_expanded_features --features <expanded-features> --cache <expanded-cache> --pilot <original-foundation-features> --out <fresh-verification-directory>`. Training uses existing train_structure plus `--train-scene-count 16` or `--train-scene-count 112`, `--scene-residency cpu`, `--save-initial-state` and `--code-sha <actual-source>`; CLI flags and values must be separate arguments. Full stage counts are in the JSON plan. Do not overwrite outputs or treat a partial feature directory as complete.

External root remains `/home/jeongeonlee/track1/experiments/track_b_scale_20261109`. Raw data/weights and detailed predictions stay outside Git. No new remote, paid resource or external data upload is involved. Overall research goal is active and incomplete.

Guarded runner source `5cf803681c2d144f3a620bc00c1ab6b841e48e7c`; launch session19640. Run `python -m track_b.scripts.run_scaling_plan --root <expanded-root> --out <fresh-scaling-output> --code-sha 5cf803681c2d144f3a620bc00c1ab6b841e48e7c`. The executor verifies data gates, locks against duplicate runners, checks dependency hashes before each stage and preserves failed attempts. Completed stages are reusable only with matching configuration and saved initial-state hashes. Runtime dependency hashes are copied in scaling_runner_protocol.json; shared checkout HEAD can change because another task works concurrently.

## Complete-factorial comparison gate

`compare_scaling_cells.py` implements only the five frozen contrasts:112vs16 at budget1 and7; budget7vs1 at16 and112;112/budget7vs16/budget1 at matched expected draws/query. It requires all36 verified stage fits,12 verified continuation initializations, identical19-scene/seed/query/method keys, fixed settings, model hashes and each cell's completed registry. Paired scene bootstrap averages seeds inside scene; intervals are uncorrected, with secondary metrics/strata descriptive. It does not select a winner, checkpoint or new hyperparameter.

The negative gate probe supplied the valid first-cell-only fit verification and was intentionally rejected before creating an output directory (exit1, expected). This tests refusal of incomplete factorial evidence; it is not an experimental failure or an executed four-cell comparison. Source and result are in factorial_gate_probe.json. Successful full comparison and its scientific conclusions remain pending. Reproduce after all cells: `python -m track_b.scripts.verify_scaling_fits --root <root> --run <scaling_fits_v1> --out <all-cell-verification>` then `python -m track_b.scripts.compare_scaling_cells --root <root> --fit-validation <all-cell-verification> --out <fresh-comparison>`. Each cell first needs fixed-only extended evaluation and register_scaling_cell outputs named `<condition>_evaluation_v1` and `<condition>_registered_v1`.

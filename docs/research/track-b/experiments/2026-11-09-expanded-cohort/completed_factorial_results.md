# Completed scene-count × update-budget comparison

2026-11-09. All36 scheduled stage fits,24 final modality models, four full19-scene evaluations and cross-cell provenance checks completed. Fixed F1 uses independently trained visual/acoustic/dual-map encoders, not M1 interaction. Evidence remains Exploratory on reused development data. No Test observations, epoch selection or calibration tuning were used.

## Main finding and competing explanation

Increasing the frozen Train cohort from16 to112 scenes improves fixed-fusion Joint at both matched update budgets. Increasing updates alone does not reproduce that gain. This distinguishes the scene-set intervention from simply spending more optimization steps in this design. It does not establish that scene count alone is universally causal: the extra scenes also change content/diversity/source coverage, only one nested SHA-ordered16-scene subset was tested, and every scene has64 sampled queries.

| Train / budget | Method | Joint % | Fine % | Raw20 % | Diverse20 % |
|---|---|---:|---:|---:|---:|
| n16_budget1 | visual | 20.285 | 1.288 | 65.104 | 81.031 |
| n16_budget1 | acoustic | 4.276 | 0.164 | 20.861 | 43.640 |
| n16_budget1 | fixed | 27.166 | 1.754 | 61.760 | 84.704 |
| n16_budget7 | visual | 21.848 | 1.974 | 66.694 | 82.867 |
| n16_budget7 | acoustic | 3.893 | 0.082 | 20.038 | 41.996 |
| n16_budget7 | fixed | 25.027 | 1.398 | 57.319 | 82.100 |
| n112_budget1 | visual | 21.848 | 1.508 | 67.928 | 81.552 |
| n112_budget1 | acoustic | 7.045 | 0.219 | 29.523 | 58.114 |
| n112_budget1 | fixed | 34.402 | 3.015 | 70.477 | 89.666 |
| n112_budget7 | visual | 24.726 | 2.138 | 70.724 | 84.485 |
| n112_budget7 | acoustic | 7.840 | 0.274 | 29.825 | 59.978 |
| n112_budget7 | fixed | 33.607 | 2.632 | 68.366 | 89.611 |

| Fixed-fusion contrast | Joint difference pp | 95% scene interval |
|---|---:|---:|
| data_at_budget1 | +7.237 | [+5.455, +8.882] |
| data_at_budget7 | +8.580 | [+5.839, +11.486] |
| budget_at_n16 | -2.138 | [-3.591, -0.630] |
| budget_at_n112 | -0.795 | [-2.549, +0.905] |
| matched_draws_per_query | +6.442 | [+3.810, +9.265] |

At matched updates, scene expansion improves fine by1.261pp[0.603,1.919] at budget1 and1.234pp[0.548,1.974] at budget7. Raw20 increases8.717pp[7.292,10.060] and11.047pp[8.361,13.788], respectively. At16 scenes,7× updates lowers fixed Joint2.138pp and raw20 4.441pp[-6.168,-2.632]. At112 scenes, the Joint change−0.795pp[-2.549,0.905] does not establish an effect, while raw20 decreases2.111pp[-4.084,-0.302]. Thus more updates are not a monotonic improvement here. The112/budget7 vs16/budget1 contrast matches expected query draws and still improves Joint6.442pp[3.810,9.265], fine0.877pp[0.247,1.508] and raw20 6.606pp[4.496,8.607].

All intervals are paired scene bootstrap,10,000 draws, seeds averaged inside each of19 scenes. They are uncorrected; secondary metrics/strata are descriptive. All480 declared method/metric/stratum contrasts, including negative/null outcomes, remain in factorial_paired.json. Original6/added13 results are retained per cell; added13 are historically exposed Val, not independent Test. No highest-scoring cell is silently substituted for the complete design.

## Last-cell outcome and retained failures

The last cell112/budget7 achieves fixed Joint33.607%, fine2.632%, raw20 68.366%, diverse20 89.611%; visual Joint24.726%, acoustic7.840%. Fixed nearest-grid NLL8.889 vs visual9.352, but confidence ECE0.307 vs visual0.230. Lower NLL and higher localization accuracy do not establish calibrated uncertainty. Absolute fine precision remains low; bank oracle fine50.822% is a separate resolution limit. Negative/uncertain within-cell precision and candidate-coverage effects in the16-scene cells remain in the earlier reports, rather than being hidden by this scaling gain.

This is baseline scaling evidence, not a successful multimodal-interaction contribution. The separate frozen/partial/joint and score-intervention studies show that current learned interaction can degrade a strong initial F1. Better standalone evidence with112 scenes motivates testing whether those conclusions transfer, but this study does not answer that question. Neither physical shift robustness nor end-to-end independence follows from this result.

## Frozen settings and full verification

Four cells share all19 Val scenes/64 queries, seeds42/43/44, batch8, AdamW3e−4/weight decay1e−4, CPU-resident scenes, the same full bank/representations and fixed.85/.15 weights/temperature1. Budget1 uses1,500 auxiliary+1,000 dense steps; budget7 uses10,500+7,000. Visual auxiliary and continued dense stages plus acoustic stage yield9 fits per cell. The budget factor changes both auxiliary and dense optimization; it does not isolate dense-only learning.

All36 fits reached scheduled final steps and passed cohort/query/hash checks. Fresh visual/acoustic initial tensors match across all four cells for each seed; all12 visual continuations begin at their own auxiliary checkpoint exactly. These checks permit intentionally different continued visual starts after different auxiliary exposure. All four evaluations share14,592 distinct query/seed/method keys each (58,368 total),7,296 unimodal replay checks/cell (29,184 total), exact checkpoint hashes and empty parameter-selection records. The final comparison refused incomplete evidence in its earlier negative gate probe and now passed with all four cells. No failed main fit or evaluation was discarded.

The earlier CPU/GPU residency smoke's strict bitwise failure and repeat diagnostic remain preserved. They do not invalidate matching initialization within this all-CPU-residency factorial design, nor do they prove long-run numerical equivalence with older GPU-resident pilot models. Expanded feature/cache equality audits and exact-map duplicate audit are linked in README; approximate/source-family/upstream independence remains unresolved.

## Costs and practical limitations

Summed reported fit-training elapsed times across9 stages per cell are about0.069/0.427/0.053/0.304hours for16/budget1,16/budget7,112/budget1,112/budget7. These exclude recorded auxiliary evaluation and final evaluation, initial cache loading, upstream encoder extraction and data generation; they can include sampling-skip/scene-transfer overhead and concurrent GPU workload. They are not dedicated GPU-hour or end-to-end timing claims. Larger cells were executed under different contention, so their lower observed time is not evidence of lower computational complexity. Peak model-process allocated GPU memory is0.737/0.737/1.098/1.099GB, not total device residency or host-memory peak. reported_fit_costs.json records exact aggregates derived from preserved fit summaries. Full deployment amortization and reference-bank cost comparison remain unfinished.

## Reproduction and status

Training source `5cf803681c2d144f3a620bc00c1ab6b841e48e7c`; fit verifier `73f8f18e68a15c51cdc21dc3b744805771476a66`; evaluator `6946a9b60a8135c46ed3d40af7980d437f519894`; cell registrar `d208efaf6d403843783ff1c896988e6a50fdb836`; factorial comparator `518f033995cd40ef353021e893278ab4dea2a9dc`. Final verification/evaluation/registration/comparison executed from unchanged code content at50ec5aa05d5a4739d440b39162ba6aebcfb3ce56. All exited0. Source-code references distinguish training content from later Docs commits. GPU fits/evaluation use avfploc:cu124-clean; CPU registrar and host NumPy bootstrap. External artifacts live under /home/jeongeonlee/track1/experiments/track_b_scale_20261109; weights/raw predictions stay outside Git.

Use the frozen run_scaling_plan command/settings in scaling_protocol.md, verify_scaling_fits without --condition, then independent_fusion --fixed-only --extended-metrics and register_scaling_cell for each cell. The last cell registrar uses --fit-validation <all_cells_validation_v1>. Final command: `python3 -m track_b.scripts.compare_scaling_cells --root <scale-root> --fit-validation <all_cells_validation_v1> --out <fresh-directory>`. Input hashes/registries are in factorial_validation.json and all_cells_validation.json. Never overwrite prior outputs.

This completes study27's planned baseline factorial experiment, not the overall research goal. Scaled M1/OOF integration, stronger deployment/cost evidence, physical-shift feasibility, independent final evaluation and consolidated protocol-level conclusions remain required work. No independent-validation promotion is made.

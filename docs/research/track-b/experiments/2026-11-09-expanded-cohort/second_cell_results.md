# Second scaling cell:16 scenes,7× update budget

2026-11-09. Completed evaluation/registration, Exploratory. n16_budget7 uses the same frozen16 Train scenes, all19 Val scenes,64 queries/scene and seeds42/43/44. Auxiliary10,500 and dense7,000 updates; visual continuation and acoustic setup follow scaling_protocol.md. This is one cell of the frozen factorial design; no partial factorial inference, tuning or epoch selection is performed. Hypothesis for the within-cell comparison is a benefit from fixed independent acoustic/visual fusion; competing explanations include score-field density and modality negative transfer.

| Stratum | Method | Joint % | Fine % | Raw20 % | Diverse20 % |
|---|---|---:|---:|---:|---:|
| all19 | visual | 21.848 | 1.974 | 66.694 | 82.867 |
| all19 | acoustic | 3.893 | 0.082 | 20.038 | 41.996 |
| all19 | fixed | 25.027 | 1.398 | 57.319 | 82.100 |
| all19 | fixed_audio_shuffle | 8.717 | 0.356 | 29.276 | 52.796 |
| original6 | visual | 19.878 | 1.910 | 67.708 | 84.115 |
| original6 | acoustic | 3.472 | 0.174 | 19.358 | 42.101 |
| original6 | fixed | 23.264 | 1.562 | 56.510 | 82.205 |
| original6 | fixed_audio_shuffle | 8.160 | 0.174 | 28.212 | 52.257 |
| added13 | visual | 22.756 | 2.003 | 66.226 | 82.292 |
| added13 | acoustic | 4.087 | 0.040 | 20.353 | 41.947 |
| added13 | fixed | 25.841 | 1.322 | 57.692 | 82.051 |
| added13 | fixed_audio_shuffle | 8.974 | 0.441 | 29.768 | 53.045 |

Across19 scenes, fixed−visual Joint+3.180pp,95% paired scene bootstrap[1.151,5.318]; R@1m+3.509pp[0.850,6.469]. Fine difference−0.576pp[-1.151,0.027] is negative in mean but does not exclude zero. Raw20−9.375pp[-12.171,-6.414] is a clear coverage regression on this development cohort. Diverse20−0.768pp[-2.988,1.343] does not establish coverage improvement. NLL difference−0.271nats[-0.315,-0.223] coexists with lower raw coverage; NLL cannot substitute for coverage or precise pose performance. Confidence ECE fixed0.223 vs visual0.201; this fixed-temperature posterior is not well calibrated. Acoustic-only Joint3.893% and fine0.082% remain weak. Shuffled-audio fixed Joint8.717% provides a query-specific perturbation control; it is not a causal proof of physical geometry reasoning.

These observations support within-cell coarse localization gain but refute an unqualified claim that fusion improves every candidate/precision metric. They do not establish the data-size or update-budget effect: all four verified cells are required by the declared comparison gate. Primary full19, original6/added13 strata, per-seed outcomes and uncertainty are archived in the JSON files; secondary comparisons are descriptive and uncorrected for multiplicity. No method choice or schedule change follows these numbers.

## Verification, provenance and limits

Evaluation and registrar exited0.14,592 unique query/seed/method records,7,296 unimodal replay checks; all model hashes match the six verified final checkpoints.19×64 exact query keys match each method/seed, weights0.85/0.15 and temperature1, empty calibration selection. Initial/continuation checks and9 fit completion records are in second_cell_validation.json. No new model was fitted by registration. Actual fitting code5cf803681c2d144f3a620bc00c1ab6b841e48e7c; evaluator content6946a9b, launched at checkout d7578fb3d3927305f25adb251c9a7941fdc6def5; registrar contentd208efa, executed at175e5d1a6fb89a630619ac1e86bbb3ae424928ca. Full identities remain in provenance.json. Shared checkout source references distinguish training source from Docs commits.

Reproduce the prior cell's fixed-only extended evaluation command with run paths n16_budget7_acoustic_attempt1, n16_budget7_visual_joint_attempt1 and output n16_budget7_evaluation_v1. Register with `python -m track_b.scripts.register_scaling_cell --root <scale-root> --run <scale-root/n16_budget7_evaluation_v1> --pilot-cache <structure/cache> --condition n16_budget7 --fit-validation <scale-root/n16_budget7_validation_v1> --out <fresh-registration>`. GPU evaluator/CPU registrar, avfploc:cu124-clean,2 CPU threads. External results n16_budget7_evaluation_v1 and n16_budget7_registered_v1; raw predictions/weights are not uploaded.

All19 scenes retain historical/research exposure, added13 are not independent Test, upstream image exposure remains uncertain. Exact-map duplicate audit does not resolve approximate/source-family overlap. No Test accessed. Physical shifts, independent evaluation, M1 scaling/backend integration and complete cost accounting remain unfinished. No failed execution in this evaluation/registration; previous study failures remain preserved.

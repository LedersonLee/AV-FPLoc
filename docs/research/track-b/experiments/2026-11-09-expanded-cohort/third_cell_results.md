# Third scaling cell:112 scenes, baseline update budget

2026-11-09. Completed evaluation/registration; Exploratory. n112_budget1 uses112 Train scenes, the same19 Val scenes and64 observations/scene, seeds42/43/44. Auxiliary1,500/dense1,000 updates, fixed visual0.85/acoustic0.15 fusion. No epoch selection/calibration fit. Hypothesis for the within-cell comparison is that independently learned acoustic evidence contributes to visual localization; alternatives include score-field/support effects and evaluation-specific dependence. Data-size/update effects remain reserved for the complete frozen factorial comparison.

| Stratum | Method | Joint % | Fine % | Raw20 % | Diverse20 % |
|---|---|---:|---:|---:|---:|
| all19 | visual | 21.848 | 1.508 | 67.928 | 81.552 |
| all19 | acoustic | 7.045 | 0.219 | 29.523 | 58.114 |
| all19 | fixed | 34.402 | 3.015 | 70.477 | 89.666 |
| all19 | fixed_audio_shuffle | 10.417 | 0.576 | 32.922 | 57.511 |
| original6 | visual | 20.833 | 1.389 | 68.576 | 81.771 |
| original6 | acoustic | 5.642 | 0.347 | 28.038 | 56.337 |
| original6 | fixed | 33.941 | 3.646 | 70.226 | 89.931 |
| original6 | fixed_audio_shuffle | 10.503 | 1.042 | 32.726 | 58.594 |
| added13 | visual | 22.316 | 1.562 | 67.628 | 81.450 |
| added13 | acoustic | 7.692 | 0.160 | 30.208 | 58.934 |
| added13 | fixed | 34.615 | 2.724 | 70.593 | 89.543 |
| added13 | fixed_audio_shuffle | 10.377 | 0.361 | 33.013 | 57.011 |

Across19 scenes, fixed−visual Joint+12.555pp,95% paired scene bootstrap[10.115,15.159]; R@1m+12.829pp[10.197,15.543]; fine+1.508pp[0.959,2.029]; raw20+2.549pp[0.247,4.797]; diverse20+8.114pp[5.702,10.582]. Nearest-grid NLL difference−0.225nats[-0.249,-0.201]. This cell shows a broader within-cell benefit than a Top1-only change, but the uncorrected secondary intervals are descriptive. Absolute fine3.015% is still poor, and the bank oracle fine50.822% imposes a separate resolution limit. Shuffled-audio fixed Joint10.417% vs normal34.402% is a query-specific perturbation control, not proof of a particular physical mechanism.

Prediction-centered support-confidence ECE is0.326 for fixed vs0.205 visual: improved localization and NLL do not establish calibrated uncertainty. The acoustic model alone reaches Joint7.045%, fine0.219%; it remains weak as a standalone localizer. All per-seed/stratum/bin/selective-risk outputs are archived. Original6 and added13 remain descriptive, historically exposed development strata. No independent-Test or M1-interaction-superiority claim is supported. All remaining settings stay frozen; this result is not used to shorten or retune the last cell.

## Verification, reproducibility and limits

Evaluation and registration exited0.14,592 distinct method/seed/query records and7,296 unimodal replay checks passed; exact checkpoint hashes match six verified final models, and all1,216 query keys match each seed/method. Fixed temperature1, weights0.85/0.15, empty selection. Nine fit update/cohort/initial-state checks are in third_cell_validation.json. Candidate generation uses the complete declared bank. No failed invocation or overwritten result in this cell.

Training source5cf803681c2d144f3a620bc00c1ab6b841e48e7c. Evaluator content6946a9b executed at01ad4694114a6973255e19c58bc093b4d8dcd9ea; registrar contentd208efa executed at414bf9172c439c500a8230e4d275b80980db0ec3. Full source identities are in provenance.json. GPU evaluator/CPU registrar, avfploc:cu124-clean,2 threads. Reproduce the fixed-only extended evaluator using n112_budget1_acoustic_attempt1/n112_budget1_visual_joint_attempt1. Register with `python -m track_b.scripts.register_scaling_cell --root <scale-root> --run <scale-root/n112_budget1_evaluation_v1> --pilot-cache <structure/cache> --condition n112_budget1 --fit-validation <scale-root/n112_budget1_validation_v1> --out <fresh-registration>`. Detailed predictions/checkpoints remain external; aggregate/protocol/verification files are copied here.

Reused Val, possible upstream exposure, fixed64-query sampling and unresolved approximate/source-family overlap limit generalization. Exact footprint deduplication does not establish independent evaluation. Test remains unopened. Final112/budget7 cell, across-cell initial-state/protocol checks, all planned factorial contrasts, M1 scaling/backend, physical shifts and complete costs remain unfinished. This cell is progress toward the original protocol, not its completion.

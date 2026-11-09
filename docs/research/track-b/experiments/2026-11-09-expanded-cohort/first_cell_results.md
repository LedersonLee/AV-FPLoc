# First scaling cell: complete development results

2026-11-09. n16_budget1,16 Train scenes,19 Val scenes,64 observations/scene,3 seeds. This is the first of four predeclared cells, not the complete scaling contrast. No result changes the remaining fixed schedule. Hypothesis: independently learned acoustic evidence helps the fixed visual score field on broader development scenes. Competing explanations include chance scene composition, ranking/support-density effects and evaluation reuse. Full19 is primary; original6 and added13 are prespecified descriptive strata. All are Exploratory; added13 are not independent Test.

## Observations

| Stratum | Method | Joint % | Fine % | Raw20 % | Diverse20 % |
|---|---|---:|---:|---:|---:|
| all19 | visual | 20.285 | 1.288 | 65.104 | 81.031 |
| all19 | acoustic | 4.276 | 0.164 | 20.861 | 43.640 |
| all19 | fixed | 27.166 | 1.754 | 61.760 | 84.704 |
| all19 | fixed_audio_shuffle | 9.814 | 0.548 | 32.018 | 56.689 |
| original6 | visual | 17.448 | 0.955 | 64.410 | 82.118 |
| original6 | acoustic | 4.688 | 0.260 | 19.618 | 40.191 |
| original6 | fixed | 25.087 | 2.083 | 61.806 | 84.722 |
| original6 | fixed_audio_shuffle | 8.767 | 0.694 | 30.556 | 56.858 |
| added13 | visual | 21.595 | 1.442 | 65.425 | 80.529 |
| added13 | acoustic | 4.087 | 0.120 | 21.434 | 45.232 |
| added13 | fixed | 28.125 | 1.603 | 61.739 | 84.696 |
| added13 | fixed_audio_shuffle | 10.296 | 0.481 | 32.692 | 56.611 |

Across19 scenes, fixed−visual Joint is+6.880pp,95% paired scene bootstrap[4.907,8.991]; R@1m+8.141pp[6.195,10.334]. Fine+0.466pp[-0.164,1.096] does not establish fine improvement. Raw20 declines−3.344pp[-5.674,-0.959], while diverse20 increases+3.673pp[1.864,5.565]. Nearest-grid NLL decreases0.210nats (difference[-0.241,-0.179]). Fixed−shuffled-audio Joint+17.352pp[14.556,20.614] supports query-specific acoustic contribution under this intervention, not a proven physical mechanism. Original6 fixed−visual Joint+7.639pp[4.948,10.330]; added13+6.530pp[3.926,9.375]. No multiple-comparison correction; secondary metrics/strata are descriptive.

Fixed fusion translation median3.458m/p90 11.209m vs visual4.606/12.415m. Yaw median75° and p90 170° remain poor. Prediction-centered support-confidence ECE is0.253 for fixed vs0.189 visual: better Joint/NLL does not imply better calibration. Acoustic ECE0.007 coexists with only4.276% Joint; ECE alone is not useful-localization evidence. Full bin tables and selective-risk curves are in first_cell_stratified_results.json. No calibration fit or confidence threshold selection was performed.

## Interpretation, counterevidence and limits

The evidence supports a baseline fusion benefit on this reused development cohort, but neither new interaction superiority nor scaling benefit. Raw20 loss alongside diverse20 gain is consistent with a changed distribution of nearby high-scoring cells; it does not prove that explanation or justify reporting only diverse coverage. Fine precision and uncertainty remain failure points. Joint position≤1m/yaw≤30°, fine position≤0.1m; same bank for every method. This cell is retrained with CPU scene residency; its original6 outcome differs from older GPU-resident pilot fits and is not a bitwise reproduction of those learned models. Previously documented short-run floating-point differences do not prove the cause of the long-run difference. All four new cells use the same residency; compare within the frozen factorial study.

No Test accessed. Upstream/historical Val exposure, repeated diagnostics, six/19-scene selection history and fixed64-query sampling restrict generalization. Independent final evaluation, physical shifts, M1 joint architecture, OOF discrimination integration, complete costs and other cells remain pending. This result alone does not complete the requested research.

## Verification and reproducibility

14,592 distinct records (19×64×3×4 methods),7,296 unimodal replay checks, fixed weights0.85/0.15 and temperature1, empty calibration selection. Model hashes match the six verified final checkpoints. Scene/seed/query counts and paired method keys were checked. Fit provenance and initial weights were checked separately; first_cell_validation.md and first_cell_validation.json remain the fit evidence. Registrar source `d208efaf6d403843783ff1c896988e6a50fdb836`, exit0; evaluator source6946a9b (full SHA in provenance). Run `python -m track_b.scripts.register_scaling_cell --root <scale-root> --run <scale-root/n16_budget1_evaluation_v1> --pilot-cache <structure/cache> --condition n16_budget1 --fit-validation <scale-root/n16_budget1_validation_v1> --out <fresh-directory>`. Docker avfploc:cu124-clean, CPU registrar; GPU evaluator. Original fit code5cf803681c2d144f3a620bc00c1ab6b841e48e7c.

Observation session33530 returned143 while a container remained briefly active. We inspected live processes/container events and completed output artifacts instead of restarting. The observed container60ff0c12f7f7 subsequently emitted exit0; its event is preserved externally as n16_budget1_evaluation_container_exit.jsonl. Complete summary/prediction artifacts then passed the independent registrar. No partial output was treated as a result and no duplicate evaluation was launched. Detailed predictions/weights remain outside Git in the scale root; only aggregates, protocols and hashes are archived.

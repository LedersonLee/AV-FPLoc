# Requirement audit — 2026-11-17

Original objective: AVLoc independent reimplementation and matched Track A benchmark, user attachment pasted-text-1.txt. This audit does not redefine completion around finished tasks. Status: **all requested seed42 benchmark requirements verified, 2026-11-17**.

| Requirement | Evidence inspected | Status / limitation |
|---|---|---|
| Method3/AppB,C,G and reviews | IMPLEMENTATION_AUDIT.md, implementation_assumptions.md | Documented original paper audit and uncertainty; initial pending rows now reconciled |
| Architecture, losses, uncertainty routing and fusion | METHOD_AT_A_GLANCE.md, architecture/initialization/audio tests; A3_router_diagnostics.json | Implemented and trained; saved fusion replay complete |
| Official-code availability investigation | IMPLEMENTATION_AUDIT.md | Investigation recorded; availability not positively verified, independent implementation explicit |
| Full eligible assets and exclusions | asset_audit_summary.json, assumptions |112Train/58,894 and19Val/9,965; additional Val-family exclusion documented |
| Six-channel echo adaptation/direct path/STFT | assumptions, audio_test.json | Explicitly differs from paper binaural sensor; original input contract retained |
| A0/A1/A2 sufficient training, gate before A3 | completed_training_cost_all_seed42.json, EXPERT_VALIDATION_GATE.md | All naturally complete;87A3epochs, best67 |
| Same cohort and20cm/10° primary grid | REQUESTED_METRIC_TABLE.md, expert_gate_seed42.json, disabled_integrity_seed42.json, A3_integrity.json | Full native aggregate validation; B2/GEOQ seeds averaged inside scenes |
| Separate10cm evaluation | REQUESTED_METRIC_TABLE.md | Complete for all five ray arms; native10cmB2 absent and never represented by interpolation |
| Position/yaw/ray/FoV/coverage/scenewise/repair-harm | metrics, ray analyses, A3_initial_contrasts.json | Primary metrics complete; both final uniform paired comparisons complete: A3_final_paired20/10.json |
| Independent fusion and disabled interaction | FUSION_DECOMPOSITION.md, INTERACTION_CONTROL_RESULTS.md | Complete; matched-parameter control changes multiple factors, not pure attention causality |
| Fixed versus learned routing | A3_router_diagnostics.json and running uniform evaluation | Ray and pose controls complete at both grids (A3_UNIFORM_CONTROL.md), with all-row aggregate checks |
| Sharpness, entropy, GT mass and peak margins | matching.py saved row fields; matched score maps | PEAK_MARGIN_ANALYSIS.md now covers all9methods/9965queries; native units differ by method |
| Temperature/NLL/ECE and reliability | all-arm calibration verification files | Complete; A2 negative10cm result preserved, A3 residual error documented |
| Acoustic observability, direction/distance/error conditions | ACOUSTIC_SUCCESS_CONDITIONS.md, a1_ray_diagnostics_20cm.json | Existing full-cohort diagnostic evidence; geometry oracle not an acoustic information bound |
| Router weight/error and scale/error | A3_router_diagnostics.json | Full36directions/9965queries; descriptive, not independent reliability proof |
| Posterior figures and failure cases | MATCHED_SCORE_FIELDS.md, A3_failure_case349.png | All8 A3-expanded figures reviewed;36field replays verified and archived in matched_fields_with_a3 |
| Cost implications | COMPLETED_RUNTIME_RESULTS.md, bank_cost_audit.json | Storage and scoped warm stages measured; historical rendering/end-to-end latency unavailable, no speedup claim |
| Phase5 five research decisions | RESEARCH_REPORT.md | INTERPRETATION_AND_DECISIONS.md answers all5questions from primary evidence; both grids reconciled with final controls |
| Reproducibility, preserved failures and provenance | per-analysis receipts, codeSHA, recovery hashes | Preserved; raw artifacts local, external recovery not clean-clone reproducible by itself |
| Git index and named deliverables | RESEARCH_INDEX.md, METHOD_AT_A_GLANCE.md, IMPLEMENTATION_AUDIT.md, RESEARCH_REPORT.md | Present; all final results reconciled; publication verified after commit in final response |

Seed43 remains deferred under the user's seed42-first decision; scene bootstrap does not establish seed robustness. Test17 remains unused. No required seed42 analysis remains. Deferred seed43, independent Test and hypothetical further causal controls are not claimed complete or silently substituted; they are outside the current user-prioritized completion scope.


Final evidence review: model.py confirms ResNet50/18, full-resolution tokens,2intra layers,11/36ray heads,4-layer joint expert, uncertainty-detached per-ray routing and exact moment matching; train.py confirms AdamW/cosine/max100/patience20/full-query clean-Val selection. Existing architecture/audio/optimizer-resume evidence was inspected together with actual completed runs. A1 diagnostics contain per-direction/distance-bin/FoV/error-calibration and scene statistics. All-arm final validator rechecks checkpoint/input/prediction hashes and every native aggregate for5arms×2grids×9965queries; uniform controls separately reaggregate both grids. Calibration reviewers replay all curves and crossfit results.36figure-field checks plus visual review cover eight final plots. Both finalization processes completed; no live research computation is required for the delivered findings.

Completion means the specified benchmark and analysis were performed, not that every hypothesis succeeded or every uncertainty vanished. Official-code availability could not be positively verified; the original search/access limits are recorded. Scientific limitations remain in the final conclusions, especially reused development, one AVLoc training seed, sensor adaptation, fixed-router inference-only control and incomplete historical compute costs.

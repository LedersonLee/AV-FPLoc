# Requirement audit — 2026-11-17

Original objective: AVLoc independent reimplementation and matched Track A benchmark, user attachment pasted-text-1.txt. This audit does not redefine completion around finished tasks. Status: **not complete**.

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
| Position/yaw/ray/FoV/coverage/scenewise/repair-harm | metrics, ray analyses, A3_initial_contrasts.json | Primary metrics complete; final uniform paired comparisons pending |
| Independent fusion and disabled interaction | FUSION_DECOMPOSITION.md, INTERACTION_CONTROL_RESULTS.md | Complete; matched-parameter control changes multiple factors, not pure attention causality |
| Fixed versus learned routing | A3_router_diagnostics.json and running uniform evaluation | Ray-level comparison complete; pose-level20/10cm controls pending |
| Sharpness, entropy, GT mass and peak margins | matching.py saved row fields; matched score maps | PEAK_MARGIN_ANALYSIS.md now covers all9methods/9965queries; native units differ by method |
| Temperature/NLL/ECE and reliability | all-arm calibration verification files | Complete; A2 negative10cm result preserved, A3 residual error documented |
| Acoustic observability, direction/distance/error conditions | ACOUSTIC_SUCCESS_CONDITIONS.md, a1_ray_diagnostics_20cm.json | Existing full-cohort diagnostic evidence; geometry oracle not an acoustic information bound |
| Router weight/error and scale/error | A3_router_diagnostics.json | Full36directions/9965queries; descriptive, not independent reliability proof |
| Posterior figures and failure cases | MATCHED_SCORE_FIELDS.md, A3_failure_case349.png | All8 A3-expanded figures reviewed;36field replays verified and archived in matched_fields_with_a3 |
| Cost implications | COMPLETED_RUNTIME_RESULTS.md, bank_cost_audit.json | Storage and scoped warm stages measured; historical rendering/end-to-end latency unavailable, no speedup claim |
| Phase5 five research decisions | RESEARCH_REPORT.md | Needs final synthesis including completed uniform controls and cost limitations |
| Reproducibility, preserved failures and provenance | per-analysis receipts, codeSHA, recovery hashes | Preserved; raw artifacts local, external recovery not clean-clone reproducible by itself |
| Git index and named deliverables | RESEARCH_INDEX.md, METHOD_AT_A_GLANCE.md, IMPLEMENTATION_AUDIT.md, RESEARCH_REPORT.md | Present; final synchronization pending final results |

Seed43 remains deferred under the user's seed42-first decision; scene bootstrap does not establish seed robustness. Test17 remains unused. Completing the remaining analyses is required before marking the active goal achieved.

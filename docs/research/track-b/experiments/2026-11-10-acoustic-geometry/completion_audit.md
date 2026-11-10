# Requirement-by-requirement completion audit

2026-11-10 final audit. **Completed:** official17-scene observation generation and frozen evaluation finished. Evidence below was inspected against current files and saved predictions. Negative hypotheses are retained as scientific outcomes, not recast as positive findings.

| Requested requirement | Current authoritative evidence | Status/limits |
|---|---|---|
| VisualEchoes network, preprocessing, supervision, loss | Main README links original paper, supplement, official repository; adapted CNN dimensions and loss recorded | Audited paper/supplement; exact original training code not located, adaptation not exact reproduction |
| Existing Track B and geometry fusion audit | README references existing `structure.py`, studies17/19/27/30, legacy matching maintained in `transfer.py` | Completed; no transplanted incompatible cohort scores |
| Six-channel data, source/receiver/microphone geometry | Input protocol, hashes and stage identity audit;48kHz6mic5cm array, source centre | Completed for used observations; body alignment requires recorded orientation metadata |
| Furnished/furniture-free pairing | `pair_summary.json`, opening-preserving native render controls, helper hashes | Completed development controls; all-object removal and opening-preserved removal have distinct meanings; neither proves all removed objects are movable furniture |
| GT floorplan/pose alignment and masks | `target_protocol.json`, stage identity audit, inspected QA overlays, target tests | Native vertical-stage hit at receiver height; max20m; uncertain canonical-boundary disagreements masked; no acoustic-path/visible-surface equivalence |
| Scene-disjoint train/val/test | Frozen95/17/19 split plus17 official Test family audit; `method_freeze.json` | Family exclusion verified; official observations generated and evaluated; previous exposure prevents blanket pristine claim |
| Echo-style and spatial models,36 directions | `models.py`, experiment registry,18 completed fits | Completed. Ordered spectral/time tokens and channel phase retained; individual early/late physical reflections are not separately identified |
| Single-depth and probabilistic heads | Shared encoders,80-bin0.25m categorical distributions, regression, matched seeds/updates | Completed. Distributions can represent multiple distance modes; no claim that learned modes recover distinct physical paths |
| Masked geometric learning | Target-v3 override/hash checks and masked losses in trainer | Completed. Optional paired consistency not used: trustworthy pairs were generated only after fits; no new tuning after freeze |
| Optional GCC-PHAT/alternative architectures | Phase-preserving CNN/query-attention implementation and phase ablation | Considered optional; GCC-PHAT and extra Transformer variants not needed to claim completion of the specified initial comparison |
| Representation MAE/RMSE/correlation/NLL/calibration | Primary/phase/development holdout summaries, CRPS,90% coverage/width | Completed development and official evaluation. Across-query per-direction and within-query angular correlations explicitly distinguished |
| Same-scene other-pose negative control | Donors constrained>0.5m with closest heading; saved normal/shuffle predictions | Completed; scalarization and180° output rotation additionally refute directional-shape success |
| Structure types and furniture robustness | `strata_summary.json`, paired reports and backend check | Completed development; associations not causal room effects, repeated renders not deterministic |
| Frozen/fine-tuned/joint transfer versus original branch |21 completed fits, same full candidate matcher and frozen shared visual branch per seed, transfer summary | Completed development and official evaluation. Same-local and same-total-update scratch controls included |
| R@1m/joint1m30°/fine0.1m/NLL/ambiguity | Full transfer and heldout summaries including entropy, rawTop20 and oracle coverage | Completed development and official evaluation |
| Latency and model size | `transfer_runtime.json`, representation registry | Raw-feature/full-field/Top20 replay verified; includes upstream backbone/STFT, excludes I/O/model load/map setup; shared GPU, two probe scenes |
| RGB/GT/prediction and probability heatmaps | `predefined.png`, `failure.png`, selection metadata in external figures output | Completed, inspected scientific figures; worst case explicitly selected by error |
| Failure cases and negative results | Results/followup reports; failed target/smoke/parity outputs preserved | Completed for finished runs; all encountered execution failures are retained |
| Geometry-conditioned matching necessity | Followup interpretation | Evaluated as unproven next direction. Failed transfer does not establish that cross-modal matching is necessary or sufficient |
| Reproducibility and code provenance | Registries, code SHAs, configs/seeds, hashes, reproduction commands, frozen checkpoint hashes | Completed for finished runs; large data and weights external, no claim of standalone clean-clone data reproduction |
| Final docs/index/non-main push | Canonical index and branch `refactor/hssd-two-branch-fusion` | Periodically synchronized; official report and index updated; final commit/push state is reported outside the commit |

## Baseline applicability

This study tests geometry representation and pretraining within an unchanged Track B matcher. Its direct causal controls are the original acoustic branch, same-capacity scratch and total-update-matched scratch, shared visual scorer, train-only geometry prior and same-scene donor controls. B2/B2+GEOQ use a different reference/proposal pipeline; existing results are retained in their studies and are not reused as same-cohort numbers here. No superiority over those pipelines is claimed. A later system-level architecture benchmark would need fresh matched inputs/candidate and reference-cost accounting; this study does not convert the pretraining experiment into that broader comparison.

## Scientific closure evidence

- [Official input verification](official_input_verification.json):2,193 waveform hashes,17 expected scenes and64 furnished+64 stage-only+1 repeat record per scene verified.
- [Official output verification](official_verification.json):39 frozen checkpoint hashes,17 cache hashes,1,224 representation scene/condition records and195,840 query localization metric records checked; shared visual predictions identical across arms and acoustic conditions; summary rows agree with saved predictions/metrics.
- [Official tables](official_results.md), [all metrics](official_summary.json), [paired representation contrasts](official_contrasts.json), [scientific conclusion](conclusion.md). Three seeds are averaged within scene before17-family bootstrap, not counted as independent scenes. No tuning after freeze.
- Development output verification separately covers19 scenes,684 representation records and109,440 query metric records. Seven targeted unit checks pass. Raw inference replay covers two declared development probes, not every official query.
- RGB/GT/prediction/probability figures were inspected; selected and worst-error examples remain labeled as development examples. Local Markdown links, JSON, archive hashes and staged file scope were checked before synchronization. Data, raw predictions, logs and checkpoints remain outside Git.

The completed scope is an evaluated initial representation/pretraining study. It does not prove all possible acoustic encoders fail, all geometry is identifiable, furniture invariance, or superiority to the different B2/GEOQ pipeline. Additional model searching after these Test results would be a new development cycle requiring new evaluation safeguards, not a condition for closing this initial study.

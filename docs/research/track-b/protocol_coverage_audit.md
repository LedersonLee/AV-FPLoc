# Track B protocol coverage audit — 2026-11-09

Status: **overall objective incomplete**. This is a work-scope audit, not a new experiment or a completion declaration. Source: the supplied v1 protocol preserved in [study15](experiments/2026-11-09-dense-se2-initial/supplied_protocol.md). Example CLI paths in section10 are suggested interfaces, not literal required commands; actual interfaces live in `track_b/scripts/`. Optional/recommended alternatives remain distinct from mandatory gates. Existing observed findings do not imply independent validation.

## Gates and required comparison coverage

| Protocol item | Current authoritative evidence | Remaining work / limit |
|---|---|---|
| Stage0 workspace and routing | study15 audit_report, manifests and bank checks; repository AGENTS and canonical index | Later source additions must retain original exclusions and no Test access |
| Scene-held-out split and upstream exposure | 16 Train/6 Val frozen cohort; all1408 source hashes/alignment checked in study17; larger metadata audit; study27 exact transformed-footprint audit131 maps,0 duplicates | Reused Val is not independent; near-duplicate geometry beyond known family IDs and upstream foundation exposure are not fully excluded |
| Bank/frame/rig QA | geometry/ray/dense tests and study15/17 actual cache audit | Numerical oracle Joint100% is not proof of fine accuracy or strict point-in-footprint validity for every future dataset |
| V0 map-only/random prior | Study27 exact uniform prior on19Val/1,216queries: Joint0.4313%; first-cell acoustic−prior+3.8451pp[3.1780,4.5852] | Uniform null only; learned map-only/query-distribution priors and final selected model comparison remain |
| V1/A1 full-bank scoring | study17 scratch visual/acoustic, study19 foundation visual | Full eligible training scale and final independent evaluation unfinished |
| F1 strong independent/calibrated fusion | study20: normal/brightness; independent modality FP maps | Occlusion curve complete: severe negative transfer and NLL calibration loss documented; matched-model full occlusion curve complete in study21; other shifts remain |
| F2/F3/M1 matched comparison | study18 nine scratch fits negative; study21 strong foundation nine fits/uncertainty/occlusion complete, all below initial F1 | No attention superiority established; study24 regimes and study28 score controls complete. Study29 expands matched M1/F2/F3 to112Train/19Val: nine fits running, main attention42 initial field replay1216 queries passes; all-fit conclusions pending |
| T0/T1/T2 frozen/partial/joint | study17 and19 common-pretrain controls | Study24 all27 fits complete; initial/frozen tensors verified, frozen/partial remain below initial F1. Encoder drift alone is insufficient; study28 score-scale versus gate/residual intervention now complete, both contribute to regression; selected-regime shifts remain |
| D0/D1/D2 backend | study23 independent-F1 OOF generation, support mass and12 new classifiers complete | Geometry input adds no benefit over score statistics; Cached-feature head audit now replays4,608 inference metrics without label/ID inputs and Python file opens; M1 backend, shifted evaluation and full candidate-builder/raw-input audit remain; no GT injection |
| X1 acoustic evidence | Shuffle,zero,direct-only/direct-masked/late-only controls in study17; strong F1 shuffle study20 | Raw-input controls must also be checked on selected final pipeline, with changed-energy/distribution confounds stated |
| X2 geometry removal/orientation | Earlier geometry/frame diagnostics and tests | Study25 all joint checkpoints: candidate-map removal,90/180/random yaw controls complete; near-uniform removal and exact position-metric invariance under yaw permutation recorded. These destructive tests do not isolate learned FP from raw geometry or prove a new mechanism. Final selected regime still pending |
| X3 capacity/compute | Same parameter count/update budget for attention/concat/separable | Actual FLOP/wall-time equality not proven; F1 has fewer head parameters and fewer updates |
| X5 grid/yaw/valid-mask sensitivity | .25m/36 bank audit | Strong independent F1 fixed-checkpoint .20/.50m,12/72yaw and .1m erosion complete in study22; same six variants also complete for all nine joint foundation checkpoints,20,736cases with exact baseline metrics within tolerance; selected future regimes still need validation |
| M2/P1/X4 alternatives | Mechanism diagnostics guided structural and foundation changes | Recommended bottleneck, echo bias, waveform/spec/spatial representation comparisons not yet executed; do not claim them as mandatory successes |
| B2ref and external visual refs | Isolated historical UnLoc replay; B2 deployment storage metadata study18 | Same-grid fair B2/GEOQ accuracy/cost and primary-source external reference boundary review recorded in external_reference_review.md; full-paper/reproduction feasibility still incomplete; historical results not substituted |
| Physical/material/furniture shifts | Selected1408 samples audited furnished_native_scene; materials/transmission flags reported | Compatible paired observations unavailable in current cohort; generation or verified alternative data still needed; Test clutter cannot be borrowed for selection |
| Cross-dataset | No verified physically aligned second dataset | Conditional on valid data/calibration; no unsupported cross-dataset claim |
| Full scale and independent final evaluation | Pilot/development results only | Study27 preflight freezes112Train/19Val×64queries and structural cache verification is complete with exact22-scene pilot overlap; upstream features completed/verified across131 scenes with exact22-scene pilot overlap; fixed2×2 baseline scaling completed36 fits/58,368 evaluation records with matched initial states and inputs; scene expansion improves F1 at fixed updates, updates alone do not. Scaled M1 is running in study29; method freeze and appropriately independent evaluation remain; do not unlock Test during design |

## Metrics, costs and artifacts

- Available: Joint≤1m/30°, R@1m, fine≤.1m, raw20/diverse20, nearest-grid NLL, entropy, scene bootstrap and three-seed baselines. These do not substitute for the other required metrics.
- Implemented and measured on strong independent F1 in study22; matched final models now measured across full occlusion curve: raw5/10, strict(.5m,15°)/(.2m,10°), continuous translation/yaw median/p90, nearest-grid rank/ties, normalized entropy, prediction-centered posterior support confidence, ECE and selective risk. Source `track_b/eval/extended_metrics.py`; 37 current tests passed including wrap, GT-independent confidence, singleton/tied-rank and calibration-bin edges. Its use completed in the full matched occlusion evaluation,13,824 cases with zero normal replay mismatches. No numeric results inferred from tests.
- Reliability confidence is posterior mass in a region centered at the model's MAP pose; GT appears only in scoring correctness. This region probability is a discretized confidence diagnostic, not a certified continuous-pose probability. Rank uses nearest valid grid target and reports score ties; nearest-grid mapping must not hide out-of-bank labels.
- Costs: weak-model cached scoring/map setup and observed existing reference storage audited in study18. Foundation extraction cache costs recorded in study19/20. Raw RIR read/preprocessing, upstream RGB encoding, full scoring, candidate extraction/discrimination, cold/warm latency, full training GPUh and same-bank B2 deployment amortization remain unassembled. No end-to-end Pareto claim is supported.
- Results registries and reproduction/provenance live per study and are linked by the single RESEARCH_INDEX.md. Do not create a second experiment index. Final consolidated results table with all mandatory columns, final_report and hypothesis falsification report are not complete; unavailable entries must be NA.
- Qualitative score-field cases exist for the older weak models (study18). Final strong models require consistent-rule cases, ambiguity/support analysis and any discriminator final pose. Existing attention heatmaps cannot establish causal mechanisms.

## Minimum test contract audit

| Section11 item | Evidence and scope |
|---|---|
| 1 full bank/index | Dense/structure tests plus actual bank profiling; current model scores exhaustive supplied bank |
| 2 GT leakage | Existing inference signature and old runtime GT-key deletion check; one-query/full-bank foundation checkpoint audit passed; study26 adds six-scene raw-observation replay; offline schedule-based channel alignment is explicitly outside GT-free inference; broader physical calibration coverage remains limited |
| 3 split integrity | Metadata family exclusion and cohort provenance; broader geometry near-duplicate/upstream exclusion unresolved |
| 4 frame equivariance | Existing structural/interaction rigid transform tests; final dual-tower path explicitly verified for all three modes in35-test run |
| 5 mic order/time origin | study15/17 calibration/raw-RIR audits and geometry tests; preserve same sensor contract |
| 6 branch gradients | Foundation interaction tests and actual smoke/main branch gradient logs; initial zero new-head internal gradients intentionally activate later |
| 7 chunk equivalence | Existing full/chunk and cached/uncached score tests; actual initial F1 replay1152cases zero mismatches |
| 8 extraction | Existing diverse/support tests; independent-F1 D2 now uses Train-only two-fold scorers, hashed candidates and GT-free builder; M1 proposals remain |
| 9 no RIR reference I/O | Old interaction runtime I/O audit; foundation scorer trace passed on one full-bank query; study26 replays raw decoded RGB/body-frame RIR through upstream/features/full-field in six first-query probes with bitwise equality; Python-only hooks do not capture native I/O |
| 10 shuffle | Actual independent-fusion accuracy control plus model perturbation tests; changing scores alone not sufficient |
| 11 new scene | Existing varied-map forward tests; explicit final-model new-map-size/orientation test now passes for all three modes |
| 12 costs | Old cached scorer benchmark; complete final end-to-end benchmark pending |

No completion gate is inferred from the aggregate test count. Nine-fit comparison, full occlusion and independent-F1 OOF discrimination are now complete with mixed/negative findings. Next dependencies include testing the training-objective explanation after score-scale and learned-interaction interventions both recovered accuracy; scale, M1 backend provenance, physical-shift feasibility and complete costs remain within the original goal.

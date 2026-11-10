# Learned Acoustic Structural Geometry — Study 31

2026-11-10. Status:18 representation fits completed; directional-geometry hypothesis not supported in the initial models. Full research objective active; transfer and physical paired robustness remain. Primary Track B. Evidence: Exploratory until independent evaluation is justified.

## Question and competing explanations

Can six-channel furnished RIR predict local structural geometry beyond a room-scale prior, and does that representation transfer to unchanged Track B localization? A room-size/material prior, memorized scene identity, or supervised target artefacts can produce low depth error without useful pose evidence. Same-scene **different-position** donors, a train-only directional prior, phase removal, and separate transfer comparisons distinguish these explanations. A depth gain alone is not localization success.

## Reference and repository audit

[VisualEchoes paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123540630.pdf): binaural responses to a 3 ms chirp; first 60 ms, 44.1 kHz, Hann length64/hop16/FFT512. The target in the case study is visible image depth, not a furniture-free 360° wall profile.
[Supplement](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123540630-supp.pdf): Echo-Net has three convolutions (8/4/3), BatchNorm/ReLU, and a512-dimensional linear feature. Echo2Depth reshapes this to4×4×32 and uses seven up-convolutions to128×128 depth; loss is mean log(1+absolute error). Case-study training uses Adam, lr1e-4, weight decay5e-4, batch128. The [official repository](https://github.com/facebookresearch/VisualEchoes) distributes data/echo generation; an exact original training implementation has not been located. Unspecified widths/strides will be declared adaptation choices, not claimed reproductions.

Existing `track_b/models/structure.py` already encodes6 log magnitudes plus12 adjacent-channel phase features, with3 CNN layers and a16-ray regression head. Its supervision uses binary footprint first-blocked rays, explicitly not proven wall hits. Studies17/19/27 and30 cover structure-assisted localization and scale uncertainty; they do not establish the requested new geometry pretraining or categorical multimodal depth head. Preserve these baselines and their negative results.

## Data and GT contract

Existing positive-pair observations:48kHz,6 omni microphones,5cm radius, source at array centre; native furnished scenes, materials/transmission disabled. World yaw increases +X toward +Z. Offline body alignment uses the existing schedule's exact cyclic channel permutation. This preprocessing uses pose metadata and must not be described as inferred orientation. At inference, the observation is already body-aligned.

Canonical map source is stage slicing but adds a semantic exterior closure. The map's blocked pixels and `desdf_occluder_mask` therefore are not uniformly direct physical-wall evidence. New targets intersect vertical architectural stage faces (absolute Y-normal≤.15) at receiver height, native X/Z with36 body bearings. No-hit and >20m rays are masked, never assigned a physical20m wall. Fixed built-ins may be part of the stage: this is a vertical architectural-surface proxy, not a semantic wall classifier. Geometric fixtures verify heading, translation, open rays and range censoring. Three predeclared training-scene overlays were inspected. One exposed external structure beyond an opening: v3 conservatively masks stage hits >0.20m beyond the first canonical blocked boundary. The map never supplies the supervised distance. This excludes uncertain floorplan/stage disagreements as well as exterior hits; evaluation is conditional on the remaining valid directions. Mean train coverage is79.96%, not full360° accuracy.

Existing FP/Observed positive pairs change native furnished geometry to canonical map extrusion (including assumed ceiling and exterior closure). They are **not** a furniture-only paired intervention. Stage-only pairs or new matched rendering remain to be audited; no consistency loss is enabled on these extrusion pairs. `scenes-uncluttered` is not accepted as furniture-free.

## Split and evaluation plan

Repartition the112 non-Test training scenes by fixed salted SHA256 **source-family** ordering:95 train /17 inner validation (16 held-out source families). Retain19 historical validation scenes as final **development test** after method freeze, with their previous exposure explicitly retained. No official Test observation has been opened. Same-family split exclusion is asserted. This supplies scene-disjoint development, not independent generalization; genuinely untouched official Test coverage must be audited separately before a final independent claim.64 uniformly scheduled queries per scene; source-family bootstrap after averaging seeds and same-family variants (inner validation17 scenes /16 families; final development19/19), seeds42/43/44. Query headings are not independent scenes.

Phase A: Echo-Net-style6-channel CNN and spatial encoder, regression and categorical depth heads; fixed update budgets and parameter/time reporting. Phase B: different-position same-scene shuffle, train-prior baseline, phase ablation and furniture pairs where genuinely available. Phase C: existing acoustic branch, same-capacity scratch, geometry-frozen, geometry-fine-tuned, joint auxiliary training in the unchanged scoring/candidate pipeline. Report MAE/RMSE/correlation/NLL/calibration, R@1m/joint1m30°/fine0.1m/posterior ambiguity, costs and uncertainty. Existing strong visual/fixed fusion is mandatory; incompatible cohort numbers are not transplanted.

## Reproducibility and failures

Code: `track_b/acoustic_geometry/`; local output `/home/jeongeonlee/track1/experiments/acoustic_geometry_20261110`. Stage preparation uses installed `echoloc/floorplan-geometry:py311` (trimesh5.1.1); training environment planned `avfploc:cu124-clean`. Existing two GPU server processes are preserved. Raw data/weights/logs stay outside Git. Actual source SHA is recorded by each execution, then in the canonical index.

Environment probes found host Python has no torch/trimesh and training container lacks trimesh; use separate installed geometry/training images. First fixture test collection from parent directory failed because PYTHONPATH was absent; rerun with repository PYTHONPATH. These are setup failures, not experimental model outcomes. Current model results and directional counterevidence are in [results](results.md). Furniture invariance and transfer gains remain unverified. Overall objective remains active.

## Preflight update

131 stage config/template/identity-transform checks pass; official Test source-family overlap0 using IDs only. Targets contain8,384 poses across95/17/19 scenes. Four unit checks pass. Four GPU2-step arms execute backward and both normal/different-position evaluations. Smoke numbers are not research results. See [preflight](preflight.json), [split/target protocol](target_protocol.json), [stage identities](stage_identity_audit.json), [GPU smoke](smoke_validation.json).

Echo adaptation:6→32/64/64 CNN with kernels8/4/3,512 latent,32×9 seed and two1D up-convolutions to36 directions. Spatial:18-channel CNN, ordered4×8 frequency/time tokens, sinusoidal-bearing query cross-attention; this distinguishes early/late token positions but does not identify individual physical echoes. Categorical heads have80 bins over0–20m and two-bin soft-label cross entropy; regression uses masked log1p absolute error. They share exact encoder initialization per seed; parameters501,089/503,696 for Echo,101,345/111,536 for Spatial. Architecture comparisons differ in frontend/time window and capacity; no attribution to phase alone without the phase-zero control.

Completed primary run:12 fits (2 encoders×2 heads×seeds42/43/44),1,500 fixed Adam updates, batch16 with scene-balanced sampling, lr1e-4, weight decay5e-4; final step only. Input prep hashes every observed waveform and sidecar. Cached v1 features contain historical v2 labels, which the trainer **always replaces** using hash-verified v3 targets and exact pose/query-ID checks. Historical19-scene development evaluation stays unopened until method freeze.

Failure history is preserved in `<study-root>/targets_v1.log`, `targets_v2`, `qa_v1`, `smoke_v1.log`. Source-family leakage guard stopped the first split before learning. No performance-based tuning has occurred. Missing host packages were handled using installed containers without modifying shared environments.

## Current outcome

[Results and counterevidence](results.md), [full primary metrics](representation_summary.json), [phase-zero control](phase_zero_summary.json), [directional diagnostic](directional_diagnostic.json), [run registry](experiment_registry.json), [reproduction](reproduction.md). Same-scene donor sensitivity is present, but rotating predicted directions180° barely changes MAE and within-query angular correlation is near0. The initial models mainly capture range-scale information; successful directional structural representation is not established.

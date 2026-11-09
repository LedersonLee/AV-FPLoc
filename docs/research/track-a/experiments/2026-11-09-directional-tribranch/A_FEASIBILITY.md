# Stage A decision receipt (before Stage B)

**MIXED.** On12 outer-held scenes×128query, two seeds averaged, acoustic-only embedding ridge obtains near/far accuracy77.67% versus fit-scene prior62.00%. Combined raw+embedding ridge77.07%, donor55.26%; raw ridge75.57%; same-input MLP71.16%. Coarse targets are structural-footprint boundary extents capped10m, not semantic wall identity or metric acoustic depth.

Combined ridge favors the successful candidate in60.47% of visually ambiguous good/bad pairs, donor50.88%, prior50.19%. Pair multiplicity does not increase independent scene count. The fit-scene threshold uses visual score gaps only; GT merely labels retrospective good/bad pairs. Conditional tree joint: same-budget base74.71%, cue74.48%, donor74.54%, nuisance74.87%. Thus target prediction is NOT established downstream gain. Scene bootstrap and stronger originalGEOQ comparison are required in final results.

Proceed to mandatory Stage B. Include one separately labeled F0_gate_cue arm because coarse-label and matched/donor pair relevance are positive; do not treat it as a successful localization method. Established acoustic query/reference embeddings remain the main Stage B input. Raw-feature ridge success suggests the signal is not exclusively created by the historical learned embedding. Roll corruption is sensitivity evidence; rotated-target and repeated-render checks follow separately.

New12 scenes are excluded from originalB2/GEOQ/acoustic training, but teacher-exposed and previously researched. Each new probe excludes its outerheld4 scenes; conditionalranker training cues are2inner-scene crossfit. Models have fixed hyperparameters, no outer-label epoch/threshold selection.

## Final evidence

All intervals below use20,000 paired scene bootstrap draws;2seeds average within each of12scenes. Ridge duplicates across seeds do not provide extra independent data.

|Acoustic-only probe|Near/far accuracy % [95%CI]|Ordinal agreement %|Matched−donor accuracy pp [CI]|
|---|---|---:|---|
|raw_ridge|75.570 [73.633, 77.734]|76.124|19.303 [15.299, 23.307]|
|embedding_ridge|77.669 [75.765, 79.443]|78.144|22.461 [17.741, 26.823]|
|ridge|77.067 [74.919, 79.134]|75.794|21.810 [17.350, 25.944]|
|mlp|71.159 [69.043, 73.079]|70.811|15.365 [12.061, 18.522]|

Embedding ridge sector accuracy front/left/back/right: 77.148, 76.693, 78.841, 77.995%; near-label balance: 61.589, 62.044, 63.021, 61.328%. No-audio fit prior62.00%, random binary50%. Isotropic predicted mean only 68.083% and ordinal 50.097%. Correctly rotated target after synchronized+60° microphone roll gives 77.458% versus 67.757% against unchanged target. This supports coarse directional structure beyond scalar room extent; it does not establish calibrated DoA or individual echo-wall correspondence.

Ambiguous-pair matched−donor effect 9.582 [7.843, 11.628]pp. Conditioned cue vs matched base joint -0.228 [-0.911, 0.423]pp; vs original GEOQ -1.497 [-2.474, -0.585]pp. Original GEOQ ambiguous-pair discrimination is 90.220 [87.776, 92.301]%, already much stronger than60.47% cue-only discrimination. Pair success is retrospective and conditional on a good/bad pair; it is not a deployable oracle.

Reliability: `A_followup.json` contains5bin reliability and Brier per sector/scene/model using a fixed sigmoid of predicted log-depth margin with temperature.5. This is an uncalibrated diagnostic confidence mapping, not a learned/calibrated posterior. Thresholds/settings were not tuned on held labels.

Repeat diagnostic reuses40poses from4older scenes, five cached renders/pose (original plus4repeat). Only8poses are Observed,32areFP references; do not describe all40 as observed sensor trials. A separate embedding ridge fits allNew12, which exclude these4old scenes; the upstream acoustic encoder saw those older scenes.
- Observed: 8poses, mean within-pose label agreement 82.812%, mean pair log-depth difference 0.114.
- FP: 32poses, mean within-pose label agreement 87.500%, mean pair log-depth difference 0.107.

This small existing repeat cache indicates nonzero render variation, not sensor reproducibility or proof of absolute depth accuracy. Furnishings, multipath, broad azimuth response, floor/ceiling contribution and height/reference mismatch are plausible limits, not isolated causal findings here. Scene102816615/102816852/108736689_177263340 show little or negative improvement over the dominant near-label prior despite useful aggregate accuracy. Unknown/door/wall semantics are not identifiable from these labels. Final judgment remains **MIXED: observable coarse directional cue, unproven additional localization information**. The next useful test is a conditional residual-information/physical-correspondence experiment, not a larger acoustic depth network.

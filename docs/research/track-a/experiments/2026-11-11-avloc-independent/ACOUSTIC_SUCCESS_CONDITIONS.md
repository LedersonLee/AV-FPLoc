# Acoustic-only success conditions — 2026-11-12

Track A; Exploratory; full reused Val19 /9,965 queries /19 scenes, seed42. A1 was selected by clean validation ray MAE; these are post-hoc analyses, not independent validation. Both grids use the identical frozen predictions. GT-derived error and geometry strata never enter inference.

## Observed outcomes

| Condition | 20cm grid | 10cm grid |
|---|---:|---:|
| Position≤0.1m successes |15 (3 scenes)|30 (7 scenes)|
| Of these, yaw error>30° |7|17|
| Of these, yaw error≤30° |8|13|
| Fine-success median ray MAE |0.398m|0.358m|
| Joint successes (≤1m,≤30°) |242|253|
| Joint-success median ray MAE |0.640m|0.662m|
| Lowest ray-error quartile, Joint failures |2,302|2,291|
| Highest ray-error quartile, Joint successes |1|2|
| Perfect-profile geometry success but A1 Joint failure |9,562|9,650|

All-query median ray MAE is1.458m. Ray-error quartile cutpoints are1.0453,1.4581,1.8789m. The20cm fine successes are confined to three scenes (5,8,2 cases); their count does not support a broad precision claim. Fine is position-only and must not be presented as full-pose accuracy.

Joint success by increasing ray-error quartile is7.627%,1.566%,0.482%,0.040% at20cm, and8.069%,1.566%,0.442%,0.080% at10cm. All19 scene-specific ray-MAE/Joint Spearman correlations are nonpositive. This is descriptive evidence linking smaller ray error to higher success, not proof that the scalar MAE alone determines discrimination. Even the most accurate quartile fails on most queries; directional pattern, uncertainty and competing floorplan poses remain relevant.

## Geometry does not by itself explain acoustic success

The ideal36-ray diagnostic uses GT distances to measure the profile-MAE margin between the best Joint-incorrect and Joint-correct hypotheses. That diagnostic is unavailable at inference and is not an acoustic recoverability bound. At20cm, Joint accuracy by increasing ideal-geometry margin quartile is5.299%,1.606%,1.405%,1.404%. Only6/19 scene-specific margin/Joint correlations are positive (4/19 at10cm). The proposed simple explanation that larger ideal geometric separation guarantees acoustic success is not supported here.

Competing explanations include systematic distance/shape prediction error and the mismatch between the unweighted ideal-profile diagnostic and learned uncertainty-weighted pose matching. Scene/pose composition may confound post-hoc strata. These data do not establish that geometry uniqueness harms localization. The9,562 cases where ideal geometry succeeds but A1 fails at20cm indicate that ambiguity of the perfect profile alone cannot explain most observed failures; they do not distinguish insufficient acoustic information from an inadequate predictor, adaptation, input representation, or objective.

## Evidence and reproduction

[Full aggregate strata, scene correlations and source hashes](acoustic_conditions_seed42.json). Geometry artifact hashes and exact scene/query alignment are checked; success aggregates reproduce the saved A1 evaluation to1e−12. No new inference, training, rendering or Test17 access. Code `50dd3b1`; training `62c58da20ed26ba95863638bbfd4926829999eec`. Run `analyze_acoustic_conditions.py --study <study-root> --out <fresh-output>` from the study code directory. Original per-query metrics and geometry results remain unchanged locally.

Quartile boundaries were computed from this reused development cohort, not preregistered thresholds. Small fine-success counts are reported descriptively without query-independence confidence claims. A3/interaction controls remain pending. This is not evidence that all acoustic localization approaches or the original AVLoc paper fail.

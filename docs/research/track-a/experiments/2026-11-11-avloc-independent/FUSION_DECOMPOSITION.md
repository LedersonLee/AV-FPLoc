# Fixed fusion decomposition: preserve visual FoV, add acoustic outside rays

2026-11-12 KST. Track A. Exploratory: predefined diagnostic after observing fusion failure, reused Val19, seed42. The full model benchmark and independent seed confirmation remain incomplete.

## Hypothesis and design

Two competing explanations for independent equal fusion failure are corruption of reliable visual FoV distributions and harmful extra acoustic rays outsideFoV. A2 training is unchanged. Using exact saved A0best99/A1best50 and independent-equal prediction arrays, construct the2×2 inference intervention below. No GT is used in construction, no parameter is fit and no candidate prefilter is changed. All19scenes/9,965queries use20cmXY/10°yaw full-bank scoring and identical success/NMS rules.

| FoV distribution | OutsideFoV | Joint | R@1m | Fine | NMS Top20 |
|---|---|---:|---:|---:|---:|
|Original A0|Absent|50.105%|50.948%|22.027%|88.751%|
|Equal moment mixture|Absent|7.396%|10.216%|0.462%|54.099%|
|Original A0|Original A1|52.303%|53.728%|22.539%|89.824%|
|Equal moment mixture|Original A1|9.553%|12.805%|0.482%|40.401%|

Construction source `457d3dd` (full source hash in prediction manifests); summary source `d9183aea316265f1567a01dc7ce43acfff55b925`. Exact FoV ordering, untouched outside slices and scale positivity tests passed. Both new full-cohort evaluations completed. [All metrics, source/row hashes and paired contrasts](fusion_decomposition_summary.json).

## Controlled findings

Adding outsideFoV A1 while preserving A0 inside yields micro Joint+2.1977pp, equal-scene mean+2.2835pp, paired19-scene bootstrap95% interval[1.2879,3.2363]pp. It repairs527 A0 failures and harms308 successes. NMS Top20 improves1.0738pp micro; macro interval[0.3368,2.1160]pp. Fine improves0.5118pp micro, but macro interval[−0.0082,1.0686]pp includes zero; a reliable fine-accuracy improvement is not established.

Conversely, replacing A0 FoV by the equal mixture without any outside rays reduces Joint42.7095pp micro (macro−42.2369pp, interval[−46.1546,−38.1055]pp). It repairs276 failures and harms4,532 successes. This demonstrates that the major loss can occur entirely within FoV. Adding outside rays to that weak fused-FoV condition improves Joint2.1576pp but reduces Top20 coverage13.6979pp; higher Top1 does not imply higher candidate coverage.

All intervals use10,000 paired scene bootstrap draws, seed20261111. No query/seed independence assumption is made.

## Interpretation, competing explanations and limits

For these frozen distributions, retaining accurate visual distributions is materially better than equal moment mixing; weak standalone acoustic predictions still provide modest complementary evidence outsideFoV. Thus the earlier A1-alone failure does not establish zero useful acoustic information. The strong claim that outside acoustic evidence is the main cause of the observed fusion collapse is contradicted by this intervention.

The inside mixture changes both mean and uncertainty scale. This study does not distinguish mean contamination, uncertainty inflation or their interaction, and does not establish that a learned router will solve either. Outside-ray gains can reflect learned directional evidence, geometry priors or correlated error patterns; causality at the physical-signal level is not established. The52.303% Joint result remains below fixed score fusion65.038%,B2+GEOQ73.071%. This is not a selected replacement architecture or an independently validated improvement. Seed43, A2, interaction-disabled and A3 remain required; no Test17 was accessed.

## Reproduction

```bash
python3 AV-FPLoc/scripts/research/track_a/avloc_20261111/test_fusion_decomposition.py
python3 AV-FPLoc/scripts/research/track_a/avloc_20261111/decompose_fusion_predictions.py --visual "$STUDY/predictions/A0_seed42" --audio "$STUDY/predictions/A1_seed42" --fused "$STUDY/predictions/independent_equal_seed42" --out "$STUDY/predictions/fusion_decomposition_seed42_v1"
```

Evaluate each new manifest with `evaluate_fields.py --predictions <mode-directory> --data <data_v1> --banks <banks_v1> --out <fresh-directory> --grid-cm20` (pass `--grid-cm 20` as separate arguments): `visual_inside_audio_outside` uses headF; `fused_fov` uses headV. The exact Docker commands and immutable source hashes are in local `fusion_decomposition_eval_v1/plan.json`; original predictions and results were preserved. Then run `summarize_fusion_decomposition.py --study "$STUDY" --out <fresh-summary-directory>`. Data and raw arrays stay local.

## Remaining calibration completion

Independent-equal10cm calibration review also passed all9,965 native pose/metric and crossfit replay checks. NLL24.2708→11.8872,ECE0.39466→0.05239,entropy0.46960→0.91567,Joint remains9.7742%; all5folds selectT10. [Verification](independent_equal_seed42_10cm_calibration_verification.json). All independent-fusion evaluation stages are complete; model research is not.

# Frozen independent A0+A1 fusion: initial negative control

2026-11-12 KST; Track A; Exploratory, single seed42, reused Val19. Full research remains incomplete.

## Hypothesis and controlled intervention

Test whether independently trained visual/audio ray distributions provide useful localization when combined without feature exchange. Freeze A0best99 and A1best50. Within11 FoV rays, use an equal mixture and match its mean and variance to a Laplace distribution; outside FoV use A1 alone. Preserve full36-ray matching, all19scenes/9,965queries,20cmXY/10°yaw grid and success criteria. GT is diagnostic only. No network training or weight/temperature selection is performed. This is a fixed inference intervention, not a model trained with a fixed router.

Runner code `846c49d4b87563e509faceff6c07b69296008d7b`. Immutable source/inputs/image hashes are recorded by the local independent_fusion_seed42_v1 plan. Analytic checks passed for mixture mean/variance, audio-only outsideFoV, equal-expert invariance, normalized weights and wrapped FoV ordering. Full20cm evaluation and ray analysis completed.10cm evaluation and calibration remain in progress.

## Observations

| Method | Joint | R@1m | R@0.1m | NMS Top20 |
|---|---:|---:|---:|---:|
|A0|50.105%|50.948%|22.027%|88.751%|
|A1|2.428%|9.252%|0.151%|22.689%|
|Independent equal fusion|9.553%|12.805%|0.482%|40.401%|

Relative to A0, fusion repairs418 failures and harms4,459 successes. Micro Joint change−40.552pp; equal-scene mean−40.078pp, paired19-scene bootstrap95% interval[−44.098,−35.965]pp. Relative to A1, repairs789/harms79; micro+7.125pp; scene-mean+7.104pp with interval[5.876,8.553]pp.10,000 paired scene resamples, seed20261111; scenes are resampling units, not queries. [Exact contrasts and row hashes](independent_fusion_contrasts.json). Computed using committed `paired_summary.paired` after exact query-ID alignment; this invocation is an ad-hoc aggregation, not an additional trained model.

[All-baseline table](INDEPENDENT_FUSION_COMPARISON.md) and [aggregate provenance](independent_fusion_comparison.json). Existing baseline aggregation source `81bcc165bba8bef19b43239d3d6a459f3cb938e6`.

## Interpretation and limits

For these fixed predictions and this particular combination rule, adding A1 harms A0 substantially. This does not show that audio inherently lacks useful information, nor that learned joint interaction or routing cannot help. The intervention changes both FoV distribution fusion and inclusion of25 outside-FoV rays; it does not isolate which mechanism causes the loss. A controlled decomposition should retain A0 insideFoV while adding A1 outside, and separately evaluate fused FoV without outside rays. These are follow-up diagnostics, not yet executed. A2 and interaction-disabled controls remain necessary, along with seed43.

A1's separate10cm result remains poor: Joint2.5389%,R@1m9.2223%,R@0.1m0.3011%. [Metrics/hashes](a1_10cm_metrics.json). Finer discretization alone does not restore this learned acoustic predictor; it is not evidence against the original paper's sensor/input setting.

## Reproduction

```bash
python3 AV-FPLoc/scripts/research/track_a/avloc_20261111/test_fusion_predictions.py
python3 AV-FPLoc/scripts/research/track_a/avloc_20261111/evaluate_independent_fusion.py --workspace "$WORKSPACE" --study "$STUDY" --out "$WORKSPACE/experiments/avloc_20261111/independent_fusion_seed42_v1" --seed 42
```

Requires completed frozen A0/A1 prediction manifests; refuses existing outputs. Source snapshots, raw arrays and logs stay local. Report contains aggregates only. Parallel A2 training and evaluation invalidate isolated-runtime claims. No Test17 use or Track B changes.

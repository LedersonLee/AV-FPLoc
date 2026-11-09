# Candidate grid, yaw and valid-mask sensitivity

Date2026-11-09, Track B, status: fixed-checkpoint F1 grid evaluation complete; matched-model evaluation remains pending. Evidence: Exploratory reused six development scenes, no Test use. This implements the candidate construction part of required X5; it does not establish resolution robustness.

## Declared comparison and hypothesis

Hold observation features and learned checkpoints fixed while varying only the exhaustive candidate bank. Compare baseline .25m/36yaw, .20m/36yaw, .50m/36yaw, .25m/12yaw, .25m/72yaw and .25m/36yaw with candidate validity restricted to native-map clearance≥.1m. Report every variant, oracle coverage, all-query performance and exclusion diagnostics; do not tune to a favorable grid or silently discard newly excluded queries. Competition: an apparent method effect may actually be discretization/valid-mask coverage. The erosion is a conservative mask sensitivity probe, not a claim about physical traversability.

## Construction and verified result

All36 map-only banks (six variants×six scenes) completed with exit0. Original map files match archived source hashes. Candidate origin is the map pixel-center origin, never a query trajectory. Rays are16 bearings, max20m, step.1m, derived from the unmodified native footprint. Erosion changes only candidate validity; it does not alter map geometry or ray endpoints. The two channels of the model FP raster will remain the same.

All six baseline candidate tensors exactly equal the frozen .25m/36 tensors. This includes ordering, positions, yaw, normalized ray lengths and hit flags. The cache stores only map/candidate data, no observations, labels, candidate RIRs or proposals. Original scene packages are read solely for baseline equality; their GT fields do not affect candidate construction. Total serialized map-variant storage: 515793764 bytes, external only. This is diagnostic geometry-bank storage for all six alternatives, not a final deployment storage claim.

## Reproduction and next checks

Actual construction code SHA `74e761a2a1584539daab0bb52d8942a7394a2beb`, Docker avfploc:cu124-clean, CPU, source/dataset/reference mounts read-only, two threads. External root `/home/jeongeonlee/track1/experiments/track_b_grid_20261109/cache_v1`. Reproduce with `python -m track_b.scripts.cache_grid_variants --root /data --reference-cache /reference --out /results/cache_v1` into a new path. Protocol and per-bank hashes/counts/times are archived; large tensors remain external.

No training occurs in this preparation, so optimizer/baseline model results/seed uncertainty/ablation gains are not applicable yet. Next evaluate the strong independent baseline and completed matched models without retraining, preserving original calibration or explicitly marking any recalibration. NLL changes across candidate counts are not directly comparable continuous densities; report candidate count, normalized entropy and interpretation. GT inclusion and oracle quantization must be recalculated for each bank, including all-query and valid-subset accounting. Full Track B objective remains incomplete.

## Fixed-checkpoint evaluation started

The strong anchored visual and independent acoustic checkpoints are frozen for all six bank variants and seeds42/43/44. Every query is scored over the entire declared bank for visual, acoustic and .85/.15 fixed fusion. No grid-dependent temperature/weight fit. The evaluator records raw1/5/10/20, diverse20, strict pose success, continuous errors, nearest-grid rank/ties, normalized entropy, prediction-centered reliability and scene-paired differences. NLL is retained as nearest-valid-grid diagnostic with explicit candidate count and gain over uniform; it must not be read as directly comparable continuous density across grids.

GT validity here means whether the GT's nearest grid raster cell is inside the declared bank validity mask. It is explicitly a grid-cell criterion, not exact equality to a sampled pose or a physical footprint boundary proof. Out-of-bounds/invalid cases are flagged, never clamped or removed; all-query and flagged-valid-subset summaries are both produced. The GT itself is unchanged in every error/coverage calculation. Per-query predictions and field-derived diagnostics stay outside Git.

Actual source SHA `1f79b9d95e699cebd80374e09f69f3874522d532`; command `python -m track_b.scripts.evaluate_grid_sensitivity --cache /structure/cache --grid-cache /results/cache_v1 --foundation /foundation --acoustic-run /structure/supervised_v1 --out /results/independent_v1`. Observation session42021 was verified live, processing seed43. Results and baseline-replay mismatch counts remain pending; do not infer successful completion from the protocol file.


## Complete fixed-checkpoint result (supersedes pending status)

All six variants × three seeds × three methods completed,20,736 query-method evaluations. Every variant preserves all384 query grid-cell memberships in this cohort, including the .1m erosion; all-query and valid-subset summaries therefore coincide. This does not prove stronger mask errors are harmless. Native map geometry and learned observation/map weights are unchanged.

| Bank | F1 Joint % | Fine % | Raw20 % | Diverse20 % | Fine oracle % |
|---|---:|---:|---:|---:|---:|
| .25m,36yaw | 23.872 | 2.170 | 61.892 | 84.375 | 52.865 |
| .20m,36yaw | 24.392 | 3.472 | 59.288 | 84.635 | 72.917 |
| .50m,36yaw | 23.003 | 1.563 | 69.358 | 83.333 | 15.365 |
| .25m,12yaw | 22.569 | 1.910 | 63.628 | 84.288 | 52.865 |
| .25m,72yaw | 24.045 | 2.083 | 56.076 | 84.462 | 52.865 |
| .25m,36yaw,eroded .1m | 24.045 | 2.170 | 61.892 | 84.375 | 52.865 |

Relative to baseline, .20m Joint +0.521 pp [95% paired scene CI −0.434,+1.302]; .50m −0.868 [−2.257,+0.521];12yaw −1.302 [−3.559,+0.694];72yaw +0.174 [−0.434,+0.781]. These results do not establish a reliable Joint improvement by densifying the bank. Fine oracle is only a geometric upper bound, not achieved precision.

Raw20 reacts strongly to density: .50m +7.465 pp [5.556,9.722] despite a much lower fine oracle;72yaw −5.816 pp [−7.118,−4.427] despite nearly unchanged diverse20. Interpretation: rawK conflates localization evidence with discretization and concentration of nearby samples. It is not comparable across candidate densities as if the search problem were unchanged. Decision: retain the original bank for the primary comparison and report the entire sensitivity table, not select the best observed variant. Scope is the trained-on-.25m independent F1, not every future model.

Baseline stricter success is16.146% at(.5m,15°) and5.122% at(.2m,10°). Median translation error4.31m, p90≈11.31m; median yaw80°, p90≈170°. These large errors constrain any practical localization claim despite coarse gains. For uncalibrated τ1, MAP-neighborhood confidence averages≈.014 against Joint≈.239 (10-bin ECE≈.225); this is a discretized region-confidence diagnostic, not certified continuous-pose uncertainty. No new calibration is fit across grids. The full reliability/selective-risk results are archived; their aggregate computation does not create independent validation.

## Numerical replay correction and preserved first run

First execution `independent_v1` completed with identical baseline Joint/R@1m/Fine/raw20 but563 nearest-grid NLL tolerance mismatches (556 acoustic,7 fixed; max0.00125122nats). It encoded all64 observations at once, unlike the archived reference's batch8. The controlled repeat `independent_v2` changes only encoder batching to8 and reproduces all3,456 baseline query-method cases including NLL with zero mismatches. Both runs/logs remain external; the first aggregate and numeric diagnosis remain in Docs. Do not describe the first run as bitwise or tolerance-exact probability reproduction. The changed batching explains this observed replay difference under the current numerical implementation; no general hardware-invariance claim follows.

Actual final evaluation code SHA `6e6f3a5380b6b124ebfdb978b4c1978baad7c275`; first evaluation code remains `1f79b9d` (full SHA in earlier provenance). Registry/aggregation source `d82a1f9bba5390679d7088e9088034a5dcd16eaa`. Reproduce the earlier evaluator command with a fresh output `independent_v2`; register via `python -m track_b.scripts.register_grid_evaluation --run <output> --out <new-registry> --code-sha 6e6f3a5380b6b124ebfdb978b4c1978baad7c275`. No model retraining. Only aggregate/provenance files enter Git; raw query rows and .pt banks remain external. Further final-model grid checks, full-pipeline OOF discrimination, scale and deployment evaluation remain incomplete.

## Completed matched-model follow-up

All nine final joint foundation models have now completed the same six-grid evaluation (20,736cases, zero3,456-case baseline replay mismatches). See [matched model sensitivity](matched_models.md). Attention remains below independentF1 on every grid; coarsening increases raw20 without demonstrating MAP improvement. Earlier independent-F1 evaluation and first-run numeric discrepancy remain preserved.

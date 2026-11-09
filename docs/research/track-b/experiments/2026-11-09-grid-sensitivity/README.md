# Candidate grid, yaw and valid-mask sensitivity

Date2026-11-09, Track B, status: map-only preparation complete; fixed-checkpoint performance evaluation pending. Evidence: Exploratory reused six development scenes, no Test use. This implements the candidate construction part of required X5; it does not establish resolution robustness.

## Declared comparison and hypothesis

Hold observation features and learned checkpoints fixed while varying only the exhaustive candidate bank. Compare baseline .25m/36yaw, .20m/36yaw, .50m/36yaw, .25m/12yaw, .25m/72yaw and .25m/36yaw with candidate validity restricted to native-map clearance≥.1m. Report every variant, oracle coverage, all-query performance and exclusion diagnostics; do not tune to a favorable grid or silently discard newly excluded queries. Competition: an apparent method effect may actually be discretization/valid-mask coverage. The erosion is a conservative mask sensitivity probe, not a claim about physical traversability.

## Construction and verified result

All36 map-only banks (six variants×six scenes) completed with exit0. Original map files match archived source hashes. Candidate origin is the map pixel-center origin, never a query trajectory. Rays are16 bearings, max20m, step.1m, derived from the unmodified native footprint. Erosion changes only candidate validity; it does not alter map geometry or ray endpoints. The two channels of the model FP raster will remain the same.

All six baseline candidate tensors exactly equal the frozen .25m/36 tensors. This includes ordering, positions, yaw, normalized ray lengths and hit flags. The cache stores only map/candidate data, no observations, labels, candidate RIRs or proposals. Original scene packages are read solely for baseline equality; their GT fields do not affect candidate construction. Total serialized map-variant storage: 515793764 bytes, external only. This is diagnostic geometry-bank storage for all six alternatives, not a final deployment storage claim.

## Reproduction and next checks

Actual construction code SHA `74e761a2a1584539daab0bb52d8942a7394a2beb`, Docker avfploc:cu124-clean, CPU, source/dataset/reference mounts read-only, two threads. External root `/home/jeongeonlee/track1/experiments/track_b_grid_20261109/cache_v1`. Reproduce with `python -m track_b.scripts.cache_grid_variants --root /data --reference-cache /reference --out /results/cache_v1` into a new path. Protocol and per-bank hashes/counts/times are archived; large tensors remain external.

No training occurs in this preparation, so optimizer/baseline model results/seed uncertainty/ablation gains are not applicable yet. Next evaluate the strong independent baseline and completed matched models without retraining, preserving original calibration or explicitly marking any recalibration. NLL changes across candidate counts are not directly comparable continuous densities; report candidate count, normalized entropy and interpretation. GT inclusion and oracle quantization must be recalculated for each bank, including all-query and valid-subset accounting. Full Track B objective remains incomplete.

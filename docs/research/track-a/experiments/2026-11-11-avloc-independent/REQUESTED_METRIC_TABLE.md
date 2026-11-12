# Requested final-pose metrics — updated 2026-11-13

All values percent; higher is better. Val19:9,965 queries,19 scenes; reused development. P=position error; Y=circular yaw error. All thresholds inclusive. Angular-only columns do not constrain position. B2/GEOQ:mean of42/43; AVLoc-style arms:seed42. Frozen baseline cache label42 is not a new training replicate.

## 20cm XY /10° yaw

| Method | Seeds | P≤0.1m | P≤0.5m | P≤1.0m | P≤1.0m & Y≤30° | Y≤10° | Y≤30° | Y≤50° |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Existing Visual | 42 | 17.43 | 44.32 | 46.99 | 45.85 | 55.29 | 56.68 | 57.03 |
| FP-RIR Cosine | 42 | 5.35 | 22.45 | 30.38 | 21.36 | 23.12 | 36.49 | 42.07 |
| Fixed Fusion | 42 | 24.28 | 62.65 | 66.75 | 65.04 | 69.58 | 71.87 | 72.20 |
| B2 | 42,43 | 22.50 | 66.21 | 70.97 | 69.31 | 73.18 | 75.17 | 75.46 |
| B2+GEOQ | 42,43 | 23.20 | 69.21 | 74.42 | 73.07 | 76.16 | 78.28 | 78.49 |
| A0 | 42 | 22.03 | 49.17 | 50.95 | 50.11 | 59.59 | 60.26 | 60.49 |
| A1 | 42 | 0.15 | 3.44 | 9.25 | 2.43 | 12.02 | 17.79 | 31.90 |
| A2 | 42 | 30.47 | 63.94 | 65.72 | 64.93 | 71.09 | 71.40 | 71.57 |
| A2_disabled | 42 | 12.75 | 38.07 | 41.71 | 40.34 | 50.02 | 51.36 | 52.00 |

## 10cm XY /10° yaw

| Method | Seeds | P≤0.1m | P≤0.5m | P≤1.0m | P≤1.0m & Y≤30° | Y≤10° | Y≤30° | Y≤50° |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A0 | 42 | 38.91 | 59.71 | 61.27 | 60.46 | 69.03 | 69.54 | 69.77 |
| A1 | 42 | 0.30 | 3.52 | 9.22 | 2.54 | 12.46 | 18.18 | 32.48 |
| A2 | 42 | 49.33 | 73.17 | 74.84 | 74.18 | 79.47 | 79.66 | 79.81 |
| A2_disabled | 42 | 19.36 | 43.97 | 47.51 | 46.39 | 55.65 | 56.81 | 57.35 |

The20cm table is the matched primary comparison. The10cm table changes pose sampling for the same frozen networks; it must not imply model superiority over20cm baselines. Only explicitly included, completed and evaluated arms appear; absence does not indicate zero performance. Seed43 A0 was interrupted and is excluded. No method was selected or retrained for these additional thresholds. Existing fine/r1/joint flags were reproduced exactly from stored errors for every query.

Training code: `62c58da20ed26ba95863638bbfd4926829999eec`. Script: `requested_metric_table.py --study <study-root> --out <fresh-output>`. Source/output hashes in [summary](requested_metrics.json). Single seed and reused validation limit generalization; no Test17 access.

Updated with the completed interaction control using code `a8f1a60`; [original pre-control snapshot](requested_metrics_before_disabled.json) retained. [Control interpretation](INTERACTION_CONTROL_RESULTS.md).

## What the grid resolution means

The20cm/10cm values are XY pose-candidate spacings, not the floorplan raster resolution and not an acoustic-specific input resolution. The prepared floorplan maps use1cm raster spacing; ray casting uses the original binary-pixel boundary geometry. The main matching grid samples valid XY positions every20cm, each with36 candidate yaws at10° intervals. The separate10cm evaluation samples finer XY positions while retaining the same floorplan, checkpoint, predicted ray distributions and yaw spacing. This applies to global localization across modalities. Val19 candidate totals are71,383 XY cells at20cm and286,731 at10cm (approximately4×; clearance/mask boundaries prevent an exact factor). See the existing bank-cost audit for underlying manifests.

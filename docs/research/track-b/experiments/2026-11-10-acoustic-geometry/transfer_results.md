# Geometry encoder transfer — inner validation

Exploratory inner-validation source families; not independently validated
10,000 paired family bootstrap draws, seeds then same-family variants averaged; uncorrected comparisons

| Arm | Acoustic R@1m % | Acoustic joint % | Fixed fusion joint % | Fixed fusion fine % | Fusion NLL |
|---|---:|---:|---:|---:|---:|
| legacy | 12.744 | 7.633 | 38.558 | 3.353 | 9.304 |
| scratch | 8.594 | 1.465 | 27.669 | 2.295 | 9.564 |
| scratch_total | 11.410 | 2.865 | 33.366 | 2.930 | 9.484 |
| frozen | 8.187 | 1.497 | 26.514 | 2.360 | 9.582 |
| finetuned | 9.082 | 1.449 | 26.204 | 2.246 | 9.558 |
| joint | 8.740 | 1.156 | 27.197 | 2.132 | 9.565 |

Shared visual joint: 24.235% (same predictions across every acoustic arm).

Same local steps and same total optimizer updates are distinct controls; pretraining and dense-scoring FLOPs differ. Full paired intervals, all seeds, errors, entropy, candidate coverage and costs are in summary.json.

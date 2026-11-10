# Frozen official Test evaluation

19 development holdout scenes previously reused upstream/other studies. Three official Test observed scene directories already exist; prior visual Test usage and upstream Hypersim asset-family overlap are not fully excluded, so no blanket pristine/full-pipeline independent claim. Fresh raw acoustic rendering does not itself erase scene exposure.

10,000 source-family paired bootstrap draws after seed averaging; uncorrected intervals. No checkpoint, model or threshold updates followed these results.

## furnished

| Encoder/head | MAE m (95% CI) | RMSE m | Angular correlation within query | Shuffle − normal MAE m |
|---|---:|---:|---:|---:|
| echo/regression | 1.535 [1.352,1.749] | 2.355 | 0.0134 | 0.306 [0.258,0.360] |
| echo/categorical | 1.587 [1.432,1.772] | 2.169 | 0.0070 | 0.239 [0.196,0.289] |
| spatial/regression | 1.567 [1.366,1.808] | 2.366 | 0.0159 | 0.263 [0.223,0.307] |
| spatial/categorical | 1.738 [1.586,1.923] | 2.251 | 0.0244 | 0.120 [0.095,0.145] |

| Categorical encoder | NLL | CRPS m | 90% coverage | 90% width m |
|---|---:|---:|---:|---:|
| echo | 3.283 | 1.125 | 0.925 | 6.965 |
| spatial | 3.265 | 1.158 | 0.945 | 7.449 |

| Arm | Acoustic R@1m % | Acoustic joint % | Fusion R@1m % | Fusion joint % | Fusion fine % | Fusion NLL |
|---|---:|---:|---:|---:|---:|---:|
| legacy | 9.375 | 5.607 | 36.887 | 34.835 | 3.064 | 9.440 |
| scratch | 6.556 | 1.011 | 28.431 | 25.613 | 1.471 | 9.699 |
| scratch_total | 8.609 | 2.298 | 31.801 | 29.075 | 2.022 | 9.623 |
| frozen | 5.515 | 1.256 | 26.103 | 23.468 | 1.379 | 9.717 |
| finetuned | 7.016 | 1.011 | 28.615 | 25.766 | 1.532 | 9.691 |
| joint | 6.036 | 1.134 | 27.757 | 25.153 | 1.501 | 9.699 |

Shared visual joint: 23.070%. Full entropy, normalized entropy, errors, rawTop20, oracle coverage, all seeds and paired transfer intervals remain in the aggregate.

## stage_only

| Encoder/head | MAE m (95% CI) | RMSE m | Angular correlation within query | Shuffle − normal MAE m |
|---|---:|---:|---:|---:|
| echo/regression | 1.620 [1.429,1.846] | 2.357 | 0.0129 | 0.210 [0.160,0.263] |
| echo/categorical | 1.691 [1.500,1.914] | 2.360 | 0.0052 | 0.059 [0.025,0.105] |
| spatial/regression | 1.657 [1.434,1.919] | 2.523 | 0.0153 | 0.198 [0.166,0.233] |
| spatial/categorical | 1.725 [1.562,1.922] | 2.266 | 0.0228 | 0.089 [0.069,0.110] |

| Categorical encoder | NLL | CRPS m | 90% coverage | 90% width m |
|---|---:|---:|---:|---:|
| echo | 3.431 | 1.270 | 0.850 | 5.328 |
| spatial | 3.279 | 1.175 | 0.905 | 6.449 |

| Arm | Acoustic R@1m % | Acoustic joint % | Fusion R@1m % | Fusion joint % | Fusion fine % | Fusion NLL |
|---|---:|---:|---:|---:|---:|---:|
| legacy | 7.782 | 4.442 | 35.631 | 33.640 | 2.727 | 9.483 |
| scratch | 5.147 | 0.705 | 25.276 | 22.978 | 1.746 | 9.723 |
| scratch_total | 5.147 | 1.225 | 26.471 | 24.295 | 1.930 | 9.662 |
| frozen | 4.810 | 0.888 | 24.479 | 22.212 | 1.379 | 9.725 |
| finetuned | 4.841 | 0.613 | 25.429 | 22.947 | 1.593 | 9.720 |
| joint | 4.779 | 0.797 | 23.928 | 21.415 | 1.624 | 9.729 |

Shared visual joint: 23.070%. Full entropy, normalized entropy, errors, rawTop20, oracle coverage, all seeds and paired transfer intervals remain in the aggregate.

Stage-only removes all object instances, including doors/windows. This is not a pure furniture intervention. The separate development opening-preserving control is required to interpret object-removal effects. Predictions and targets use the same conditional valid-direction mask. Scene-disjoint evaluation does not erase prior visual/upstream asset exposure.

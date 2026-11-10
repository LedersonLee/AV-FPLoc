# Frozen development holdout evaluation

19 historical validation scenes/19 families,64 queries each; all18 representation and18 acoustic transfer checkpoints plus3 shared visual checkpoints frozen before evaluation. Historical reuse/upstream exposure remains; Exploratory evidence, not independent Test.

| Encoder/head | MAE m | Within-query angular correlation | Scalarized MAE | Rotated180° MAE |
|---|---:|---:|---:|---:|
| echo/regression | 1.4900 | -0.0103 | 1.4836 | 1.4861 |
| echo/categorical | 1.5415 | 0.0003 | 1.5382 | 1.5410 |
| spatial/regression | 1.4924 | -0.0100 | 1.4911 | 1.4914 |
| spatial/categorical | 1.6778 | -0.0132 | 1.6775 | 1.6775 |

| Arm | Fusion joint1m30° % | Fusion fine0.1m % | Fusion NLL |
|---|---:|---:|---:|
| legacy | 33.032 | 2.166 | 9.441 |
| scratch | 24.068 | 1.316 | 9.697 |
| scratch_total | 28.043 | 1.700 | 9.620 |
| frozen | 23.629 | 1.453 | 9.713 |
| finetuned | 23.657 | 1.343 | 9.691 |
| joint | 24.205 | 1.261 | 9.701 |

The frozen holdout reproduces the directional failure and absence of a clear geometry-pretraining gain. All different-position shuffle, probabilistic calibration, phase ablation, localization entropy, per-family values and paired transfer intervals are in [aggregate](development_heldout_summary.json). No model or threshold changes follow these results.

Actual cache source45e4ce0; evaluation6907b72. Source abbreviations resolve in this repository; frozen weights are identified separately in [method freeze](method_freeze.json).

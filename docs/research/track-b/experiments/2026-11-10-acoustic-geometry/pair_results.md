# Native matched object-removal controls

16 development families,8 poses per family,3 seeds; scene/seed averaged before paired family bootstrap. Masks/GT identical across conditions. Exploratory simulation evidence.

| Encoder/head | Fresh furnished MAE | All objects removed MAE | Openings retained MAE | Furnished − openings retained,95% CI |
|---|---:|---:|---:|---:|
| echo/regression | 1.510 | 1.589 | 1.474 | 0.035 [0.007,0.071] |
| echo/categorical | 1.536 | 1.675 | 1.559 | -0.023 [-0.057,0.001] |
| spatial/regression | 1.535 | 1.661 | 1.523 | 0.012 [-0.011,0.035] |
| spatial/categorical | 1.695 | 1.715 | 1.762 | -0.066 [-0.091,-0.045] |

All-object removal also removes door/window objects. The opening-preserving control keeps registry-classified openings and removes other objects, potentially including fixed fixtures. The initial stage-only comparison therefore cannot identify a furniture-only effect. A near-zero difference is not an equivalence test or proof of invariance. Directional reconstruction remains unsuccessful; no paired consistency training was used.

Historical-to-fresh furnished differences and first-query fresh-simulator repeat prediction/waveform changes are retained in [full paired results](pair_summary.json). These repeats cover one query per family, not every pose. CPU evaluation/source `4d36bc4`; renderer/source hashes are in the manifests.

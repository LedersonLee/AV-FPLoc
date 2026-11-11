# Matched Val19 comparison (development)

Not independent validation; pending AVLoc arms must not be interpreted as missing failures.

| Method | Seeds | R@1m | Joint | R@0.1m | Macro joint |
|---|---|---:|---:|---:|---:|
| b2 | [42, 43] | 70.968% | 69.313% | 22.499% | 68.918% |
| visual | [42] | 46.994% | 45.850% | 17.431% | 45.300% |
| acoustic | [42] | 30.376% | 21.365% | 5.349% | 21.098% |
| fixed | [42] | 66.754% | 65.038% | 24.285% | 64.631% |
| B2_GEOQ | [42, 43] | 74.415% | 73.071% | 23.201% | 72.733% |
| B2_Support | [42, 43] | N/A | 71.139% | N/A | 70.838% |
| A0 | [42] | 50.948% | 50.105% | 22.027% | 49.670% |
| A1 | [42] | 9.252% | 2.428% | 0.151% | 2.488% |
| independent_equal | [42] | 12.805% | 9.553% | 0.482% | 9.592% |

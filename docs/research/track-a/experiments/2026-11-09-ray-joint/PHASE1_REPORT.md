# Phase1 — Ray capacity oracle (completed; overall study active)

Nondeployable geometry truth diagnostic; same12reused development scenes ×128queries. Existing20cmXY/10degyaw bank, same pureVTop20 and GT-free ambiguity masks. No acoustic model is trained or selected by these results. K8 interpolates the common36source at45degree intervals; subset8 selects nearest source bins. Raw4/raw12 are exactly subsets of36outputs, verified numerically. Body-left convention is unchanged. Continuous-query / quantized-candidate mismatch remains.

| Representation | Fullbank joint% | Fine% | Top20% | FixedpureV joint% | Ambiguous pair% |
|---|---:|---:|---:|---:|---:|
| raw4 |48.893|28.971|98.763|69.206|91.166|
| raw8 interpolated |92.122|48.307|100.000|79.102|92.989|
| raw12 |93.164|43.880|100.000|80.143|93.085|
| raw36 |96.289|46.745|100.000|83.919|93.825|
| 36subset8 rounded |90.039|47.266|100.000|77.995|92.127|
| front12 only |66.536|30.599|94.466|56.576|71.031|
| old smooth4 |51.107|25.651|98.958|69.271|91.636|

Observation: increasing angular coverage/density materially improves oracle joint selection. Fine accuracy is not monotone withK, so ray count alone does not establish improved metric precision. Fixedcandidate coverage stays90.690% for allarms; improvements there reflect discrimination rather than candidate extraction. Top20 nearly saturates even for4rays: ambiguity among plausible peaks, not absence of a good candidate, limits Top1.

Interpretation: four extents admit many distinct layouts/poses. Extra angular samples resolve shape differences that the coarse summary discards. Front-only12 vs omnidirectional12 shows coverage matters as well as count. Eight interpolation vs rounded subset differentiates angular sampling from output dimensionality. Physical acoustic predictability is untested by this oracle and must be established in Phase2.

Competing explanation not eliminated: quantization and ray-casting mismatch can affect fine rankings, and many correlated ray directions should not be treated as independent evidence. The100%Top20 result is not a real acoustic localization result or independent-scene claim.

Code introduction:0455b8a (full SHA resolvable in repository), oracle_capacity.py; per-scene scores/indices/labels in local oracle/. Aggregate oracle_results.json and oracle_summary.json. NoTest observations/results. Phase2/3 remain mandatory; study must not stop here.

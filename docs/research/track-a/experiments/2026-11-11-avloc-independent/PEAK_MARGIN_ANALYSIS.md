# Peak margin analysis — 2026-11-17

Track A; Exploratory; matched20cm/10°Val19,9,965queries, seed42 or frozen baseline cache42. No new inference, training or Test17 access. Raw and distinct-NMS peak gaps, entropy and GT-region mass were aggregated for nine methods with exact scene/query alignment. [Full aggregates and input hashes](peak_margins_seed42.json).

| Method | Distinct gap median: success | Failure | Spearman with Joint | Failure entropy median |
|---|---:|---:|---:|---:|
|FP-RIR cosine|0.01728|0.01054|0.1721|0.99746|
|B2|0.13194|0.02126|0.5609|0.97686|
|A1|0.10229|0.05098|0.0785|0.49256|
|A2|91.6461|17.4095|0.5569|0.000182|
|A3|80.7195|13.6758|0.5774|0.000777|

Margins are native-score differences: their units/scales differ across methods, so raw magnitudes cannot establish which method has stronger evidence. Within each method, larger gaps associate with success, but A2/A3 failures still have extremely low entropy. Thus sharpness or a nonzero gap does not certify correct localization. These are pooled descriptive correlations, not causal or scene-independent significance tests. GT-region mass and success labels are post-hoc diagnostics only. Temperature controls are reported separately.

The first run failed before output creation because installed SciPy exposes a tuple result without the newer .statistic attribute. Accessing the first tuple element preserves the same statistic; the complete rerun passed. No failed result was published as a measurement.

Reproduce: `summarize_peak_margins.py --study <study> --out <fresh.json>`. Executed source commit 288ce1c9f53210000f7edaf5a31fcaf101541ea9; exact source SHA256 and input row hashes are in the JSON. Existing predictions and final poses were not changed.

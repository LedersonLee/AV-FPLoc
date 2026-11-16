# A3 initial matched results — 2026-11-16

Track A; seed42, Val19/9,965queries, reused development, Exploratory. Training completed87epochs, best67. No Test17 use.

## Observations

| Method | P≤0.1m | P≤0.5m | P≤1m | P≤1m & yaw≤30° | yaw≤10° | yaw≤30° | yaw≤50° |
|---|---:|---:|---:|---:|---:|---:|---:|
| A2 |30.47|63.94|65.72|64.93|71.09|71.40|71.57|
| A3 |30.92|65.84|67.75|66.92|72.87|73.21|73.28|
| B2+GEOQ |23.20|69.21|74.42|73.07|76.16|78.28|78.49|

All percentages, identical20cm XY/10° yaw candidate grid. A2/A3seed42; B2+GEOQ averages seeds42/43. Angular-only metrics do not constrain position. Fine is positional only. A3 Top5/Top20 NMS joint coverage87.01/95.32%; rayMAE0.55555847m, in-FoV0.17745747m. Native entropy0.0115324 is not calibration evidence. [Native aggregate](A3_20cm_summary.json).

A3−A2 Joint micro+1.997pp, equal-scene bootstrap95%CI[1.222,2.964]pp;759repairs/560harms. Fine micro+0.452pp, macroCI[−0.211,1.253]pp includes zero. A3−B2+GEOQ Joint micro−6.147pp, macroCI[−7.808,−4.439]pp; fine+7.717pp, macroCI[5.969,9.157]pp. [Paired comparison evidence](A3_initial_contrasts.json).

## Interpretation and competing explanations

A3 improves the joint success metric over A2 on these reused development scenes, but does not exceed B2+GEOQ on broad localization success. Fine-position accuracy favors A3. No universal superiority follows. A3-vs-A2 changes training and architecture together; this comparison cannot isolate the router. The same-A3-head uniform-weight control is still pending and will test an inference-level contribution, not independent fixed-router training. Single-seed scene bootstrap does not estimate optimization-seed variance or pristine generalization.

## Remaining checks

10cm evaluation, both-grid calibration replay, router error/weight analysis, uniform control and score figures remain in the running evaluation/finalization queues. No result is inferred from an incomplete scene subset. No models were retrained or new FP-RIRs rendered. The aggregate and requested metric flags were recomputed from all19scene outputs. Full checkpoint/both-grid integrity is queued after10cm completes.

Reproduction: `compare_router.py --study <study> --out <fresh-dir> --grid-cm 20` at code f20d22f; exact source hashes in comparison. Native evaluation source snapshot and checkpoint hashes are recorded in the native aggregate and local evaluation plan. Training62c58da20ed26ba95863638bbfd4926829999eec. Postprocessing source baf28c0 waits for completion receipts and stops on failure, using fresh outputs and immutable source snapshots. It does not publish unchecked results.

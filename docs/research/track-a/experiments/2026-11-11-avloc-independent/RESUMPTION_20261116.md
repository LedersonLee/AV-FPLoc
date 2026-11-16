# Resume audit — 2026-11-16

Track A; Exploratory development evidence. Test17 not accessed.

## Observed status

A0, A1, A2 and A2-disabled seed42 training and both-grid evaluation/calibration reviews are complete. A3 container exited successfully: 87 contiguous epoch records, best epoch67 ray MAE0.5555584779410163. The completion receipt reports last_epoch88 because the patience termination counter is one beyond the final completed epoch; original receipt retained. No A3 localization result is available yet.

The previous A3 evaluation waiter was absent and had created no output directory or predictions. Resumed the frozen checkpoint evaluation with unchanged training-source hash checks, pinned image, immutable evaluation snapshot, fresh output directory and no training. The initial detached launch did not survive the tool shell; verified absence before launching the active foreground-managed execution. First stage `predict` started successfully.

Local execution receipt: `experiments/avloc_20261111/a3_evaluation_resumed_20261116/plan.json` under the workspace. Driver source is inherited from code base60bb20e; training source62c58da20ed26ba95863638bbfd4926829999eec. Evaluation sequence: predict,20cm evaluation/ray analysis,10cm evaluation/ray analysis, both calibration sweeps, router analysis, uniform-weight export and both-grid uniform evaluations. These are11 stages with no model training or FP-RIR rendering.

Seed43 remains deferred under the seed42-first decision; its epoch8 checkpoint is preserved. No evidence yet establishes seed robustness.

## Recovered backend integration

The previously pending integration actually completed: one reproducibly selected query from each of19Val scenes. Rebuilt support candidate IDs, score features, GEOQ gains and chosen pose all satisfy the replay assertions. See [safe receipt](rebuilt_support_integration.json). Executed source SHA256 is in that receipt; source committed as b75215d.

The first smoke run failed because the historical extractor expects original perspective-depth vectors, whereas the draft supplied angular rays. Restoring the original input contract fixed replay without changing tolerances. Failure artifacts remain local. This checks implementation consistency, not improved localization or isolated causality. One timing repetition is insufficient for performance conclusions; timings are omitted from the published receipt. Raw-input encoding, adaptation, static geometry preparation and FP-RIR creation are excluded; this is not end-to-end latency.

## Remaining work

1. Complete A3 evaluations and validate native aggregates, calibration, learned-vs-uniform router and paired comparisons against A2/B2+GEOQ. Uniform inference weights are not a separately trained fixed-router control.
2. Finish repeated cost measurements with scope and shared-host limits, including A3 training accounting.
3. Update requested metric tables, figures, integrated interpretation and provenance. The study remains incomplete; reports are not a stopping criterion.

Routine long-running reviews should be hourly as requested, with failures handled when detected. All existing models, original results and unrelated modifications are preserved.

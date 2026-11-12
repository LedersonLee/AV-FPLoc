# A3 prerequisite evaluation gate — 2026-11-12

Track A; Exploratory; completed integrity validation, not completion of the benchmark.

A0/A1/A2 seed42 completed-run checkpoints and predictions passed a fresh CPU verification against the original input manifest. All 19 Val scenes / 9,965 query IDs align for each expert and each 20cm/10cm grid. Positive finite scales, array dimensions, label/prediction/checkpoint hashes, best epoch identity, per-query ray MAE, every saved metric aggregate and best-checkpoint validation MAE were checked. Tolerances are explicit in `validate_expert_gate.py`; validation does not rerun dense pose scoring. The gate records hashes and metrics in [expert_gate_seed42.json](expert_gate_seed42.json).

The completed-run receipt is not treated as proof of executed epoch count (the known pre-break counter issue remains documented). Gate validation checks best-checkpoint identity and measured evaluation, not the faulty counter.

Fault injection rejected mismatched checkpoint hash, missing query count and wrong seed. No original evidence was edited. Training/model source remains unchanged. Reproduce with `python3 scripts/research/track_a/avloc_20261111/validate_expert_gate.py --study <local-study-root> --seed 42 --out <new-gate.json>`; output creation is exclusive.

A1 poor localization is retained; no minimum accuracy threshold is used to admit experts. These checks permit the planned A3 experiment but do not establish independent calibration, seed stability or causality. Remaining A2 calibration work continues separately. A3 will train after the current seed42 interaction-disabled control, using the original full protocol and this evidence-backed gate. It has not started as of this report. A3 uses fresh common initialization, not warm-started expert checkpoints. Test17 remains unused.

Training source: `62c58da20ed26ba95863638bbfd4926829999eec`. Gate validator source SHA256 is recorded in the attached JSON; its Git commit is the commit introducing this report and validator (not an experiment training SHA).

## A3 scheduling

The seed42 A3 queue is now waiting on the exact live interaction-disabled container ID. Scheduler code: `0b3f8e74e9db3065b14e075d1757c88618b805df`; training code remains `62c58da20ed26ba95863638bbfd4926829999eec`. It uses a hash-checked immutable Python snapshot, pinned image and existing expert-gate hash. After Docker exit0 and the control training-complete receipt, it revalidates the three expert artifacts before launching A3. Nonzero predecessor exit, changed evidence/source, or existing A3 output stops execution; there is no retry or overwrite. It waits on a completion event rather than repeatedly polling. Seed43 is excluded. A3 training is not yet complete or evaluated.

Local plan/log: `experiments/avloc_20261111/a3_priority_queue_v1/plan.json` and `a3_priority_queue_v1.log`. Reproduce scheduling with `queue_a3_after_control.py --workspace <workspace> --study <study> --out <new-queue-directory> --await-container <live-control-container>`; initial scheduling requires a live predecessor. Syntax compilation passed; the actual launch validated evidence, pinned image, source closure and predecessor identity before entering its wait. The eventual A3 launch path remains to be exercised on predecessor completion. The full protocol retains max100 epochs/patience20, batch8/effective32 and best clean Val19 ray-MAE selection.

## Prepared A3 evaluation

Evaluator source `564d2cb` now accepts completed A3 and evaluates its fused F head on 20cm and 10cm grids, with native ray diagnostics and temperature calibration. After those stages it checks saved Router weights against fused mean/scale, measures per-direction expert weight/error relationships, and exports a uniform-weight inference control from the exact same learned expert heads for both grids. This control isolates the inference weight intervention conditional on trained heads; it does not isolate the effect of training with a learned versus fixed router. No independently trained fixed-router result is claimed.

Verification: the refactored A2 stage plan exactly matches its previously executed frozen plan; A3 plan has eleven stages, unique output paths and satisfied prediction dependencies. Six Router analytic/corruption-rejection tests pass. Actual A3 inference and full-cohort evaluation remain pending trained checkpoints. Reproduce after A3 completion with `evaluate_completed_expert.py --workspace <workspace> --study <study> --out <new-evaluation-queue> --arm A3 --seed 42`. Existing active training/evaluation snapshots are unaffected.

## Control evaluation scheduling

Evaluator `ff08754` adds optional `--await-container` for an initially live, exact training container. It validates the requested run path and pinned image, waits for exit0, then applies the existing completed-run/source-hash/new-output checks. No retry or restart. The A2-disabled seed42 evaluator is scheduled this way, so its full9965-query prediction, both grid evaluations, ray diagnostics and calibration begin after natural training completion. This can overlap A3 training; its elapsed time must not be reported as isolated inference cost. Current status is waiting, not evaluated. Local log: `experiments/avloc_20261111/a2_disabled_evaluation_queue_v1.log`. Initial predecessor validation and wait were exercised; later evaluation remains pending. The30-minute human-facing monitoring cadence is unchanged.

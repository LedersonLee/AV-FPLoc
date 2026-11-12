# A3 prerequisite evaluation gate — 2026-11-12

Track A; Exploratory; completed integrity validation, not completion of the benchmark.

A0/A1/A2 seed42 completed-run checkpoints and predictions passed a fresh CPU verification against the original input manifest. All 19 Val scenes / 9,965 query IDs align for each expert and each 20cm/10cm grid. Positive finite scales, array dimensions, label/prediction/checkpoint hashes, best epoch identity, per-query ray MAE, every saved metric aggregate and best-checkpoint validation MAE were checked. Tolerances are explicit in `validate_expert_gate.py`; validation does not rerun dense pose scoring. The gate records hashes and metrics in [expert_gate_seed42.json](expert_gate_seed42.json).

The completed-run receipt is not treated as proof of executed epoch count (the known pre-break counter issue remains documented). Gate validation checks best-checkpoint identity and measured evaluation, not the faulty counter.

Fault injection rejected mismatched checkpoint hash, missing query count and wrong seed. No original evidence was edited. Training/model source remains unchanged. Reproduce with `python3 scripts/research/track_a/avloc_20261111/validate_expert_gate.py --study <local-study-root> --seed 42 --out <new-gate.json>`; output creation is exclusive.

A1 poor localization is retained; no minimum accuracy threshold is used to admit experts. These checks permit the planned A3 experiment but do not establish independent calibration, seed stability or causality. Remaining A2 calibration work continues separately. A3 will train after the current seed42 interaction-disabled control, using the original full protocol and this evidence-backed gate. It has not started as of this report. A3 uses fresh common initialization, not warm-started expert checkpoints. Test17 remains unused.

Training source: `62c58da20ed26ba95863638bbfd4926829999eec`. Gate validator source SHA256 is recorded in the attached JSON; its Git commit is the commit introducing this report and validator (not an experiment training SHA).

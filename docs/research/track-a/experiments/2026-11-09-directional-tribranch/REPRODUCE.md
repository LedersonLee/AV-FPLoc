# Reproduction and artifact map

Workstation root `/home/jeongeonlee/track1`; study `experiments/directional_tribranch_20261109`. Read USER_REQUEST.md, PROTOCOL.json and FOLLOWUP_PROTOCOL.md first. The original textual cue decision is retained at the start of A_FEASIBILITY.md. Corrections are in CORRECTIONS.md.

```bash
cd /home/jeongeonlee/track1
P=experiments/directional_tribranch_20261109
bash "$P/run.sh" verify_and_visual.py
python3 "$P/analyze.py"
python3 "$P/diagnostics.py"
python3 "$P/write_reports.py"
python3 "$P/plot_results.py"
python3 "$P/final_verify.py"
```

These recompute verification/analysis/report artifacts; save a copy first if preserving byte-exact timestamps/measurements. They do not overwrite old studies or fitted checkpoints. Environment versions are in environment.json; Docker invocation is in run.sh (network disabled, old workspace read-only, study read-write). Docker image ID is in final verification. CPU host requires numpy/matplotlib for reports.

Training entry points, for a **fresh copied/versioned output directory**, in order:

```bash
bash "$P/run.sh" prepare.py
bash "$P/run.sh" stage_a.py
bash "$P/run.sh" a_followup.py
bash "$P/run_b_matrix.sh"
bash "$P/run.sh" nested_followup.py
```

Do not blindly rerun Stage A in the existing directory: probes resume but conditional trees are refit. B and nested complete receipts guard against redundant fitting. For a fresh study, copy scripts, PROTOCOL.json, FOLLOWUP_PROTOCOL.md and USER_REQUEST.md into a new directory and set P to it; run.sh resolves its own destination. Keep original results immutable. Original dependency paths/cache receipts must remain available.

`prepare.py` verifies source raw E/C/N receipts, schedule/yaw/IDs and exact encoder compatibility; failed original batch experiment remains in prepare_initial_batch_mismatch.log. Frozen acoustic_initial.pt is derived from the original checkpoint, not a new model. `stage_a/` holds24outer probes,12inner probes,24conditional trees and1repeat diagnostic probe. `stage_b/` holds66final neural models, loss curves, trained-layer deltas, fit/held lists and7held interventions. `nested_followup/` holds24inner learning-curve models and12final models (8explicit zero-update initializations). Total155runs with optimizer fitting plus8zero-update final states; different fit types are not equivalent compute units.

`verification.json` checks264neural forward replays (maxabs0), original GEOQ gains/choices and frozen visual full-bank poses. `visual_fullbank/` stores F2 pure-visual complete-bank results. `analysis.json` includes scene/seed metrics and bootstrap statistics; `A_followup.json` has sector reliability/rotation/repeat evidence; `diagnostics.json` has paired examples and inner curves. `MANIFEST.json` hashes all persistent study files except itself and final_verification.json, with model weights included. `final_verification.json` records counts, splits, source checkpoints, immutable old diff and manifest check.

GT-free deployment boundary is models.py::TriBranch.forward and stage_a.py::predict. Baseline gains must come from the original GT-free GEOQ inputs; coarse cue must come from observed acoustic features. labels/true pose/target rays are for losses and evaluation only. A scorer result is candidate-relative utility evidence, not a calibrated likelihood. Frozen candidate extraction and the original B2 weights are not replaced.

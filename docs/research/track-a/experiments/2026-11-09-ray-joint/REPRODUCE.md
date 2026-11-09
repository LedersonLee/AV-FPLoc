# Reproduction and provenance

This is a completed controlled study with historical local dependencies; it is not a clean-clone standalone benchmark. See source_inventory.json for exact scripts, execution_receipts.json for actual execution repository SHAs and model hashes, input_inventory.json for dependency hashes, and environment.json for package versions. The tracked source is `scripts/research/track_a/ray_joint_20261109/`. Local artifacts are `${WORKSPACE}/experiments/ray_joint_20261109/`; weights, raw observations, predictions and logs remain local.

Dependencies are the historical HSSD scaling, directional tri-branch and directional structure studies, their cached observations/geometry/visual scores and original encoder checkpoints. Exact input provenance is recorded in audio receipts and input hashes. The renderer and historical teacher are not retrained here. The repository can contain unrelated concurrent Track B commits; use source hashes as well as receipt SHAs. Two pre-existing runtime file modifications were preserved and were not part of this study's source changes.

`run.sh` starts the existing `avfploc:cu124-clean` container with no network, read-only workspace/dataset mounts, a writable study mount,4CPU/16GB and a per-process20%GPU allocation where Torch is used. It currently assumes the local workspace path; adapt mount paths for another host. Scripts use `/workspace` for read-only historical inputs and `/output` for this study. Never run in an original artifact folder when intending a fresh training reproduction: receipt-based resume intentionally preserves completed models. Copy protocols/scripts into a new empty study directory and resolve required dependency mounts first.

Executed order (these are reproduction commands, not an instruction to rerun during documentation work):

```bash
bash experiments/ray_joint_20261109/run.sh oracle_capacity.py
bash experiments/ray_joint_20261109/run.sh prepare_audio.py
bash experiments/ray_joint_20261109/run.sh train_acoustic.py
bash experiments/ray_joint_20261109/run.sh train_acoustic.py --controls
bash experiments/ray_joint_20261109/run.sh train_joint.py
bash experiments/ray_joint_20261109/run.sh late_fusion.py
bash experiments/ray_joint_20261109/run.sh evaluate_candidates.py
bash experiments/ray_joint_20261109/run.sh evaluate_fullbank.py
bash experiments/ray_joint_20261109/run.sh calibrate.py
bash experiments/ray_joint_20261109/run.sh candidate_conditioned.py
bash experiments/ray_joint_20261109/run.sh conditioned_fullbank_shared.py
bash experiments/ray_joint_20261109/run.sh train_joint_followup.py
bash experiments/ray_joint_20261109/run.sh evaluate_candidates.py --followup
bash experiments/ray_joint_20261109/run.sh transfer_models.py
bash experiments/ray_joint_20261109/run.sh evaluate_transfer.py
bash experiments/ray_joint_20261109/run.sh fov_late.py
bash experiments/ray_joint_20261109/run.sh direction_permutation.py
bash experiments/ray_joint_20261109/run.sh verify_replay.py
bash experiments/ray_joint_20261109/run.sh verify_matching.py
bash experiments/ray_joint_20261109/run.sh verify_calibration.py
python3 experiments/ray_joint_20261109/verify_inputs.py
bash experiments/ray_joint_20261109/run.sh case_diagnostics.py
python3 experiments/ray_joint_20261109/analyze.py
python3 experiments/ray_joint_20261109/analyze_extensions.py
python3 experiments/ray_joint_20261109/analyze_remaining.py
# Once both full-bank output aggregates exist:
python3 experiments/ray_joint_20261109/analyze_fullbank.py
python3 experiments/ray_joint_20261109/figures.py
python3 experiments/ray_joint_20261109/build_report.py
python3 experiments/ray_joint_20261109/capture_provenance.py
```

The earlier followup_analysis.json was generated inline; analyze_remaining.py now reproduces those mean results from candidate_followup_evaluation.json and preserves a standalone paired summary in remaining_analysis.json. Full-bank aggregates contain2880primary and1728conditioned records; analyze_fullbank.py verifies completeness and exact uniform-metric agreement across the two implementations. Inference replay verifies228checkpoints/1542arrays exactly; it does not verify bitwise retraining across hardware/software environments. Interrupted determinism/short-RIR/donor diagnostic results remain in local logs and diagnostics directories. No paper PDFs or full extracted manuscript text are redistributed.

# Reproduction and artifact provenance

This study is Track A, not the independent Track B implementation. Evidence remains Exploratory because New12/Transfer11 have historical teacher and research exposure. Test observations/results are not inputs.

## Locations

- Repository: `/home/jeongeonlee/track1/AV-FPLoc`.
- Source: `scripts/research/track_a/directional_structure_20261109/`.
- Artifact root: `/home/jeongeonlee/track1/experiments/directional_structure_20261109/`.
- Prior Stage A: sibling `directional_tribranch_20261109/`; original manifest and fitted models preserved.
- Cached scene banks: sibling `hssd_scaling_20261108_1332/inputs/`.
- Historical repeat features: sibling `pairwise_acoustic_evidence_20261107_2100/repeat_features/`.
- Dataset: `/file2/jeongeon/AV-FPLoc/datasets/HSSD`; structural targets also under `/ssd1/AV-FPLoc/datasets/HSSD/UnLoc/structural_targets/`.

`PROTOCOL.json` source_commit is the initial repository base, not the commit of every later experiment. Actual fit code commits are in prediction/integration fit receipts, uncertainty receipts and transfer freeze. Source history includes the commits introducing symmetric/oracle and feature ablations. Analysis/replay source history begins at `0e3099e`; final source inventory records the committed file hashes. The final Docs commit is reported separately and is not inserted as an experiment execution SHA.

## Environment

The checked-in run.sh uses Docker `avfploc:cu124-clean`, network disabled, 4 CPUs, 16 GB RAM and GPU access. It mounts workspace/data read-only and only its external study directory writable. It uses workspace AV-FPLoc source plus the historical deps directory. Run `docker image inspect avfploc:cu124-clean` to resolve the local image identity. Analysis/figures/report scripts use host Python with NumPy/SciPy/Matplotlib. Exact package versions and GPU are recorded in environment.json.

This is not a self-contained clean-clone reproduction: licensed scene assets, original checkpoints and external source/cache files are required. Do not overwrite the existing artifact directory when reproducing fits. Copy the study scripts and protocols to a new external study directory first, and retain the read-only prior inputs. No source folders or model weights need to be uploaded.

## Execution order

From the workspace root, with an external study directory containing these scripts:

```bash
bash experiments/directional_structure_20261109/run.sh audit_prior.py
bash experiments/directional_structure_20261109/run.sh prepare_signal.py
bash experiments/directional_structure_20261109/run.sh prediction.py
bash experiments/directional_structure_20261109/run.sh prepare_candidates.py
bash experiments/directional_structure_20261109/run.sh discrimination.py
bash experiments/directional_structure_20261109/run.sh uncertainty.py
bash experiments/directional_structure_20261109/run.sh fullbank.py
bash experiments/directional_structure_20261109/run.sh integration.py
bash experiments/directional_structure_20261109/run.sh repeat_diagnostic.py
bash experiments/directional_structure_20261109/run.sh prepare_candidates.py --transfer
bash experiments/directional_structure_20261109/run.sh transfer.py
bash experiments/directional_structure_20261109/run.sh symmetric_control.py
bash experiments/directional_structure_20261109/run.sh oracle_diagnostic.py
bash experiments/directional_structure_20261109/run.sh feature_ablation.py
python3 experiments/directional_structure_20261109/analyze.py
python3 experiments/directional_structure_20261109/figures.py
python3 experiments/directional_structure_20261109/report.py
```

D/E were triggered by the C results; read INTEGRATION_PROTOCOL.md. Transfer was fixed after New12 and before new-method Transfer11 scoring; read TRANSFER_PROTOCOL.md. The commands above document the sequence and are not authorization to repeat training. Some scripts overwrite outputs or reuse existing checkpoints: use a fresh artifact directory for a genuine rerun.

## Verification without training

```bash
bash experiments/directional_structure_20261109/run.sh verify_models.py
```

This loads all73new checkpoint files, checks receipt hashes and disjoint scene splits, replays held predictions and selected metrics/choices, and checks the body-left convention. The recorded4,287 numerical comparisons pass with maximum6.12e-17, from trigonometry. Original24probes/96predictions are audited separately. Preserve the exact cardinal-first prior reduction order: algebraically equivalent float32 means can cross a trained tree threshold. See AUDIT.md.

Models live in prediction/, uncertainty/, integration/, symmetric/, feature_ablation/ and transfer/ under the local artifact root. Large query predictions, raw fields, logs and weights stay local. Reports export only compact aggregate evidence and figures.

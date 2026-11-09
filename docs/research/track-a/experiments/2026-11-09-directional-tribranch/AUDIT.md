# Implementation and asset audit — 2026-11-09

Audit/code and checkpoint inspection preceded fitting. This written receipt consolidates that inspection and subsequent exact replay checks. Repository `/home/jeongeonlee/track1/AV-FPLoc`, branch `refactor/hssd-two-branch-fusion`, starting HEAD `19c1bd02f6edc6758ee9b18675957a2d30b3aeed`. Original dirty diff is preserved in `original_working_tree.patch`; no repository code or old experiment has been intentionally changed. New artifacts live only in this study. No external upload.

## Authoritative implementations and baseline sources

Paths below are relative to `/home/jeongeonlee/track1`.

- `AV-FPLoc/src/avfploc/acoustic/model.py`: actual E/C/N encoder. Frozen energy/coherence/envelope trunks and stream heads yield three128 vectors; concatenated384 -> original Linear384→128 + LayerNorm128 -> normalized128. F1/F2 update this original projection on both observed and reference features, not just a newly invented fusion layer. Convolutional trunks remain frozen.
- `AV-FPLoc/src/avfploc/geometry/rays.py`: visual40 columns ->11 angular rays, -50°..50°. Interpolate depth and divide by cosine; uncertainty is interpolated without cosine division.
- `AV-FPLoc/src/avfploc/preprocessing/hssd/hssd_furnished_2dfp_rays.py` and `hssd_unloc_dataset.py`: structural map stepping and world/raster contract. New `prepare.py::rays` generalizes the same half-pixel stepping to36 horizontal bearings.
- `experiments/mbt_remaining_suite_20261106/runs/cap020/B2_seed{42,43}_full_epoch8.pt`: original B2, eight epochs, oldTrain12. `data/visual_heads.pt` supplies the frozen visual depth decoder.
- `experiments/structural_evidence_20261107_1730/GEOQ_seed{42,43}/model.joblib` and `inference.py`: S26/G26/Q26 plus baseline-relative78=156 features, candidate utility gain. Seed42/43 thresholds0/.01. This is not a calibrated posterior.
- `experiments/hssd_scaling_20261108_1332/inputs` and `candidates_seed{42,43}`: approved cached inputs and unchanged Support-NMS Top20. Historical B2 output replay provenance is inherited; this study newly replays original GEOQ predictions/choices exactly on all12×128×2 records, plus pure-visual full-bank argmax exactly.

Acoustic checkpoint: `/file2/jeongeon/AV-FPLoc/datasets/HSSD/AcousticContrastive/pilots/scene_subset_v1/input_ablation_v1/runs/all/best.pt`, SHA256 `f7ba7267c66490a034338d19867049e7392fef6795a6b947dfbbdc02f57d27b8`. Visual upstream checkpoint SHA256 `b9626c0f8b5411f365d6151e7a760b8e79fffe5e6ee2abced6caaa7705a1c36f`. Actual B2/GEOQ source hashes are in `verification.json`; new checkpoint hashes and source inventories are in `MANIFEST.json` and per-run receipts.

## Inputs, geometry, and labels

Observed visual[128,40,128], mu/scale[128,40]; observed preprojection[128,384], embedding[128,128]; candidate acoustic preprojection[128,20,384]; FP rays[128,20,36]. Acoustic raw probe uses channel-synchronized E/N coarse flattened-bin averages plus C, optionally the embedding. Flattened-bin averages and learned CNN tokens are not physical directions or calibrated DoA.

Six microphones on a horizontal5cm ring, centered active source,48kHz. Logical yaw forward in worldXZ=(cosθ,sinθ); left=-90°, right=+90°, up=+Y. Canonical raw reference bearing uses (-yaw)%360 and synchronized microphone rolls. A +1 channel roll corresponds to the +60° target-bearing rotation in the diagnostic. Ten-degree candidate yaw bins are not sensor angular resolution.

Observed RIR uses furnished native geometry; FP reference uses canonical2D-map extrusion, not stage-only mesh. The inherited non-Test metadata audit checks diffraction on, materials/transmission off, centered source and receiver geometry. Semantic region heights are assumed;5 older scenes use approved mean indoor height. These are not realistic unknown-material/height guarantees. No new RIR was generated.

FP raster spacing1cm, DESDF10cm, candidate grid20cm and36yaw bins. A targets: log1p of median five rays per front/left/back/right sector, range cap10m; near<3m. Raster exit is unknown, not a wall; current selected query rays have no unknown exits. A black pixel means blocked structural footprint, not semantic wall/door identity. Query target uses continuous pose map rays; candidate geometry uses cached grid DESDF, leaving discretization and 2D/3D mismatch. Only training labels and diagnostics use true pose/target geometry. GT is absent from `TriBranch.forward` and `stage_a.predict` feature arguments.

Success definitions follow existing code: position≤1m; joint position≤1m AND circular yaw≤30°; fine is position≤0.1m (no yaw requirement). Geometry-label accuracy, candidate-oracle coverage, ambiguous-pair accuracy and realized localization are separate quantities.

## Readiness, leakage and scope

The prior135 non-Test sample/asset audit is inherited, not claimed as a fresh exhaustive re-audit:168 source configurations;135 approved Train116/Val19 sampled assets;55 dense banks before conservative source-family exclusions,54 after. The prior stage-mesh identity and source-prefix grouping audits remain applicable. For this study all23 selected Train scenes had source receipts/raw feature hashes, schedule hashes, query IDs, target poses, candidate cache hashes and observed/reference encoder compatibility revalidated. The corrected preparation reproduces all23 observed and both-seed candidate embeddings exactly (maxabs0). A batch-shape discrepancy was caught before fitting and fixed by reproducing original256-cell encoder blocks; original failure log and correction remain.

New12 = prior medium24 minus oldTrain12. Three fixed outer scene folds,8fit/4held,128queries/scene,2seeds. Original acoustic/B2/GEOQ Train12 exclude New12; visual teacher trained on Train116 and monitored Val19. New12 and prepared transfer11 were previously researched. These are exploratory head-held/teacher-exposed results, not pristine full-pipeline generalization. Split IDs are in `PROTOCOL.json`; excluded Test families are inherited and rechecked administratively. No Test observations/results used. Transfer11 inputs were prepared, but no new model performance was evaluated there because no candidate passed the primary screen.

## Budget and execution

RTX4090; cached preparation at4CPU/16GB per isolated Docker runner, training likewise. Initial estimate: short cached preparation and small A probes,66 fixed-epoch B fits; no full-bank RIR regeneration. Actual preparation task sum 340.3s across23 scenes, excluding process startup and the preserved failed attempt. B66 fit/evaluate/save task sum 104.3s. These are scoped task times, not end-to-end GPU-hour accounting. Exact replay plus full-bank visual preservation took 4.91s, peak allocated CUDA 1.56GiB for that verification process only. Existing unrelated jobs were preserved. See `environment.json`, `run.sh`, logs and `FOLLOWUP_PROTOCOL.md`.

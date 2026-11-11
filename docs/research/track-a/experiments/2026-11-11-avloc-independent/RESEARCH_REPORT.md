# AVLoc independent reimplementation and matched HSSD benchmark

Study `avloc_20261111`, Primary Track A, started2026-11-11. **Active, incomplete**. Evidence level Exploratory.

Objective: reproduce the paper-based architecture as an explicit6ch HSSD adaptation, train on the full eligible cohort, and complete all requested matched baseline, calibration, observability, interaction and cost comparisons. No superiority of Track A is assumed.

## Current evidence

Paper Method3/AppB,C,G and the three supplied reviews were read. The implementation reproduces the specified token/expert depths and routing formula, with deviations explicitly listed. Synthetic full-resolution architecture checks passed for A0/A1/A2/A3 and the interaction-disabled control. Unimodal predictions are invariant to changes in the other input; detached routing gradients and absent vision weights outside FoV were checked. Independent SciPy tests verify the chirp and convolution, and reflection is involutive. These are code checks, not learned localization results.

Full Train/Val asset validation completed: all68,859 scheduled queries in112 Train and19 Val scenes passed after family exclusions. Both image decoding and RIR finite/shape/checksum/pose checks are performed, rather than one waveform per scene. Test metadata was used only to exclude source families; no Test observation evaluation. Existing checkpoints/results and Track B are unchanged.

## Hypotheses and competing explanations

- Audio36-ray predictions may localize by geometry; alternatively direct-path or sensor-layout shortcuts may produce apparent accuracy. Diagnose ray error, fine-pose success conditions and geometry uniqueness.
- Joint attention may add information beyond independent V/A fusion; alternatively complementarity alone or extra capacity explains gains. Use independent fusion and matched-parameter interaction-disabled controls with compute reported.
- Sharp posterior may correlate with correctness; alternatively scale/temperature causes overconfidence. Separate entropy/margins, GT mass, NLL/ECE and held-development temperature calibration.
- FP-RIR matching may justify its storage/render cost; alternatively ray matching achieves similar accuracy without that bank. Measure both accuracy and deployment/training cost definitions.

## Results and failures so far

A0 seed42 full-cohort training has been launched; no completed trained A0–A3 benchmark table exists yet. Baseline/ablation numeric rows are pending, not zero or failed methods. The alternative SSD-root audit found no observation schedules (0 valid queries) and is unusable; it is preserved as a failed asset-location attempt. The authoritative network-backed HSSD audit completed with no invalid queries. SSD RGB availability does not imply the entire RIR dataset is mirrored.

## Remaining completion gates

1. Completed: whole-cohort asset audit/cache, exact geometry bank checks, real-data backward smoke and deterministic optimizer-resume integration checks.
2. Training/augmentation/checkpoint pipeline implemented; train A0/A1/A2 sufficiently on all valid queries, validate before large A3 router optimization.
3. Train A3 and required independent/disabled/router controls with paired initialization and multiple seeds.
4. Evaluate UnLoc,cosine,fixed fusion,B2,B2+GEOQ,A0–A3 on the identical cohort/grid; report all requested pose/ray/coverage/repair-harm/scene metrics, and separate10cm grid.
5. Complete posterior comparison figures, acoustic-only observability, calibration/correlation and cost analysis, with failure cases and paired scene uncertainty.
6. Update this report with actual numbers and all required deliverables. Keep the active goal until requirement-by-requirement audit proves completion.

See [method](METHOD_AT_A_GLANCE.md), [paper/implementation audit](IMPLEMENTATION_AUDIT.md), and [assumptions](implementation_assumptions.md). Reproduction currently covers only the initial code tests; full experiment reproduction must be added with checkpoint/data provenance after runs.

## Full-cohort training launch and engineering evidence

Train112 has58,894 queries; Val19 has9,965. An extra Train scene sharing a Val source family is excluded in addition to the requested three Test-family exclusions. Scene labels are finite/bounded, query IDs unique per scene, and all RGB/RIR inputs were audited. All data preparation jobs are terminal and successful.20cm and10cm Val banks are prepared independently of acoustic observations.

Real full-resolution backward/optimizer checks passed for A0/A1/A2/A3/interaction-disabled. A mixed-precision indexed assignment failed in the first smoke run, was corrected, and all five arms then passed; the original failure log remains local. Common initialization, ImageNet conv1 equality and perspective-column ray seeding tests passed. A four-query disposable A2 integration test exercises the real DataLoader, ESC/Gaussian augmentation, FP32 validation, optimizer checkpoint and exact resume. These checks establish implementation behavior, not learned accuracy.

A0 seed42 starts with full clean Val19 evaluation, then full-cohort training for at most100epochs and patience20 by minimum clean query-mean ray MAE. No model selection uses Test17. The source commit and run protocol identify the actual executable implementation; initial tests used base0480eb7 plus the now-committed source additions. A3 remains gated on completed A0/A1/A2 validation.

See [training state test](training_state_test.json), [initialization test](initialization_test.json), [backward smoke](runtime_smoke_v2.json), [geometry test](geometry_test_v2.json), and [asset audit](asset_audit_summary.json).

Production training source commit: `62c58da20ed26ba95863638bbfd4926829999eec`. Local run root: `$AVLOC_STUDY/runs/A0_seed42`; `$AVLOC_STUDY` is the dedicated SSD study directory, not a Git asset.

## Current run reproduction

Runtime image: `avfploc:cu124-clean` (PyTorch2.4.1+cu124, torchvision0.19.1+cu124). Mount the committed source at `/source`, audited SSD study at `/study`, and ImageNet weights at `/weights`; use GPU,4 data workers,8 CPU limit,20GiB memory,2GiB shared memory. Set `CUBLAS_WORKSPACE_CONFIG=:4096:8`, disable bytecode writes. Command inside the runtime:

```bash
python /source/train.py --data /study/data_v1 --noise /study/esc50_v1 \
  --weights /weights/resnet50-11ad3fa6.pth --out /study/runs/A0_seed42 \
  --arm A0 --seed 42 --batch 8 --workers 4 \
  --code-sha 62c58da20ed26ba95863638bbfd4926829999eec
```

Use the full training source SHA stated above for `--code-sha`; the local `protocol.json` records it together with source/data/noise/ImageNet hashes. A new run requires a new output directory. `--resume` is only for an incomplete run with matching protocol and `latest.pt`; do not restart a live container. No trained-checkpoint hash is claimed before a checkpoint is created. Dataset/cache access is required; this is not self-contained public-data reproduction.

## Geometry-only matching implementation

Source `8eaaf247ae20ac39e132e2192bafee3f8b18ae4c` adds Eq2 summed Laplace log densities over36 rays (11 for V), positive logical-yaw bank permutations, stable GT-free joint-distance/yaw NMS, and separate diagnostic functions. Independent SciPy log-density agreement has maximum absolute error1.14e−13. All19 Val scenes × two grids pass known-profile maximum consistency, including circular yaw and symmetric ties. [Numeric tests](matching_test.json), [real bank tests](real_bank_matching_test.json), [execution provenance](matching_provenance.json). These are numerical tests, not a trained localization claim.

`predict.py` requires completed training, selects the recorded best clean-Val checkpoint, validates architecture/data hashes, and extracts all9,965 clean Val ray distributions in float32. `evaluate_fields.py` verifies prediction/bank hashes and query IDs, then evaluates either20cm or10cm. Both CLI paths compile; end-to-end learned inference awaits completed training.

Reported Top5/20 will distinguish raw score ranks from1m/30° NMS candidates. GT affects only subsequent error/coverage/region diagnostics. Eq2 normalization uses a uniform discrete pose prior and conditional ray-independence assumption, not demonstrated calibration. Record entropy, raw and distinct-peak gaps, nearest-grid NLL, GT joint-region mass/NLL, and predicted-pose neighborhood confidence against joint correctness. Regions outside grid support are explicitly flagged, not silently assigned zero NLL. Fifteen-bin ECE and Brier are descriptive; scene-held temperature fitting and independent-fold correlation analysis are still pending.

Reproduction after training completes (inside the documented runtime):

```bash
python /source/predict.py --run /study/runs/A0_seed42 --data /study/data_v1 --out /study/predictions/A0_seed42
python /source/evaluate_fields.py --predictions /study/predictions/A0_seed42 --data /study/data_v1 --banks /study/banks_v1 --out /study/evaluation/A0_seed42_20cm --grid-cm 20 --head V
```

Use a separate output for10cm and the appropriate A/AV/F head for subsequent arms. Do not treat partial output without a complete manifest/summary as a completed evaluation. Matching never loads an acoustic reference bank.

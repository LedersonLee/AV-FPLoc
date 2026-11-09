# OOF full-field candidates and new discrimination — preparation

Date2026-11-09. Track B. Status: 18 scorer fits, three candidate caches and 12 discriminator fits completed; mixed/negative geometry result. See [completed results](completed_results.md). Preparation narrative below is retained as the pre-execution plan. Evidence: Exploratory. This is a newly trained independent-F1 backend control, not a substitute for proving the proposed joint model or an independently validated Test result.

## Hypothesis and leakage boundary

The strong independent full-bank scorer has substantially higher diverse Top20 coverage than MAP success. A new geometry/observation discriminator may reduce that gap, but candidates produced by a scorer trained on the same scenes would confound discrimination with memorized proposals. Therefore every locally trained visual/acoustic/FP component must be fit only on the fold's eight Train scenes before producing candidates for the other eight. No GT pose is injected into Top20 and GT is allowed only for training targets/evaluation. Existing B2/GEOQ candidates, weights or historical HSSD visual heads are forbidden.

The frozen upstream RGB encoder/features are reused and have separately audited provenance; uncertain upstream asset-family exposure is not ruled out. Hyperparameters were developed on the existing reused development cohort. Thus locally held-out proposals do not create pristine full-pipeline independence or erase method-selection exposure.

## Frozen split and training protocol

Partition only the original16 Train scenes into two8/8 folds. Sort known source-family IDs by SHA256 and greedily balance counts without viewing observations, GT or outcomes. Keep each known family within one fold. Original six Val scenes are not in either training-fold partition; their observations are not loaded by these fold runs. Geometry-near-duplicate completeness beyond the known family IDs is not claimed.

Three seeds42/43/44 per fold:

1. Fresh foundation visual adapter,1500 auxiliary-only updates. Frozen upstream encoder, no HSSD head initialization.
2. Same fold's archived auxiliary checkpoint, fresh AdamW,1000 full-bank updates, visual auxiliary1000 and raw-geometry FP anchor1000; skip1500 sample draws to match established continuation order.
3. Fresh independent acoustic/FP model,1500 auxiliary then1000 dense updates, auxiliary10, following the previously established acoustic baseline.

All batch8, lr3e-4, weight decay1e-4, scheduled final checkpoints only, no selecting epochs based on held-out-fold evaluations. This is18 planned fits (six pretraining, six visual continuation, six acoustic). Fold losses never consume held-out-fold observations/labels. Training script still computes held-out diagnostics at scheduled points; these must not be used to retune this frozen fold protocol.

The fold validator rejects original-Val inclusion, duplicated/empty lists, incomplete partitions, source-hash mismatch and known-family crossing. A pretrained continuation is rejected if its recorded effective scene split differs. These are necessary safeguards, not proof that arbitrary external pretrained weights are clean.

## Verification and current execution

36 tests passed, including fold source/family/partition rejection cases. Actual fold0 smoke (one auxiliary plus one dense update) exited0 and verified effective Train8/held-out8 loading with aligned foundation features. Its poor accuracy is expected for a runtime smoke and is not a scientific model comparison. Source SHA `0aedb2ad9cb4690501c7f8ff0e648630241f983b`.

Full fold training launched in observation session11903, six sequential run directories across two folds and three stages, each containing three seeds. External root `/home/jeongeonlee/track1/experiments/track_b_oof_20261109`. Existing main matched interaction batch remains separately running. Completion requires all expected fit artifacts and subsequent OOF exclusion checks, not a log timestamp or pending protocol file.

## Candidate/discriminator plan and limitations

Freeze each fold scorer and combine its independently learned visual/acoustic fields with fixed .85/.15 weights. Extract actual raw20 and diverse20 from the full field; retain no-positive-candidate queries. Report both oracle coverages. Train a new geometry/observation discriminator on fold-held predictions, with candidate features built only from query encodings, candidate/map geometry and score-neighborhood statistics. Candidate GT distances/labels must never enter the feature builder. No classifier result can exceed its actual candidate coverage; if coverage fails, report that failure rather than injecting GT.

For final development evaluation, use the previously established full-Train independent scorer on the original six Val scenes. This introduces a known half-Train versus full-Train proposal-distribution difference, to be reported. It does not establish an M1-attention backend result. MAP, Support-NMS and discriminator must be compared on clearly identified candidate policies, with end-to-end latency and scene uncertainty. The classifier architecture/training and actual candidate generation are not yet implemented in this preparation report; they remain explicit work, not implied success.

## Reproduction

Docker avfploc:cu124-clean, RTX4090, source/input caches read-only, no network, non-root UID. Generate folds: `python -m track_b.scripts.make_oof_folds --cache <structure-cache> --out <new-fold-directory>`. Every `train_structure` call supplies `--scene-fold /results/folds_v1/fold_0.json` (or fold1). Visual calls use `--visual-cache /features`; acoustic calls omit it. Output directories are `fold{0,1}_{visual_pretrain,visual_joint,acoustic}_v1`. Visual joint adds `--pretrained-dir /results/fold0_visual_pretrain_v1 --skip-sampling-steps 1500 --aux-weight 1000 --fp-anchor-weight 1000 --aux-steps 0 --dense-steps 1000`; pretraining uses aux1500/dense0; acoustic uses aux1500/dense1000/weight10. Always use new output directories and preserve failures/checkpoints.

Docs and code branch `refactor/hssd-two-branch-fusion`; large weights/features/logs remain external. No Test observations are accessed. Overall research goal remains incomplete.

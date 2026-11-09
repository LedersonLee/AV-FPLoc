# Prior Stage A / current research audit

Status: prior audit and Phase A–E experiments completed; model replay and artifact preservation audits passed. Scientific stopping evidence is recorded in COMPLETION_AUDIT.md; Git synchronization is reported separately.

Prior source: ../directional_tribranch_20261109. All3,723 manifest entries were independently hashed and verified. audit_prior.py loads the24actual outer probe checkpoints, checks fitted parameter norms, and replays their observed-only held predictions. No previous model is retrained or overwritten.

The old target was log1p median of5rays around each front/left/back/right bearing, range cap10m, near<3m. Source targets are continuous-pose2D blocked-footprint ray labels, not identified acoustic reflectors. Original logical-world yaw increases toward+Z: body-left+90 maps to world offset−90; the requested body front/left/back/right order therefore remains[0,27,18,9] in the old36ray raster. No left/right label relabeling is silently introduced.

Old raw_ridge used552compressed E/C/N features, not raw waveform. embedding_ridge used128frozen features; combined ridge680; MLP same680inputs,64hidden,60epochs. Ridge alpha100, fit-scene normalization only. Three8fit/4held scene folds,12new scenes×128queries, two seeds. Deterministic ridge duplicates are not independent replications. Query GT enters targets/diagnostics only.

Old embedding-ridge near/far77.669%, prior61.995%; combined ridge77.067%. Ambiguous good/bad pairs: combined cue60.466%, donor50.884%; conditioned tree cue74.479% joint vs same-budget base74.707% and originalGEOQ75.977%. Thus the previous observability claim has saved quantitative evidence but not incremental localization evidence. The old pair mask used all visual-gap pairs; it lacked required spatial/geometry difference filters and a pure-visual-only proposal set. New C addresses those omissions.

Old repeated-render diagnostic includes8Observed and32FP poses across4historical acoustic-training scenes; not40independent observed poses. It is a limited render-variation check, not sensor reproducibility. Old uncertainty used a fixed logistic mapping, not calibrated prediction intervals. New B supplies metric/ordinal/balanced/AUROC/per-direction results and fit-only uncertainty in prediction_results.json and uncertainty_results.json. Current raw descriptor controls inspect actual48kHz six-channel arrays and centered5cm-ring metadata with shared-clock windows; no echo is assigned to a physical wall.

SameNew12 is reused for controlled comparability, with historical visual-teacher exposure and research reuse explicit. Test observations/results are never opened. Transfer11 remains exploratory, not independent. Existing TrackB code is outside this study and is not modified. Original dirty patch/status are preserved. See PROTOCOL.json for settings and RESEARCH_REPORT.md for completed mandatory figures and pureV evaluation. Saved new checkpoints73 were independently replayed on held scenes:4,287 numerical checks passed; maximum error6.12e-17 (body-frame trigonometry).

## Numerical replay correction

The first verifier implementation reconstructed the integration geometry prior via an all36-ray array followed by cardinal selection. Although algebraically equivalent, float32 layout/reduction order perturbed a tree split and changed one saved gain by0.01628. The verifier now uses the original cardinal-first coarse function and exact reduction order; all73checkpoint replays pass without relaxed tolerance or changing models/results. Preserve this operation order when reproducing trained trees.

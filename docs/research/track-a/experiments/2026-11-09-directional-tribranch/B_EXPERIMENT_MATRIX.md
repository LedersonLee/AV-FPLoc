# Stage B controlled pilot

All66fits completed:11arms ×3scene folds ×2seeds,12fixed epochs. Stage B was executed despite Stage A's unconfirmed localization benefit. No outer label epoch/threshold selection in this matrix.

|Arm|Total parameters|Trainable parameters|Active gradient tensors|Fits|
|---|---:|---:|---|---:|
|F0_concat|101,633|52,097|[22]|6|
|F0_gate|101,633|52,097|[22]|6|
|F0_attention|101,633|52,097|[22]|6|
|F0_gate_cue|101,633|52,097|[22]|6|
|F0_noaudio|101,633|52,097|[22]|6|
|F0_noFP|101,633|52,097|[22]|6|
|F1_concat|101,633|101,633|[26]|6|
|F1_gate|101,633|101,633|[26]|6|
|F2_audio|114,081|114,081|[30]|6|
|F2_self|114,081|114,081|[30]|6|
|F2_donor|114,081|114,081|[30]|6|

Counts include the frozen original acoustic projection; active tensors were measured after the first training epoch. Per-layer maximum changes and active names are saved in each `complete.json`. F0 has22 active tensors, F1 adds4 original projection tensors, F2 adds4 visual-adapter tensors. These are tensor counts, not identical effective function complexity.

F0_concat / F0_gate / F0_attention share exactly the same16 V/A tokens, FP input, Q/K/V/O modules and576-input head. They differ in additive pooled, multiplicative gated, or softmax attention message computation. Concatenation includes candidate geometry and the same embeddings; an MLP can implicitly learn interactions. The comparison tests an explicit interaction operator, not mathematical absence of all dependence. F0_gate_cue replaces a redundant four-cardinal FP basis with acoustic-only coarse predictions; same parameters. Training cues are two-inner-scene crossfit, held cues from the outer fit.

F0_noaudio and F0_noFP are trained same-capacity concatenation controls, substituting query visual features into the removed new branch. Frozen GEOQ and candidate references still contain original audio/map information. These do not ablate all audio or all map information in the system.

F1_concat / F1_gate update the actual historical acoustic Linear384→128 and LayerNorm128 (49,536parameters), shared observed/reference. Trunks and stream pooling remain frozen. F2_audio adds a40×128 residual visual adapter conditioned on observed audio; F2_self conditions on global visual features; F2_donor uses same-scene donor audio. All use the original frozen depth head, scale unchanged. This is targeted partial joint optimization, not full RGB/RIR encoder retraining.

## GT-free scorer and loss

`TriBranch.forward(visual, mu, scale, observed_pre, reference_pre, fp_geometry, base_gain, coarse_prediction, condition)` accepts no label, target pose, scene ID or true query geometry. V has11physical FoV ray tokens +1global token; A hasquery/reference/product/abs-difference tokens with no physical direction interpretation. FP is the known hypothesized reference, not an observed third sensor. No token is invented for an unobserved visual direction.

Every initial model exactly equals frozen GEOQ, including seed-specific threshold. Learn a raw head bounded±.5, subtract its candidate0 value, then add that residual (bound±1) to GEOQ. Candidate0 is always zero. Thus this is strong-baseline residual hypothesis ranking, not a standalone posterior model or a new full-bank system. No softmax/NLL calibration claim.

Objective = candidate-relative joint+.5fine utility MSE +.01 normalized structural-depth error +.01 normalized visual drift penalty. The depth terms are constant in F0/F1. Query batches32 within scene; each fit scene128queries gives equal scene weight. AdamW headlr.001, acoustic/adapterlr.0001, wd.01, clip1. All arms get the same12epochs and supervision. No target-dependent candidate filtering.

F2 donor training rolls17 within a32query same-scene batch; held intervention rolls1/17/31 within128queries. Recipient FP references stay fixed. F2_donor also retains its architectural roll17 at inference. New-branch donor tests leave cached original GEOQ unchanged. `disabled` switches the message operator to concatenation under learned weights; this is an OOD sensitivity test, not a trained causal control.

Each held model receives matched,3donors,noaudio,noFP,disabled interventions. Save query IDs, gains, residuals, adapted depth, choices, all final checkpoints, losses and actual layer changes. Subsequent `verify_and_visual.py` reproduces264held scene/model forward outputs with maxabs0 and compares F2 visual localization against the complete bank.

## Bounded optimization followup

See `FOLLOWUP_PROTOCOL.md`. After observing F1/F2 train/held deterioration, compare frozen gating and10x lower acoustic learning rate, selecting epoch0/1/3/6/12/24 on2inner scene folds with a fine-preservation guard.24inner learning curves plus12final fits/initializations;8final selections choose no update. This is exploratory adaptation of the research plan, not prespecified confirmatory testing. No F3 escalation or new full-bank tri-branch scorer was justified.

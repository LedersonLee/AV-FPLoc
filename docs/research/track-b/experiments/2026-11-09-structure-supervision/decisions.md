# Decisions and full protocol status

1. Expand tiny pilots to16Train/6Val ×64queries after full-bank profile. Keep official0.25m/36yaw rather than pilot0.5m/12. Select scenes by hash before training. Same112-family-filtered eligible training universe as initial audit; use no Test observations.
2. Explicitly test alternative direct structure prediction (protocol risk-A alternative2). Add auxiliary warmup to separate representation learning from pose retrieval. Do not rename this additive model M1.
3. Keep wrong-target and dense-only controls. Wrong-target loss is harmful, but auxiliary-vs-dense gain is small; reject a claim that correct-label vs corrupted-label difference proves superiority over dense training.
4. Save original failed/non-improving joint fits. Frozen/partial use fewer trainable weights. Seed repetition did not produce identical warmup weights, so load one archived checkpoint for each seed and reset optimizer identically across all three regimes. This is an additional comparison, not silent replacement.
5. Support-NMS uses fixed posterior radius and temperature; it does not improve MAP. Retain negative result. Do not fit a downstream discriminator to hide weak raw retrieval.
6. Replay raw audio and physical channel roll. Query dependence is supported; isolated physical echo semantics remain unverified. Temporal masking changes energy and representation distribution. Matched feature retraining and furnished/material interventions remain needed.
7. Isolate historical UnLoc as a strong reference with declared Train116/Val19 exposure. Do not use its trained weights to initialize the proposed Track B model. Future upstream visual pretraining must have a separately audited source and fresh Track B head.
8. Preserve external artifacts and hashes. Only small summaries/source/plots enter Git. Existing renderer edits belong to user work and remain untouched.

| Protocol requirement | Evidence now | Remaining work |
|---|---|---|
| E0 full dataset/frames/splits/bank | Prior135scene audit; new22scene hashed cache; fullbank checks | Physical extrinsics/unknown-wall completeness and runtime I/O audit |
| E1 visual dense | Scratch CNN3seeds and isolated UnLoc | Strong trainable visual model under matched training/exposure |
| E2 acoustic dense |3seeds, shuffle/zero/temporal/roll controls | map-only/geometry-removal on new cohort, material/furniture mismatch |
| E3 independent calibrated F1 | Shared training additive baseline only | Independent V/A scores, grouped calibration, fixed fusion |
| E4 F2/F3 capacity controls | Old tiny pilot only | Current-cohort matched active-capacity concat/non-interaction |
| E5 M1 true QKV interaction | Old tiny pilot only | Strong matched full-field interaction and ablations |
| E6 optional alternatives | Structure prediction tested | M2/physics only if motivated; not all optional variants required |
| E7 D0/D1/D2 | MAP and Support tested; support negative | Full-pipeline OOF Top20 and new discriminator after retrieval improves |
| T0/T1/T2 | New matched pretrained weights/order/reset optimizer3seeds | Equal-FLOPs comparison and stronger initialization |
| X1 audio controls | Query/zero/direct/late/roll on fixed model | Matched-retraining early/late representations |
| X2/X3 | Prior tiny geometry ablation | Current matched geometry/capacity tests |
| X5 density | Prior original vs pilot grid evidence | Controlled .20/.25/.50 and12/36/72 on same checkpoints/protocol |
| E8 robustness/cost/calibration | Six scene bootstrap, NLL, limited profile | Photometric/audio/FP shifts, physical furnished pairs, full latency/storage cost/Pareto, temperature/uncertainty |
| Independent final evaluation | Not done | Freeze method/protocol before accessing unused evaluation |

**Scientific stopping condition is not met.** This completed study does not complete the user's full goal. Follow-up should discriminate weak visual backbone/optimization from genuine benefit of interaction; maintain all negative baselines and avoid selecting the best seed/checkpoint from reused Val.

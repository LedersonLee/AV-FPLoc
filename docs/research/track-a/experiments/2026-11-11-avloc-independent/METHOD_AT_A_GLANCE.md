# AVLoc-style 6ch adaptation — method at a glance

Status: architecture/input implementation verified; training and matched evaluation pending.

RGB B×3×480×640 → ImageNet ResNet50 final conv B×2048×15×20 → 1×1 projection to128 → learned row/column+modality embeddings → 2 intra-modal self-attention layers →300 visual tokens.

Body-frame 6ch observed RIR → causal convolution with50ms log chirp (20Hz–20kHz,48kHz) → first100ms → STFT nFFT512/window64/hop16 →6 log-magnitude+6 adjacent sin-IPD+6 adjacent cos-IPD = B×18×257×301 → ResNet18 final conv (global pool/classifier removed) →128 projection and learned time/frequency+modality embeddings →2 intra-modal self-attention layers →90 audio tokens.

All attention uses128 dimensions and8 heads. A0:11 learned ray tokens (−50°…50°) seeded from geometrically aligned image columns,2 expert layers. A1:36 learned ray tokens,4 layers, audio only. A2:36 tokens,4 layers, concatenated V/A tokens; in-FoV ray tokens receive visual seeds. Separate ray and directional embeddings per expert. Softplus outputs distance μ and Laplace scale b in meters.

A3 contains V/A/AV experts. Each of36 two-layer routers consumes only detached expert scales;3 experts inside FoV,2 outside, vision weight exactly0 outside. Softmax weights form μF=Σwμ and bF=sqrt[Σw(2b²+(μ−μF)²)/2]. Fusion gradients still reach expert predictions; only the router-input path is detached.

Loss per branch: mean(|target−μ|/b+log(2b)); V uses11 rays. Full objective LF+0.1LAV+0.5LV+0.5LA. Standalone experts use their own NLL. Planned AdamW5e−4→1e−6 cosine,≤100epochs,patience20, checkpoint by Val ray L1.

Inference matches predicted Laplace rays against known2D floorplan rays at each candidate pose, summing log densities. Primary20cm/10° bank matches historical baselines; separate10cm evaluation must not be attributed to model improvement. AVLoc model receives no FP-RIR reference embeddings/bank. Ray scores can be normalized into an explicit discrete posterior, but calibration is a separate held-development analysis, not implied by sharpness.

Implementation assumptions and incomplete components are recorded in the companion audit. No performance claims are available yet.

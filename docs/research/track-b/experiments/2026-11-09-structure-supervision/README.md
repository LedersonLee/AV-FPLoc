# Track B: explicit structure supervision and failure diagnosis

Study ID `track-b-structure-supervision-20261109`, 2026-11-09. **Completed development experiments; mixed/negative results; entire Track B protocol still in progress. Evidence: Exploratory.**

This is an independent, FP-RIR-free full-field **non-interaction baseline**, not the protocol's geometry-mediated attention M1. It predicts body-relative boundary distances from observations and compares them with FP geometry plus a learned FP residual. No Track A weights, reference RIRs, B2 candidates, or GT proposal injection enter this model. The isolated UnLoc comparison is not an input to it.

## Question, competing explanations, and decisions

Does explicit geometry supervision make observed audio useful for localization, and does joint fine-tuning improve transfer? Competing explanations are real pose-dependent acoustic evidence, scene/layout priors, weak visual learning, optimization instability, and simulator artifacts. Tests include independent visual/acoustic models, shared-encoder additive fusion, shuffled auxiliary targets, dense-only training, frozen/partial/joint tuning, audio interventions, and a stronger isolated visual reference.

Observation: original joint-supervised fusion achieves Joint 6.424%, versus visual 2.083% and acoustic 5.122% (three seeds, six scenes, 384 development queries). Shuffled-target fusion is 2.865%, but dense-only fusion reaches 6.163%. Thus the correct auxiliary labels beat wrong labels; a localization benefit over ordinary dense training is **not established**. Auxiliary prediction error improves relative to a constant training-mean geometry prior, but a representation probe is not the same as successful localization.

Interpretation: there is query-dependent acoustic information, while weak visual learning and the retrieval objective remain bottlenecks. This does not prove a specific echo mechanism or a useful multimodal attention interaction. Decision: retain the structure baseline and its controls; improve/compare the actual full-field interaction and visual baseline before investing in a reranker.

## Data and evaluation protocol

- HSSD eligible Train/Val lists from the previous [audit](../2026-11-09-dense-se2-initial/audit_report.md), conservative source-family separation. SHA256 scene ordering fixed 16 training scenes and six development scenes; 64 uniformly scheduled queries per scene, 1,024/384 queries. See [exact selection](data_protocol.json).
- Every selected RGB/RIR input hashed; 1,408 observed six-channel RIR sidecars, waveform finiteness, and centre alignment checked. Camera contract checked across all 22 scenes: 640×480 source, horizontal FoV106.2602°, height1.5m. Test observations/labels not read. Development scenes overlap earlier experiments and historical teacher exposure; this is not an independent test.
- Native X/Z and yaw conventions follow the audited data contract. Actual array channels canonicalized by rig yaw; no central microphone. RGB resized128×96. RIR first16,384 samples at48kHz (341.3ms), original-window RMS normalization; log-magnitude and adjacent-channel IPD cos/sin, nfft256/hop32, first44 frequency bins, FP16 spectral cache.
- Full valid footprint bank at0.25m/36 yaw bins. FP0.1m raster, 16 capped20m boundary rays at0.1m step. Footprint boundary is not a verified physical wall; unknown/exterior reliability is incomplete. Largest selected bank231,156 candidates. Cache1,567,174,794 bytes includes observations, not simulated reference responses.
- Joint: position≤1m and circular yaw≤30°; R@1m; Fine: position≤0.1m, tolerance1e-6m. Raw20 and greedy diverse20 are separate. NLL is nearest-grid pose NLL, not soft-target CE; temperature is not calibrated. Val joint oracle100%, fine quantization oracle52.865%. No nearest-cell GT injection.
- Three seeds42/43/44, final scheduled checkpoint only. Paired bootstrap resamples six scenes10,000 times after averaging seeds within scene. Six scenes yield coarse uncertainty; queries/seeds are not independent scenes. Numerical results and CIs: [aggregate](aggregate.json), [per-run metrics](results_master.csv), [paired comparisons](paired_scene_bootstrap.json).

## Architecture and training

2,402,041 total parameters. Separate trainable visual CNN, acoustic CNN and FP CNN. Candidate FP features use a rotated3×3 stencil and geometry; a bounded0.15tanh residual adjusts 16 normalized ranges. Exact dense matching is weighted squared distance, with five visible bearings for visual and16 for acoustic. Fusion is fixed0.85visual+0.15acoustic under shared training; **not independently calibrated F1**, not nonlinear joint interaction.

Original supervised runs:1,500 auxiliary-only updates then1,000 full-bank dense soft-CE updates plus10×auxiliary MSE. Dense-only uses2,500 dense updates, no auxiliary loss. Shuffled targets use a fixed within-scene permutation; dense GT remains correct. Batch8, AdamW lr0.0003, weight decay0.0001, gradient norm cap5. Both continuous GT geometry and pose are training/metric targets only.

Original frozen/partial comparisons reran auxiliary pretraining with the same seeds, but archived weights were **not bitwise equal** for all three seeds. They remain exploratory and are not a strict shared-initialization experiment. The corrected matched study loads exactly the same archived frozen-run pretrained checkpoint for every regime, resets AdamW for all, skips1,500 sampling steps to align subsequent data order, then performs1,000 dense+aux updates. Frozen trains the scalar score scale (one parameter); partial trains last modality projections and last FP residual layer (5,169 parameters); joint trains all2,402,041. Frozen is therefore a restricted scale-only head baseline, not every possible frozen-backbone head. Same update count is not equal FLOPs; actual training time/VRAM are reported.

Matched initialization results: frozen8.073%, partial9.028%, joint6.163%. Partial−joint is+2.865pp, paired scene95% CI[+1.128,+4.601]; frozen−joint+1.910pp,[+0.260,+3.819]. This supports a development-set advantage for restricted tuning under this learning rate/update schedule, not a universal benefit of freezing. Original supervised−dense-only difference is+0.260pp,[−1.215,+1.565].

## Acoustic interventions and hypothesis extraction

[Extended results](extended_summary.json) reload all384 raw waveforms and reproduce cached spectra. Normal pose metrics replay2,304 query-model cases with zero mismatches.

Acoustic Joint falls from5.122% to0.521% under within-scene query shuffle and0.434% under zero spectral input. Direct-only(samples0:32) gives0.521%; masking these samples gives3.038%; late-only(samples7200:16384) gives0.608%. Original RMS is retained: truncation changes energy/distribution, so these are interventions on a fixed model, not matched-retraining proof of temporal information absence. Zero spectral input is not the spectrum of a silent waveform. No claim that arbitrary ray distances correspond to specular echo times.

A three-channel circular roll represents180° array yaw. Acoustic Joint against the rotated label is3.125%, versus0.434% against the original label. This supports partial orientation sensitivity, not exact equivariance or real-room generalization. Fusion with unchanged RGB and rotated audio gives3.125%: inconsistent modalities degrade localization.

Support-NMS integrates full posterior mass in1m/±30° neighborhoods, selects original-field3×3×3 local maxima, then applies1m/30° suppression, fixed temperature1. Acoustic MAP5.122%→support3.993%; fusion6.424%→6.250%. Acoustic raw/diverse/support20 coverage19.271/41.840/39.497%; fusion33.941/57.899/50.000%. Support did not improve these models. Reranking has not been run; no OOF claim.

## Strong isolated reference and limitations

UnLoc on the same384 queries and candidate IDs: original clamped likelihood Joint35.938%, R@1m37.240%, Fine13.802%, raw20 71.354%, diverse20 79.948%. Predeclared unclamped ablation42.969/44.792/15.625/78.906/86.719%. Both are retained, not selected after observing performance. [Reference details](unloc_source.json) and [metrics](unloc_summary.json).

This reference uses original640×480 input,11 rays and larger range, a historical model exposed to Train116/Val19, and external depth pretraining. Scratch models use16 training scenes and no external weights. The comparison demonstrates baseline weakness, **not a fair architectural ranking or independent held-out improvement**. External UnLoc checkout and existing modifications are preserved; its code/weights are isolated from proposed inference.

## Verification, failures, reproducibility

22 functional tests pass, including full-bank indexing, no GT dependence, frame checks, three-branch gradients and support mass against brute force. Largest-bank batch8 five-step profile:0.384s and681,462,784 peak allocated GPU bytes with one scene resident; full cache adds memory. This is a narrow training profile, not full deployment cost/Pareto evidence. [Profile](profile.json).

Correction: original supervised non-normal ray-MSE/MAE diagnostics accidentally used normal predictions. 5,759 affected rows corrected by checkpoint replay; normal metrics and pose localization unaffected. Originals are preserved externally; [correction summary](ray_correction.json). CPU-only checkpoint comparison initially failed without map_location; explicit CPU mapping resolved it. No training data or checkpoint was modified by that failed inspection.

[Registry](experiment_registry.csv), [provenance](provenance.json), [commands](reproduction.md), [decisions and remaining gates](decisions.md). Raw logs, per-query predictions, caches and checkpoints remain local under `/home/jeongeonlee/track1/experiments/track_b_structure_20261109`; they are intentionally not uploaded. Code SHA and source hashes distinguish clean versioned Track B code from unrelated preexisting dirty renderer files. Clean-clone reproduction still requires the external dataset, prior audit and isolated baseline assets.

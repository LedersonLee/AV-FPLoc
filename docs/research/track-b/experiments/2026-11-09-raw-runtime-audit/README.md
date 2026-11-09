# Raw RGB/body-frame RIR to full-field MAP: runtime replay audit

2026-11-09. Track B, Exploratory. Completed: offline packet export and raw inference audit both exit0,41 tests pass. Source `0c0f0eece4b1bcd6595dc420463684c668875710`. This is a bounded engineering validation, not a new trained method, independent accuracy evaluation or complete deployment-cost benchmark.

## Question and competing explanation

Prior full-bank evaluations consumed cached visual features and RIR spectra. A hidden mismatch between cached preprocessing and deployable raw observation handling could invalidate a practical pipeline claim. This audit asks whether decoded original RGB and a calibrated body-frame6ch RIR reproduce the cached features, entire field and raw Top20 exactly, without passing GT pose into inference. It also makes the upstream input cost boundary explicit instead of calling cached-score timing end-to-end latency.

## Dataset assembly is distinct from inference

The existing synthetic positive-pair dataset reuses ring-array renders at base acoustic yaw. Offline dataset construction uses the schedule's world/base yaw to cyclically reorder the6 channels into the RGB rig's body frame. `export_body_packets` materializes that body-frame waveform once and copies the exact original RGB file; input hashes and observed-RIR role are checked. This assembly step **does use orientation metadata** and is not called GT-free inference. The schedule-based transform was already part of the structure cache in study17; the current export makes that boundary explicit.

After export, `audio_features(body_rir,sample_rate_hz)` accepts only the aligned waveform and48kHz contract. It pads/truncates to16,384 samples, applies original-window RMS normalization, computes nfft256/hop32 log-STFT+adjacent IPD, retains44 bins and rounds to FP16 exactly as the established cache. `visual_features(backbone,decoded_bgr)` accepts only the decoded image, runs upstream preprocessing/frozen DINOv2 ViT-L, pools4×16 and rounds to FP16. No world yaw, GT pose, scene ID or query ID enters either function.

This does not establish that a new physical sensor's unaligned channels can be corrected without calibration metadata. A physically calibrated rig would supply the body-frame channel convention directly. Metadata-based synthetic assembly cannot substitute for that real-world calibration validation. The ring-symmetry simulator assumptions and upstream asset exposure remain as previously documented.

## Fixed protocol and observed result

Use the first already frozen query in each of six existing Val scenes, chosen without examining outcomes. Run completed joint attention/seed42 with its original checkpoint, frozen upstream encoder and full prepared map bank. No model selection, training, candidate pruning, B2/GEOQ proposal, reference RIR or GT feature is added. Reference caches containing evaluation labels are read only by the outer auditor; map preparation receives an explicit floorplan/calibration/candidate whitelist, and observation inference receives only image/waveform and prepared map tensors. Query IDs align records outside inference.

| Scene | Full candidates | Features / spectrum / field / Top20 |
|---|---:|---|
|108736611_177263226|99,612|all exactly equal|
|108736635_177263256|81,684|all exactly equal|
|105515448_173104512|154,764|all exactly equal|
|108294783_176710410|81,720|all exactly equal|
|108294624_176710203|92,880|all exactly equal|
|102344457|94,644|all exactly equal|

Six probes,605,304 candidate scores, zero feature/spectrum/full-field/Top20 mismatches at **zero absolute and relative tolerance**. MAP returns a three-component pose from the candidate bank. No new accuracy statistic is computed from six selected probes; prior cohort-level results remain authoritative. No statistical confidence interval is appropriate for this deterministic replay assertion, and it does not imply every query/model/backend was audited.

During the decoded-input→feature→full-field forward, Python file hooks observe only the first-call PyTorch opt_einsum lazy import; later probes open no Python-visible files. No dataset/reference path is opened in that region. These hooks do not trace native-library I/O, and raw image/wave disk reads occur before that region. This is narrower than a complete operating-system file-access audit. Upstream source is the reviewed UnLoc checkout's nested DepthAnything package; historical HSSD visual-head weights are not loaded.

## Costs and limitations

Observed decoded-input→MAP times are104–210ms across the six probes, including upstream feature calculation, CPU audio preprocessing, GPU transfer, exhaustive cached-map scoring and raw Top20. They exclude model/checkpoint loading, image/wave disk decode, map raster/ray preparation and cache construction, diverse selection and any discriminator. The first call includes framework warmup; these are single observations during concurrent authorized GPU training, **not isolated warm/cold benchmark distributions**. No speedup, Pareto or same-bank B2 amortization claim follows.

Additional requirements remain: all-candidate map setup, cold/warm repetitions, CPU/GPU peak resources, upstream/static/model/per-map storage separation, final backend latency, raw-input audits for other selected architectures/regimes and matched reference-bank costs. The existing independent-F1 discriminator is not part of this attention MAP audit.

## Decision and next falsification

Retain this raw-input entrypoint: it reproduces established cache behavior in all six predefined probes and clarifies the body-frame alignment boundary. It rules out a cache/runtime preprocessing discrepancy for these cases, not all leakage or calibration risks. Next use this path for controlled end-to-end timings after the regime comparison, keeping map preparation and backend costs explicit. Broader raw-query checks should use predetermined coverage, not cherry-picked successes. The overall goal remains incomplete.

## Reproduction and provenance

External root `/home/jeongeonlee/track1/experiments/track_b_runtime_20261109`, packetspackets_v1 and auditaudit_v1. Packet images/waveforms and logs remain external; only protocol/hash manifest and small audit summaries are in Git. Upstream checkpoint SHA`6f82ff2bc543ac02ddff4aa31fa363676a8305dd3ccf04e80e2af115a044cb6d`; attention checkpoint SHA`88cd5514a935c3cf2ef7f404a58dd3bca494749fb732b03bbb18eadd3a61af40`. Same Dockeravfploc:cu124-clean, RTX4090, torch2.4.1+cu124, no network, non-root UID. xFormers was unavailable; existing PyTorch fallback completed normally. No failed run/checkpoint was overwritten.

Commands executed: `python -m track_b.scripts.export_body_packets --root /dataset --cache /structure/cache --out /results/packets_v1`; then `python -m track_b.scripts.audit_raw_inference --packets /results/packets_v1 --cache /structure/cache --features /foundation/features_v1 --source /work/external/checkouts/UnLoc/depth_anything_v2 --upstream-checkpoint /pretrained/depth_anything_v2_metric_hypersim_vitl.pth --checkpoint /interaction/foundation_matched_v1/attention_42.pt --out /results/audit_v1`. Mounts map to the existing HSSD root and study17/19/21 external roots. Use fresh output directories. Clean-clone reproduction requires the recorded external assets; Git alone does not supply weights/data.

Baseline is the same checkpoint's cached-feature forward at batch1; no ablation/training is applicable to this deterministic replay. Tests additionally reject wrong sample rate/channel count and verify channel-sensitive preprocessing. Docs branchrefactor/hssd-two-branch-fusion. Code SHA is separate from the later Docs commit.

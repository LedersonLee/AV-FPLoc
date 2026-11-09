# Separately sourced foundation visual features

Study `track-b-foundation-visual-20261109`, 2026-11-09. **Status: one-query implementation smoke complete; localization training/evaluation not yet run. Evidence: Exploratory.**

Hypothesis: a stronger upstream RGB representation may address the weak scratch visual baseline. Competing explanations include different capacity/pretraining exposure and an adapter that learns scene priors rather than transferable geometry. The prior isolated UnLoc result does not establish a fair architectural advantage; this study must use a fresh Track B head and the same selected scenes/queries. Historical HSSD-trained UnLoc/Track A weights are not loaded.

Actual smoke: strictly loaded the existing upstream DepthAnything-V2 ViT-L metric Hypersim checkpoint, SHA256 `6f82ff2bc543ac02ddff4aa31fa363676a8305dd3ccf04e80e2af115a044cb6d`. External source checkout `a561b849ebae10a6f5ef49e26c83cbbcd36c71bf`, origin `https://github.com/DepthAnything/Depth-Anything-V2.git`; checkpoint directory is preexisting/untracked. Python sources are hashed in protocol. No upstream HSSD asset-family independence claim is made from a dataset name alone.

One already-selected **training** RGB (scene108736779_177263484) was verified against its original input hash and processed on CPU. Upstream preprocessing: BGR→RGB, cubic resize shorter side518/multiple14, ImageNet normalization. Actual transformed shape1×3×518×686. Last DINOv2 block features are pooled4×16 and stored as1×1024×4×16 FP16. All values finite. One-query extraction7.852s; serialized cache132,440bytes. This is a CPU smoke measurement, not GPU production latency. Process exit0; xFormers unavailable message uses ordinary implementation and was not a failure.

A fresh trainable visual adapter (1024→64 convolutions plus16-ray head) passes gradient, shape and finite-value tests. Together with earlier tests,28 pass. The upstream encoder is frozen for this planned cache path; this is **not** full-foundation joint fine-tuning. Its inference cost must include upstream extraction, not only adapter matching. No checkpointed adapter has yet been trained, so localization metrics, baselines, ablation gains and CIs are not applicable yet. This does not complete the full Track B protocol.

Next: build the same16Train/6Val feature cache, train fresh adapter/FP matching with fixed budgets and seeds, compare against the existing scratch baseline, then decide whether improved visual evidence merits repeating interaction. Preserve low-light tests by extracting perturbed RGB features separately; zeroing cached features is not equivalent to a black image. Test observations remain unused.

Executed source `bf032b5` (full SHA in provenance). Standard Docker image `avfploc:cu124-clean`, PyTorch2.4.1+cu124, CPU-only smoke, network disabled. `/data` is the existing HSSD root; `/prior` is the preceding structure study; `/out` is `/home/jeongeonlee/track1/experiments/track_b_foundation_20261109`.

```text
python -m track_b.scripts.cache_foundation_visual --root /data --cache /prior/cache --source /work/external/checkouts/UnLoc/depth_anything_v2 --checkpoint /data/UnLoc/pretrained/depth_anything_v2_metric_hypersim_vitl.pth --out /out/smoke_v1 --device cpu --smoke
```

[Protocol/source hashes](protocol.json) · [Actual manifest](manifest.json). Checkpoints and feature tensors remain local; no weights or image observations are uploaded. This smoke is separate from the ongoing matched attention comparison and does not modify its models or results.

Full16Train/6Val feature-cache extraction has now started at runtime source1610e0f with `--device cuda` and output `features_v1` (omit `--smoke`). It is a new artifact; one-query smoke remains preserved. Complete cache/training results are still pending.

# Separately sourced foundation visual features

Study `track-b-foundation-visual-20261109`, 2026-11-09. **Status:12fits completed with mixed/negative findings; objective controls running. Evidence: Exploratory.**

## Completed first comparison and matched tuning

The full22scene/1,408query feature cache completed,184,583,546bytes,134.665s of per-scene extraction (checkpoint loading excluded). RGB hashes and query IDs match the original16Train/6Val cohort. The original smoke files below remain a historical record; [complete manifest](feature_manifest_complete.json) and [complete feature protocol](feature_protocol_complete.json) are authoritative for training.

Three fresh visual fits use the original1,500aux+1,000dense updates, batch8, AdamW0.0003/0.0001, seeds42/43/44. Visual and FP adapters train; upstream foundation encoder stays frozen. Three additional regimes load the exact same archived foundation-adapter checkpoint per seed, reset AdamW, skip1,500 sampling draws, and run1,000 dense+aux updates. Frozen trains score scale only; partial trains last visual projection and FP residual output; joint trains all participating adapter/FP parameters. The acoustic branch is unused by the visual score; nominal model2,792,297 parameters include it and exclude the upstream foundation model. These are not capacity/pretraining-matched comparisons to scratch RGB.

| Condition | Joint | R@1m | Fine | Raw20 | Diverse20 |
|---|---:|---:|---:|---:|---:|
| Scratch original visual |2.083%|3.038%|0.087%|21.441%|38.281%|
| Foundation original dense continuation |5.208%|7.031%|0.000%|37.500%|57.986%|
| Matched frozen |19.271%|20.573%|1.042%|66.840%|81.858%|
| Matched partial |11.024%|12.587%|0.347%|51.476%|75.087%|
| Matched joint |5.556%|6.944%|0.174%|38.889%|58.073%|

Observation: matched frozen−joint+13.715pp, scene95%CI[+10.243,+17.717]; partial−joint+5.469pp,[+2.951,+7.813]. Original foundation continuation−scratch+3.125pp,[+2.604,+3.559], but Fine drops to0 and external pretraining/capacity differ. All are reused development results, not independent evaluation. [Aggregate](aggregate.json), [paired CIs](paired_scene_bootstrap.json), [all12runs](experiment_registry.csv).

The auxiliary-only checkpoint evaluated against **raw** FP geometry has Joint19.531%; the original dense continuation uses learned FP residuals and reaches5.208%. That before/after contrast alone changes both optimization and FP representation. The stricter matched frozen/partial/joint comparison uses the same initial residual model and optimizer reset, and confirms harmful tuning under this schedule. Frozen normalized-ray MSE0.003638 versus joint0.011937; ray MAE0.458m versus1.164m. Yet NLL improves9.884→9.482 while Joint falls: likelihood optimization and MAP accuracy diverge. This does not prove a universal defect of dense CE or foundation tuning.

Competing explanations: auxiliary gradient too weak relative to dense CE, learned FP residual drift, optimizer dynamics, limited training data and overfitting. Decision: preserve the strong frozen visual reference and all failed tuning results; discriminate loss weight and FP residual before more complex attention. This frozen visual baseline is not a substitute for the requested three-trainable-branch framework.

Predeclared next2×2 control: same pretrained state/order/1,000updates, auxiliary coefficient10 versus1000, learned FP residual versus fixed raw FP geometry. Existing matched_joint supplies learned-FP/10; three new cells ×3seeds run at source6874941. Weight1000 is a coarse mechanism intervention, not a validation-selected optimum. Fixed-FP runs disable a trainable branch deliberately as an ablation, not as the proposed final method. Results are pending and not counted above.

[Validation](validation.json) records actual exit0 for full extraction and both completed training batches. The encoder state provenance comes from immutable checkpoint hashes; no Test observations are used. Photometric tests must start from perturbed RGB and re-extract foundation features; feature zeroing/scaling is not a physical light intervention.

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


Completed training command (standard GPU container mounts):

```text
python -m track_b.scripts.train_structure --cache /prior/cache --visual-cache /out/features_v1 --out /out/visual_supervised_v1 --modes visual
```

For matched regimes add `--regime frozen|partial|joint --aux-steps 0 --dense-steps 1000 --pretrained-dir /out/visual_supervised_v1 --skip-sampling-steps 1500`, each in its own fresh output directory. Original actual training source isd283a4b; matched source2762c2f; full SHAs are in registry. The original one-query smoke and its limitations below are preserved rather than overwritten. Overall Track B requirements, including stronger multimodal comparisons, OOF discrimination and final independent evaluation, remain incomplete.


## Completed objective controls (21 total fits; supersedes pending status above)

All nine additional runs completed normally (batch exit 0, three COMPLETE markers), with the same three archived adapter initializations, query sampling, fresh optimizer, 1,000 dense updates and reused six development scenes. No Test observations were accessed. The 2×2 reference cell is the existing matched_joint_v1, not an extra run.

| FP descriptor | Auxiliary weight | Joint % | R@1m % | Fine % | Raw Top20 % | Diverse Top20 % | NLL |
|---|---:|---:|---:|---:|---:|---:|---:|
| Learned residual | 10 | 5.556 | 6.944 | 0.174 | 38.889 | 58.073 | 9.482 |
| Fixed geometry | 10 | 6.250 | 7.813 | 0.000 | 38.802 | 62.934 | 9.555 |
| Learned residual | 1000 | 11.719 | 13.021 | 0.521 | 50.347 | 70.486 | 9.764 |
| Fixed geometry | 1000 | 18.142 | 19.705 | 1.215 | 63.976 | 79.774 | 9.810 |

Observed: increasing auxiliary weight improves Joint by +6.163 pp [95% scene-bootstrap CI 3.819, 8.681] with learned FP and +11.892 pp [9.549, 14.497] with fixed FP. At weight 1000, fixing FP adds +6.424 pp [4.688, 8.333]; at weight 10 its +0.694 pp CI [-1.215, 2.951] includes zero. Strong auxiliary supervision preserves visual ray MAE (~0.47 m versus 1.16 m at learned FP/10), yet learned FP still gives lower localization than fixed FP. This supports objective imbalance and learned map distortion as interacting explanations under this training schedule, not a universal indictment of dense loss. Direct descriptor-drift measurement and inference-time descriptor swaps are still needed to isolate the representation mechanism.

The strongest trainable-FP control remains below frozen-adapter/FP Joint 19.271%: −7.552 pp CI [-10.330, -4.514]. Fixed FP is a diagnostic ablation that removes a trainable branch and does not satisfy the final three-trainable-branch proposal. Weight 1000 is a deliberately coarse mechanism intervention, not an independently selected optimum. Reused Val, upstream exposure uncertainty, limited 16-scene training, and near-zero fine accuracy constrain generalization. All failed/lower-performing arms remain in the registry.

Actual control training source: `6874941298622eee18819fd48207fea8d5df2812`; aggregation source: `4aa0d6c5e80987bb813316c1cdc1e625d12184a2`. Reproduce the three additional arms with the preceding matched-joint command, distinct output directories, and respectively `--fixed-fp --aux-weight 10`, `--aux-weight 1000`, `--fixed-fp --aux-weight 1000`. Aggregate with `track_b.scripts.summarize_foundation` and `--controls-code 6874941298622eee18819fd48207fea8d5df2812`. The 29 Track B tests pass after these changes. Timing fields are recorded implementation timers, not complete pipeline throughput; feature extraction and final evaluation are excluded.

Next discriminating analysis: evaluate each learned-FP checkpoint with its own versus raw geometry, and quantify FP descriptor drift. This separates visual adapter changes from inference-time map changes without retraining or touching Test. Then build independently trained fusion and capacity-matched interaction controls around the preserved visual signal; the overall Track B goal remains in progress.

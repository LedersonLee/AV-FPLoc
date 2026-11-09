# Expanded development cohort and scene-residency preparation

2026-11-09. Track B, Exploratory. Status: preflight complete;131-scene cache generation running; CPU scene-transfer implementation and three short fits complete. Strict bitwise equivalence check **failed and is retained**, with follow-up numeric diagnostics. No expanded-cohort training result or independent validation is claimed.

## Purpose and frozen dataset protocol

The existing16 Train/6 Val pilot cannot establish broad scene generalization. Use every eligible scene from the already audited development split:112 Train and19 Val, preserving known source-family separation and Test-family exclusions. Each scene uses64 uniformly spaced schedule entries, selected before expanded outcomes. This adds96 Train and13 Val scenes; total7,168 training/1,216 development queries,8,384 overall. It is all eligible **scenes**, not all available queries. The larger Val set retains prior upstream/research exposure and is **not an independent Test set**. Test observations remain unopened.

Preflight checked all selected RGB/Observed-RIR paths,64 distinct query IDs per scene, map/metadata/schedule hashes, grid cardinality and disk headroom. No model score or GT accuracy was used to pick scenes/queries. Source split hash`f4005163927feeeaefe767926032dc9096cf79513bae89b61c37beade4c4e9ed`. Existing .25m/36-yaw grid,16 geometry rays,128×96 auxiliary RGB cache,16,384-sample48kHz RIR/STFT-IPD representation remain unchanged. Full upstream foundation features will be prepared separately after the structural cache validates.

Estimated structural-cache storage11,909,755,468bytes; available space at preflight63,763,406,848bytes. The guard required twice the estimate plus10GiB spare before launch. This is an estimate, not actual storage or complete deployment cost. Old results/checkpoints are preserved. Generated inputs remain outside Git. No new repository or remote was created.

## Runtime preparation and hypothesis

Loading the expanded cohort into GPU memory at once would unnecessarily couple dataset size to accelerator memory. `--scene-residency cpu` stores scene tensors on the host and transfers the requested scene on access; observations, geometry, query order, sampling RNG and losses are unchanged. The existing default remainsGPU residency. Reference caches are read identically; this is not an approximation or candidate pruning scheme. Peak GPU use can still include the previous computation graph/current scene and model tensors; it is not asserted to equal exactly one scene allocation. Host RAM must hold the dataset.

Hypothesis: moving residence should preserve training behavior within normal numerical execution variability. Competing explanation: transfers, changed layout or altered data order could change features/gradients/metrics. Three fresh visual/foundation fits on the existing16/6 cohort use seed42, one auxiliary plus one dense update, same auxiliary1000/map-anchor1000 and otherwise matched settings: residentGPU, streamingCPU, repeatedresidentGPU. No checkpoint/epoch selection from these probes.

## Observed short-run comparison and preserved failure

Both residency arms and the resident repeat completed, exit0. The strict comparison requires exact states, loss/gradient histories and all evaluation rows; it exited1 for GPU versusCPU and wrote failed_bitwise_check.json. This failure is **not relabeled a pass**.

| Comparison against first GPU fit | Auxiliary checkpoint | Final maximum parameter difference | Changed tensors | Core Train/Val metric mismatches |
|---|---|---:|---:|---:|
| CPU-resident scenes | bitwise equal |8.5682e−8|2|0/0|
| GPU-resident repeat | bitwise equal |2.2352e−8|2|0/0|

Only floorplan.0.weight and floorplan.2.weight differed. Step loss values were identical. Train1,024 and Val384 query rows align by(scene,query,arm); Joint/R@1m/Fine/raw20/diverse20 mismatch counts are zero in both comparisons. Maximum train NLL differences:9.5367e−7 forCPU and1.9073e−6 forGPU repeat; maximum Val NLL differences0 and9.5367e−7 respectively. Train diverse20 is not measured in this smoke and its matching nulls are not coverage evidence. Full detail is in residency_numerics.json.

**Interpretation:** bitwise differences also arise without changing residency, so they cannot all be attributed to CPU scene transfer. The difference first appears after the dense/map update and is localized to FP convolutions. GPU reduction/backward execution variability is a plausible explanation, **not isolated causally by this test**. CPU-versusGPU variation is not identical in magnitude to the single repeat. No claim of deterministic long-run equivalence, identical pose indices, full-field equality or equal final convergence follows from matching aggregate correctness indicators.

**Decision:** retain the streaming implementation with this explicit limitation and source tracking; use it consistently for future expanded training, not mix residency schedules within a matched comparison. Preserve strict failure and all three outputs. The short smoke is an implementation check, not a scientific accuracy result. Expanded training budgets/controls must be frozen separately before fitting; cache completion alone does not establish a frozen training/evaluation protocol.

## Current execution and required verification

External root `/home/jeongeonlee/track1/experiments/track_b_scale_20261109`. Structural cachecache_v1, logcache_v1.log, observation session72407. At the recorded checkpoint26/131 scene artifacts existed and the Docker process was live; this is progress, not completion. Require all131 entries, aligned query manifests/hashes and equality checks against the original overlapping22 scenes before relying on the cache. No incomplete directory is used as a finished dataset. Existing study24 regime training continues separately.

After structural validation, extract the same frozen upstream RGB features for the expanded cohort and prepare matched scaling fits. Expanded existing Val must remain development data; method freeze and independent evaluation are still separate requirements. GPU data residency addresses memory, not scientifically valid generalization or complete cost accounting.

## Sources and reproduction

- Preflight: `f70fe38bf674327afbc2764dfc93dc818d3cf962`.
- Cache launch and initial two residency smokes: `461cb6ea52dd76bec09acf3a2bb5be08f50139c6`.
- Strict checker and GPU repeat launch: `748942aba3817f20b1e35b6a78c413cbaf1e3de3`.
- Numeric follow-up: `07d6e27202b247e8573dfd7ad53e74cdabba2aa4`.

Preflight: `python -m track_b.scripts.preflight_expanded_cohort --root <HSSD> --audit <study15-audit_v1> --pilot <study17-cache> --out <fresh-preflight>`.

Cache: `python -m track_b.scripts.structure_cache --root /dataset --audit /audit --out /results/cache_v1 --train-scenes 112 --val-scenes 19 --queries 64`. Dockeravfploc:cu124-clean, non-root UID, no network, OMP/OpenBLAS2threads; CPU job, dataset/code read-only. Full cache retains existing body-frame waveform assembly contract; it does not remove prior calibration assumptions.

Smokes: `python -m track_b.scripts.train_structure --cache /structure/cache --visual-cache /foundation/features_v1 --out <fresh-output> --modes visual --seeds 42 --aux-steps 1 --dense-steps 1 --aux-weight 1000 --fp-anchor-weight 1000 --scene-residency gpu` orcpu. Repeatgpu to a third directory. Check with `python -m track_b.scripts.check_scene_residency --root /results --out <new-json>` (observed exit1 preserved), then `python -m track_b.scripts.audit_residency_numerics --root /results --out <different-new-json>` (exit0 diagnostics). Existing GPU training ran concurrently; timings are not used to compare speed.

All source paths and hashes remain linked by this study and the canonical RESEARCH_INDEX. No raw images/audio/spectra/weights are uploaded. Docs branchrefactor/hssd-two-branch-fusion. Clean-clone replay depends on external data/checkpoints. Overall research remains incomplete.

# Matched foundation interaction from a strong independent starting field

Date: 2026-11-09. Track B. Status: smoke completed, nine-fit main comparison running. Evidence: Exploratory reused development. This study tests H-B2/H-B3 with stronger representations, and is not a declaration of interaction success.

## Hypothesis, alternatives and design

The previous small scratch interaction comparison failed to beat capacity-matched concat/separable alternatives. The stronger independent foundation baseline now reaches 23.872% Joint and 84.375% diverse Top20 (three seeds). Hypothesis: full-bank candidate-conditioned token interaction can improve on the stronger starting point. Competing explanations: improvements arise from extra trainable capacity, additional optimizer updates, map drift or observation overfitting rather than cross-modal interaction. Attention versus equal-parameter pooled concat and additively separable modality functions isolates interaction more directly than comparison only to unchanged F1. F1 remains an essential deployment baseline with fewer head parameters and fewer training updates; differences against F1 are not pure causal attention effects.

The FP modality uses two learned towers to preserve the independently trained visual/acoustic map representations. Both consume the same footprint and candidate geometry; neither consumes candidate RIRs. This is explicitly a dual-tower FP branch, not a single shared FP encoder. The three modality pathways remain trainable (the upstream foundation RGB encoder remains frozen, with a trainable visual adapter). No historical HSSD UnLoc head, B2 proposals or Track A weights are loaded.

## Dataset / Evaluation Protocol and initialization

Same 16 Train scenes/1,024 queries, six Val scenes/384 queries, three seeds 42/43/44, .25m/36-yaw exact full candidate bank. Source scene/query/cache hashes are in protocol. No Test observations. No independent-evaluation claim: Val reuse and uncertain upstream asset-family exposure persist. Joint ≤1m and ≤30°, R@1m, Fine ≤.1m, raw and diverse Top20, NLL and entropy are reported. Scene-paired bootstrap is required after all nine fits finish; no selecting the best seed or intermediate step.

All three modes use 2,834,908 nominal parameters excluding the frozen upstream RGB encoder, identical serialized initial tensors per seed, same sampled queries, 1,000 updates and fresh AdamW lr3e-4, weight decay1e-4, batch8. Equal parameters/updates do not imply equal FLOPs. Visual adapter/map initialize from foundation `learned_fp_w1000_anchor1000_v1`; acoustic encoder/map from structure `supervised_v1/acoustic_<seed>.pt`. Separate score scales are preserved. New gate output and joint-head final layers start at zero, giving gate=1 and joint correction=0, hence exact independent .85V+.15A starting scores. Initial internal attention gradients are intentionally zero for the first update; they activate after final-layer updates and must be checked thereafter.

Loss: exact full-bank soft-target CE + 1000×visual boundary-ray MSE + 10×acoustic boundary-ray MSE + 1000×map anchor. Visual map anchor is mean squared deviation from raw normalized geometry; acoustic map anchor is mean squared deviation from that seed's initial learned acoustic descriptors. The acoustic anchor is computed from map inputs and frozen initialization, not GT. All 16 descriptor dimensions are anchored. This asymmetry preserves the empirically supported visual geometry constraint without erasing the already learned acoustic map. It is a declared design choice, not a tuned optimum or proven physical model.

Normal, within-scene cyclic audio replacement and re-extracted RGB brightness .2 are evaluated with scheduled final weights. Brightness .2 did not reduce the stronger baseline's visual accuracy in study20; it is a compatibility condition, not established visual failure. Further predeclared occlusion/noise conditions and physical/material shifts remain required. Full-pipeline discriminator, final scaling, independent evaluation and end-to-end cost remain incomplete in the overall goal.

## Validation completed before main fitting

31 Track B tests passed, including exact independent starting scores, full-bank chunk/cache consistency, active modality and FP gradients, equal parameter counts and the separable score identity. These tests use small synthetic geometry; actual dataset provenance and full-grid numerical replay are separate checks.

Two-update seed42 attention smoke exited 0, completed six-scene final evaluation, and peaked at 2,890,003,968 allocated GPU bytes. It is a runtime/gradient check, not a selected model or a scientific performance claim. Both attention retrieval modules show nonzero gradient after the second update; main step100 also confirms active visual/acoustic/map/attention paths. Full raw checkpoints and per-query outputs stay outside Git.

Main run directory: `/home/jeongeonlee/track1/experiments/track_b_interaction_20261109/foundation_matched_v1`; log: sibling `foundation_matched_v1.log`. Actual training source SHA `6285d4b89e127f5b71479c49071dcc96a2ff908c`. Runtime observation client session70126 is distinct from the Docker process: if the observation client disconnects, inspect Docker and existing logs before any restart. The run has not been declared complete.

## Reproduction

Docker `avfploc:cu124-clean`, existing RTX4090, no network, non-root UID, read-only code and source dataset caches; only external result root writable. Do not stop unrelated GPU processes. Code/Docs branch `refactor/hssd-two-branch-fusion`.

```bash
python -m track_b.scripts.train_foundation_interaction --cache /structure/cache --visual-run /foundation/learned_fp_w1000_anchor1000_v1 --acoustic-run /structure/supervised_v1 --visual-cache /foundation/features_v1 --lowlight-cache /foundation/features_lowlight_v3 --out /results/foundation_matched_v1 --steps 1000 --seeds 42 43 44 --modes attention concat separable
```

Source mounts: `/structure` is external `track_b_structure_20261109`, `/foundation` is `track_b_foundation_20261109`; output `/results` is `track_b_interaction_20261109`. Use a new output path; never overwrite the live run. Smoke used the same command with steps2, seed42, attention only, and a distinct output. No claim of complete clean-clone reproduction without the hashed external checkpoints/caches is made.

References: [foundation representation controls](../2026-11-09-foundation-visual/README.md), [strong independent F1 and brightness](../2026-11-09-strong-independent-fusion/README.md), [prior scratch matched comparison](../2026-11-09-matched-interaction/README.md). See decisions.md for the architecture changes and their limits.

## Actual initial-state replay completed

Seed42 serialized initial model reproduced all 1,152 query/condition cases (normal, audio shuffle, brightness) against archived independent F1, with zero Joint/R@1m/Fine/raw20/NLL mismatches at declared numerical tolerance. Replay job exited0. This checks the actual full-grid initial state rather than relying only on the synthetic unit test. Replay source `f0dfc32` (full SHA in provenance). Main batch container `d0d4fbe4e1d2` was verified live at attention seed42 step400; evaluation outcomes are still pending.


## Runtime interruption and verified resume (supersedes original live snapshot)

The original batch exited1 after completing attention/seed42, when comparing the next model's CPU initial-state tensors with a checkpoint loaded to CUDA. This was a device mismatch in an initialization equality assertion, not a training-loss or data failure. Original `foundation_matched_v1.log`, complete attention weights/metrics and initial checkpoints were retained. The inspection helper itself was fixed to load the comparison checkpoint on CPU. The optimizer/loss/data schedule was not changed.

Resume source SHA `18c0e3bf4297b9e0176ac3008710de06ec604da9` adds an explicit `--resume` path that checks the stored protocol for equality, skips only completed scheduled fits with all expected artifacts, and retains their existing results. Each newly completed fit records its actual execution SHA. The original attention/seed42 remains attributed to `6285d4b89e127f5b71479c49071dcc96a2ff908c`; the later fits are attributed to the resume source. Resume log `foundation_matched_v1_resume1.log` starts with `RESUME_SKIP_COMPLETED attention 42`; concat/seed42 has passed the exact initial-tensor comparison and reached update100. Observation session98367 is the resumed live handle. Do not rerun the completed first fit or merge the source identities into the Docs SHA.

The first fit's scheduled final normal Joint is14.0625%, Fine0.5208%, raw20 47.1354%, NLL9.2144. Its serialized initial-state normal Joint was22.6563%, NLL9.5963 on the same seed. This is an unfavorable single-seed observation, not a final mode comparison or an independently validated effect. Other seeds and matched alternatives remain pending. `first_completed_fit.json` preserves this intermediate completed result without presenting it as the planned nine-fit aggregate.

All declared occlusion conditions will also be applied to every completed model. `evaluate_foundation_occlusion` refuses to start until nine final checkpoints are recorded, uses the same cached full-bank scorer, and records normal-condition replay mismatches explicitly before shifted results are interpreted. Its implementation source is `0d05d1e9b05b4a36e6993c4b4225682175d85283`; it has not yet run on the incomplete main batch. Full protocol remains ongoing.

## Final-architecture structural/runtime audit

All35 Track B tests pass, including explicit rigid world-frame invariance and new17×23 map forward for each foundation attention/concat/separable mode. This strengthens architecture coverage beyond inherited scratch tests. Actual completed attention seed42 checkpoint was traced on scene102344457, one query/all94,644 candidates: removing GT/target fields then rebuilding whitelisted inputs preserves score bits exactly; finite raster cells equal full bank count. Only scene cache, foundation-feature cache and checkpoint files were opened during load; forward opens only the explicitly allowed PyTorch opt_einsum lazy import. No candidate-RIR/B2/GEOQ reference path appears. This CPU trace covers this entrypoint/query; upstream RGB feature generation, other queries and future discriminator are outside its scope. Source `fbec4fbbae2e2350a5d82a53dc5bd3aafb15597e`; audit exited0 and is preserved in foundation_runtime_audit.json.

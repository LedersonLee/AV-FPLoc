# Foundation interaction: frozen/partial/joint encoder boundary control

Date2026-11-09. Track B. Status:all27 fits completed and matched-state verification passed. Evidence: Exploratory. [Completed results](completed_results.md): frozen/partial did not rescue the joint-model regression. Actual main source `062860067f82a5f6a0e6ce8411b5c43702ae0ab5`; smoke source `b50f10d1afc71396024a30e821f3409f5a41199e`.

## Hypothesis and competing explanations

[Study21](../2026-11-09-foundation-interaction/completed_comparison.md) found that jointly trained attention/concat/separable all underperformed their initial independent F1 while reducing NLL. Updating useful precursor representations may hurt MAP more than fitting the new interaction helps. Alternatively the interaction/scaling objective itself may disturb ranking even when precursor encoders are fixed. Comparing update boundaries at identical initialization/data/order helps distinguish these explanations. A frozen-arm gain would not independently prove the precise cause: effective trainable capacity and optimization differ by design.

## Fixed comparison

Reuse the completed joint reference with16 Train/6 reused Val scenes,64 queries each, .25m/36 yaw bins. Three architectures attention/concat/separable, seeds42/43/44. Add frozen and partial regimes, nine fits each. Each uses1,000 updates, batch8, AdamW lr3e-4/weight decay1e-4, skip the same1,500 initial sampling draws, identical precursor weights and loss coefficients. No epoch selection. Scheduled final checkpoints only. No Test observations.

- **Frozen:** fix visual adapter, acoustic encoder, visual FP tower/residual head and acoustic FP tower/residual head. Both score scales plus new V/A/Q projections, retrieval modules, gate and joint head learn.27,620 trainable parameters.
- **Partial:** same as frozen, but also train the last visual/acoustic descriptor Linear layers and final visual/acoustic FP residual Linear layers. Convolutional and earlier descriptor layers remain fixed.33,828 trainable parameters.
- **Joint reference:** all2,834,908 parameters train, as completed in study21.

The separately sourced upstream RGB backbone remains frozen in every condition. Thus “joint” means all local Track B components, not foundation-backbone tuning. All regimes have2,834,908 nominal parameters. Equal steps/sampling do not mean equal FLOPs, elapsed time or effective trainable capacity. Auxiliary losses and map-anchor terms remain in the objective even where they become constant for frozen components; they must not be interpreted as active gradients there.

Loss remains full-field soft CE +1000 visual ray MSE +10 acoustic ray MSE +1000 dual-map anchor. The initial acoustic map anchor is computed from the same original model/map, with no query GT. Every fit compares every initial tensor against the corresponding archived joint initial state. The new training code asserts all frozen parameter tensors are bitwise unchanged after all updates. Aggregation independently compares serialized initial/final states, includes buffers and refuses partial batches. AdamW receives the same full parameter list; tensors with no gradients receive no optimizer update.

## Initial runtime evidence (historical)

38 tests passed, including named-layer update boundaries, matched active counts across architectures and actual AdamW preservation of frozen parameters. Both real-data attention/seed42 smokes ran two updates and full six-scene evaluation, exit0. Initial states equal archived joint initialization; frozen parameters unchanged. Frozen smoke has zero precursor encoder gradients and nonzero new V/A retrieval gradients by update1. Partial smoke has nonzero last visual/acoustic descriptor gradients and frozen convolutional FP towers. FP residual heads are separately enabled; a zero tower gradient is expected and not evidence the entire FP path is frozen in partial mode.

Smoke output is retained in the adjacent protocol/summary JSONs. Smoke accuracy is not used to choose settings or make a regime-performance claim. Main frozen attention/seed42 was verified live at update100, with zero encoder gradients and active retrieval gradients. Main Docker08dd0bfc987c, host observation session55621. That session sequentially runs frozen nine fits then partial nine fits. An observation timeout is not a training failure; inspect that handle/Docker/logs before considering restart. No main-run failure has been observed at this snapshot.

## Analysis plan and scientific limits

Compare frozen−joint, partial−joint and partial−frozen within each architecture on normal, audio replacement and brightness. Use Joint≤1m/30°, R@1m, Fine≤.1m, raw20/diverse20 and nearest-grid NLL. Report every seed, paired six-scene bootstrap and actual time/peak memory. No seed/query pseudoreplication. All27 fits must be complete before final aggregate comparisons; evaluator `summarize_foundation_regimes` enforces initial equality and frozen final tensor equality. `summarize_interaction` now preserves actual regime and trainable parameter count in its registry instead of incorrectly labeling every run joint.

If fixed encoders preserve more accuracy, that supports retaining pretrained representations under this schedule; it does not establish attention superiority, generalization or a unique causal mechanism. If all regimes degrade, investigate new-head/scale updates and objective/ranking mismatch instead of attributing everything to representation drift. Severe occlusion and other controls must eventually cover the final selected protocol; brightness0.2 alone is not meaningful visual failure for this foundation representation. Existing Val reuse, uncertain upstream exposure, small cohort and multiple comparisons remain limitations.

## Reproduction and artifacts

External root `/home/jeongeonlee/track1/experiments/track_b_interaction_20261109`. Joint referencefoundation_matched_v1; new outputsfoundation_frozen_v1 andfoundation_partial_v1; separate2-step smokefoldersfoundation_frozen_smoke_v1/foundation_partial_smoke_v1. Checkpoints/features/raw predictions/logs remain external. Dockeravfploc:cu124-clean, RTX4090, no network, non-root UID, code/input caches read-only. No unrelated process was stopped.

Run `python -m track_b.scripts.train_foundation_interaction --cache /structure/cache --visual-run /foundation/learned_fp_w1000_anchor1000_v1 --acoustic-run /structure/supervised_v1 --visual-cache /foundation/features_v1 --lowlight-cache /foundation/features_lowlight_v3 --initial-reference /results/foundation_matched_v1 --out /results/foundation_frozen_v1 --regime frozen --steps 1000 --seeds 42 43 44 --modes attention concat separable --code-sha 062860067f82a5f6a0e6ce8411b5c43702ae0ab5`; repeat withregimepartial and freshpartialoutput. Mounts match study21. Smoke usedsteps2/seed42/attention and distinct outputs. Do not overwrite outputs.

Precursor/hash provenance is in frozen_main_protocol.json; partial protocol must retain the same sources. Training code SHA is separate from the later Docs commit. Clean-clone reproduction requires the recorded external source caches/checkpoints. Docs branchrefactor/hssd-two-branch-fusion. Overall goal remains unfinished; this preparatory report does not end the experiment or establish any scientific success.

## First completed fit — not final regime evidence

Frozen attention/seed42 completed1,000 updates with frozen weights bitwise unchanged. Joint15.104%, Fine.521%, raw20 47.396%, diverse20 75.521%, NLL9.22704. Same-seed joint reference was14.0625% Joint; initial independentF1 was22.6563%. A single seed does not establish a regime effect; it already shows that freezing these encoders does not automatically preserve the starting MAP accuracy. New heads and score scales still train. All other scheduled fits remain required, with no early-stopping decision from this result. Summary retained in first_frozen_fit.json.

## Completed comparison supersedes progress snapshots

All27 fits are complete. See completed_results.md for matched validation, quantitative uncertainty, preserved first-aggregation failure, checkpoint source provenance and mechanistic limits. The historical smoke/first-fit sections above are retained and are not the current completion status.

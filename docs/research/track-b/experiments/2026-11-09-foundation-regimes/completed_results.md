# Frozen/partial/joint: completed matched comparison

2026-11-09. All27 fits completed:3 encoder-update regimes×3 interaction architectures×3 seeds. Evidence is Exploratory on16 Train/6 reused Val scenes,64 observations/scene. The independently initialized F1 reference has Joint23.872%. All nine regime/architecture means remain below it. Frozen encoders do not rescue the regression under this1,000-update schedule.

## Observed outcomes

| Update regime | Architecture | Joint % | Fine % | Raw20 % | NLL |
|---|---|---:|---:|---:|---:|
| joint | attention | 16.753 | 0.694 | 49.826 | 9.219 |
| joint | concat | 18.229 | 1.302 | 49.913 | 9.230 |
| joint | separable | 15.538 | 0.955 | 48.438 | 9.201 |
| frozen | attention | 16.753 | 0.781 | 50.087 | 9.232 |
| frozen | concat | 17.101 | 0.868 | 50.434 | 9.249 |
| frozen | separable | 16.319 | 0.694 | 50.000 | 9.245 |
| partial | attention | 16.319 | 0.608 | 49.913 | 9.208 |
| partial | concat | 16.927 | 0.781 | 50.521 | 9.225 |
| partial | separable | 16.146 | 0.694 | 49.045 | 9.174 |

Frozen attention−joint attention Joint is0.000pp,95% paired scene bootstrap[-1.823,1.563]; identical aggregate accuracy does not mean identical predictions or weights. Partial attention−joint−0.434pp[-3.125,2.257]. Frozen concat−joint−1.128pp[-2.778,0.957], frozen separable−joint+0.781pp[-0.694,2.257]. No robust frozen/partial superiority is established. Partial concat−joint−1.302pp has upper interval essentially0 (floating residue−4.6e−16), which must not be called a strictly negative bound. Six scenes and uncorrected multiple comparisons do not support equivalence claims.

Relative to identical independent initialization, frozen attention/concat/separable lose7.118/6.771/7.552pp Joint; their95% upper bounds are−4.253/−4.425/−5.556pp. Partial loses7.552/6.944/7.726pp, upper bounds−5.903/−4.688/−6.076pp. These comparisons match actual query keys and original precursor hashes. Thus preserving frozen precursor encoders is insufficient to preserve MAP accuracy. Fine means also remain low (0.608–1.302% across all regimes), and raw20 remains around48–51%; complete normal/audio-shuffle/low-light results and NLL are retained in completed_per_seed.json and completed_paired.json. Brightness0.2 is not a validated severe visual failure for this upstream representation.

## Interpretation and discriminating next step

The hypothesis that encoder drift alone explains the joint-model regression is insufficient: the frozen condition reproduces the regression with all35 inactive tensors unchanged. In frozen training the new interaction projections/retrieval/gate/head and two score scales still update, so this intervention does not isolate scale drift from interaction residuals or the CE objective. Partial retains27 frozen tensors and updates final descriptor/residual layers, with similarly poor outcomes. These are intervention-supported statements about this update boundary; they do not establish a unique physical mechanism or generalize beyond this schedule/data.

A next discriminating control is to replay final frozen models with original score scales and with the learned interaction contribution removed, including their combination, while preserving all inputs/checkpoints. That can separate inference-time scale and interaction contributions; it cannot by itself prove the learning trajectory's causal mechanism. This report registers the motivation, not a completed control or improved method. Existing all-joint occlusion/grid/map controls remain separate; frozen/partial physical-shift and final-pipeline tests are still incomplete. No architecture is promoted based solely on these negative results.

## Matched settings and verification

All27 initial state dictionaries equal their corresponding archived joint initialization tensor-for-tensor. Every frozen/partial final state preserves all inactive tensors exactly (35 and27 respectively).1,000 final updates, batch8, learning rate3e−4, weight decay1e−4, the same1,500 skipped sampling draws, source caches/precursors and loss coefficients are matched. Every model has2,834,908 nominal parameters; frozen27,620/partial33,828 trainable parameters vs joint2,834,908. This does not equate effective capacity, FLOPs or wall-clock cost. The frozen upstream RGB encoder is unchanged in all arms; “joint” refers to local Track B components. Matching IDs/arms yields1,152 evaluation records per fit;27 fits total31,104 records. Seed observations are averaged inside each of six paired scenes for bootstrap uncertainty.

A first aggregation invocation exited1 because old joint records lack execution_code_sha. The empty partial output and failed_aggregation_v1.json are retained. No model reran and no numeric results were emitted from that invocation. Corrected aggregation reads the previously recorded joint experiment_registry.csv, verifies each checkpoint hash and then obtains its historical source SHA; no current Docs SHA substitutes for training provenance. Updated aggregation exited0, independently checking27 initial states, frozen tensors, settings and exact query/arm keys. Frozen/partial versus-initial comparisons also exited0. Failure preservation and the legacy-schema correction do not alter source predictions.

## Source and reproduction

Frozen/partial actual training source `062860067f82a5f6a0e6ce8411b5c43702ae0ab5`; original joint attention42 `6285d4b89e127f5b71479c49071dcc96a2ff908c`, remaining joint fits `18c0e3bf4297b9e0176ac3008710de06ec604da9`. Corrected aggregation source `39606eab0ab6a83467168fe808adf347b974f017`. Checkpoint-level source/hash/timing/peak allocation are in completed_validation.json. Run `python -m track_b.scripts.summarize_foundation_regimes --root <interaction-root> --out <fresh-directory> --joint-registry <study21/experiment_registry.csv>`. Run compare_foundation_to_initial for each frozen/partial directory against foundation/independent_fusion_anchor_v1 using fresh outputs. Aggregation used CPU Dockeravfploc:cu124-clean,2 threads; training RTX4090 under concurrent workload. External outputs foundation_regimes_registered_v1(failed)/v2(completed), foundation_frozen_initial_comparison_v1 and foundation_partial_initial_comparison_v1 are preserved. No raw weights/logs/predictions uploaded.

No Test access, independent validation or scientific completion declaration. Reused Val, uncertain upstream exposure, small scene count, fixed query sampling and multiple diagnostics remain limitations. Full scaled M1 comparison, OOF backend integration, physical shifts, complete costs and final independent evaluation remain within the original goal.

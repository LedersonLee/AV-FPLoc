# OOF independent-F1 discrimination: completed, mixed/negative geometry result

2026-11-09. All18 fold scorer fits and12 newly initialized classifier fits completed, exit0. No original Val observations were loaded by fold scorer training. Original six development scenes were used only after the classifier protocol was frozen, for scheduled final evaluation. Overall evidence remains **Exploratory** because method choices used this development cohort, upstream exposure is uncertain and only16 Train/6 Val scenes are covered. This is an independent-F1 backend experiment; no result is attributed to M1 attention.

## Observation

Each seed's OOF training set contains1,024 queries from16 scenes, with every scene scored by locally trained components fit on the other8 scenes. Each classifier evaluates384 queries from six original Val scenes. Seeds42/43/44 are averaged; uncertainty resamples six scenes, not1,152 nominal seed/query records. Full-grid scoring uses .25m/36 yaw bins. Original18-fold-fit preparation and checkpoint hashes are retained; all30 fits appear in experiment_registry.csv.

| Candidate policy / selection | Joint≤1m/30° % | R@1m % | Fine≤.1m % | Actual Top20 oracle % |
|---|---:|---:|---:|---:|
| MAP (both policies include it) |23.872|26.563|2.170|61.892 raw /84.375 diverse|
| Raw20 support-mass selection |17.969|19.878|.347|61.892|
| Raw20 new geometry classifier |25.521|27.951|.694|61.892|
| Raw20 score-statistics classifier |26.042|27.865|1.128|61.892|
| Diverse20 support-mass selection |10.069|13.108|.521|84.375|
| Diverse20 new geometry classifier |24.479|26.649|2.431|84.375|
| Diverse20 score-statistics classifier |29.167|31.597|3.038|84.375|

Diverse score-statistics versus MAP: +5.295pp, paired-scene95% bootstrap CI[3.125,7.899]. Geometry versus score-statistics on the same diverse candidates: −4.688pp CI[−6.944,−2.344]; Fine −.608pp CI[−1.128,−.087]. Raw geometry versus MAP: +1.649pp CI[.260,2.778], accompanied by Fine2.170→.694%. Raw geometry versus score-only: −.521pp CI[−3.038,1.649], inconclusive. Multiple exploratory comparisons are not multiplicity-corrected.

The OOF folds' raw oracle is49.479/57.943%; diverse oracle76.302/80.273%; MAP17.969/22.526%. Their lower performance than full-Train development proposals is a known proposal-distribution shift, not evidence of independent validation. All no-positive-candidate queries remain in fitting and reporting; no GT candidate is injected. For diverse score-statistics, oracle−final remains55.208pp despite the gain. High oracle alone does not guarantee recoverable information in these features.

Additional strict thresholds and continuous median translation/yaw are in discriminator_summary.json. Diverse score-statistics gives(.5m,15°)20.052%,(.2m,10°)6.424%, median translation3.775m and median yaw75°. Backend logits are BCE scores, **not calibrated full-bank posteriors**. Dense-field NLL is unchanged by reranking and is not rebranded as discriminator NLL. Classifier ECE, end-to-end latency, scaling and physical domain shifts remain unmeasured.

## Method and controls

The scorer is frozen independent .85V+.15A full-field fusion. Raw20 uses the actual score tensor topk; diverse20 greedily suppresses candidates within1m and30° in decreasing score order. The support baseline selects the candidate with maximum full-posterior mass inside that same relative1m/30° neighborhood. This is the precisely implemented support-mass baseline; it is not claimed identical to every historical Support-NMS implementation.

New features contain133 dimensions: predicted visual16/acoustic16 rays; candidate raw geometry16/confidence16; learned visual16/acoustic16 FP descriptors; visible visual residual16/acoustic residual16; and five score statistics (visual/acoustic/fusion score relative to their respective maxima, log neighborhood mass, normalized full-field entropy). They use no absolute pose, scene/query ID, GT rays/pose/distance, B2 proposal or reference RIR. GT is accessed only after candidate indices and feature tensors are fixed, to create BCE labels and evaluation distances. Relative map poses serve only neighborhood construction.

Both arms use a newly initialized133→64→32→1 GELU MLP, same seed/sampling,1,000 AdamW updates, lr.001, weight decay.0001, batch32 queries, candidate-mean BCE, all-negative queries retained. Training-only OOF mean/std normalization uses a .001 std floor. Score-statistics control zeros the first128 standardized dimensions. Nominal architecture is matched, but those zeroed inputs have no effective capacity; this is an input ablation, not proof of equal effective function capacity or equal information. No validation checkpoint selection or early stopping. Classifier fits ran on CPU; precursor scorers ran on RTX4090. Timings and parameter counts are in the registry/fits JSON. No pretrained GEOQ classifier/feature code was reused.

## Interpretation, alternatives and decision

**Supported observation:** score statistics learned from OOF candidate outcomes improve this development reranking task, particularly across diverse modes. Explicit geometry/observation features add no demonstrated benefit here and materially hurt diverse reranking relative to the score-only control.

**Unverified mechanisms:** the extra features may overfit scene-specific descriptor statistics, fail under half-Train/full-Train scorer shift, or be redundant after full-field scoring. This experiment does not distinguish these explanations. Geometry is still implicitly present in the base scores; score-only superiority does not show that floorplan geometry is useless. Likewise this does not establish acoustic causal benefit in the classifier; audio-removal/shuffle controls for this backend remain required.

**Decision:** retain score-only as a necessary backend baseline; reject a claimed positive geometry-conditioned contribution for this configuration. Do not select the winning arm as an independently validated final model. Next falsification should control scorer-training-size shift and repeat on separately selected scenes; matched M1 candidate provenance must be established before assigning a backend result to the proposed interaction model. Larger attention complexity is not justified by this result.

## Validation, provenance and reproduction

37 Track B tests passed before candidate extraction, including GT-free feature signature, rigid world-frame invariance of features/support and classifier gradient checks. Candidate generation, all12 classifier fits and aggregation exited0. Development MAP23.8715/raw61.8924/diverse84.375 reproduce the existing strong-F1 summaries. The tests are not a complete file-I/O audit of this new backend; broader raw-input inference tracing remains pending.

- Fold training source: `0aedb2ad9cb4690501c7f8ff0e648630241f983b`.
- Candidate feature/generation source: `e04336cb2fdbaa8ccde4d5d862840d85298e5490`; later checkout commits did not modify these running dependencies.
- Classifier source: `a8cbe16bee26188c1dec6312ca9e780977dcf90e`.
- Registration/paired-ablation source: `180b1c56e01b9bbdcd8887bf02c934c6ce26ef58`.

External root `/home/jeongeonlee/track1/experiments/track_b_oof_20261109`; original logs/checkpoints/candidate tensors and per-query predictions are retained there, not Git. `external_hashes.json` pins them. Source input caches/checkpoints are described in the preparation README and candidate protocol/hash files. Docker `avfploc:cu124-clean`, no network, non-root UID. Clean-clone reproduction requires these external inputs and is not claimed from Git alone.

Commands executed: `python -m track_b.scripts.cache_discriminator_candidates --cache /structure/cache --features /foundation/features_v1 --visual-run /results/fold0_visual_joint_v1 --acoustic-run /results/fold0_acoustic_v1 --role oof --out /results/fold0_candidates_v1`, repeated forfold1. Development uses `/foundation/learned_fp_w1000_anchor1000_v1`, `/structure/supervised_v1`, roledevelopment, outputdevelopment_candidates_v1. Then `python -m track_b.scripts.train_discriminator --root /results --out /results/discriminator_v1 --code-sha a8cbe16bee26188c1dec6312ca9e780977dcf90e`; register with `python -m track_b.scripts.summarize_discriminator --root /results --out /results/registered_v1`. All outputs require fresh paths.

Earlier smoke/preparation status is superseded by this completion record, not erased. No experiment in this stage failed to execute; unfavorable scientific outcomes above are retained. Main matched interaction, its robustness curve, broader scaling, independent evaluation, physical-shift compatibility and complete costs remain separate unfinished work.

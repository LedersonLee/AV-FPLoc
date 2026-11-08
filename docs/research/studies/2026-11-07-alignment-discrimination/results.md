> 역사적 연구 기록. 원본: `alignment_discrimination_20261107_2221/RESEARCH_REPORT.html`. 원본 SHA256은 [산출물 목록](artifacts.json)에 있습니다.
> 당시의 완료·재개 지시는 현재 실행 요청이 아닙니다. 최신 판정은 [현재 결론](../../current_findings.md)을 먼저 확인하세요.

# Acoustic domain alignment vs. localization discrimination

Completed controlled research, 7 November 2026. Train12 / reused development Val8 and Val11. No Test data.

**Distance reduction is not a sufficient success criterion.** Historical contrastive training improves acoustic retrieval relative to a reconstructed random initialization, while absolute same-pose cosine distance can increase. Positive-only consistency sharply contracts representations and weakens discrimination; tiny nonzero discriminative signals remain. Adding alignment to NCE sometimes improves acoustic retrieval relative to matched NCE continuation, but none of the tested matched downstream heads establishes an improvement over the existing strong GEOQ on the fixed cohort. Native-reference results are diagnostic, not deployable floorplan-only performance.

## 1. What the existing contrastive training actually optimizes

The actual pilot uses 6,387 Train12 query pairs and Val3 with 1,704 queries for checkpoint selection. Shared normalized 128D E/C/N encoder outputs are compared by dot product / temperature 0.07. Cross entropy selects the exact same-pose, same-yaw FP positive at slot 0 among eight candidates. There are four spatial-hard, two orientation-hard, and one spatial-random negatives; no in-batch or cross-scene negatives. Hard negatives already come from a frozen visual epoch-20 ray-likelihood model. Thus visually informed mining is existing functionality, not a newly invented improvement. Pose and scene metadata control pairing and mining, not neural inference input. Channel/yaw structure is retained; yaw invariance is not the objective.

Actual optimizer: AdamW 3e−4, weight decay 1e−4, batch 64, gradient clipping 1, BF16 and cosine schedule T_max=40. Training stopped at epoch 12; best was epoch 4, selected by Val3 loss. Epochs 5, 10 and 12 survive; no archived epoch-0 checkpoint was located. “Init” is reconstructed seed 42, not an authenticated historical pretraining checkpoint. The original best eight-candidate validation loss, Top1, Top5 and MRR replay exactly with current code/cache: 0.34637264, 86.6197%, 100%, 0.92486309. Some historical source hashes differ from current source, so this replay establishes functional compatibility, not recovery of the original bytes.

The standalone encoder cosine is distinct from B2’s exchanged acoustic scoring head. Adding a pure acoustic score to a retrained ranker does not replace the encoder inside frozen B2. GEOQ already receives audio-influenced fused scores and proposals and is not an audio-free baseline. [training_contract_replay.json](artifacts.json) (`training_contract_replay.json`) · [source_contract_audit.json](artifacts.json) (`source_contract_audit.json`) · [sampling_audit.json](artifacts.json) (`sampling_audit.json`)

## 2. Domain matrix and repeatability

Twelve Train scenes, two preselected queries per scene, 684 physical pose conditions, two domains and two independent-call repetitions: 2,736 renders. Reference candidates are the GT-free union of existing Top20 supports; exact query poses are used only for alignment diagnostics and are not inserted as candidate answers. All domain comparisons share pose/yaw/sensor/ray settings. Default materials are used (enableMaterials=False); semantic material effects are not identified. No explicit RLR seed is exposed; repeated calls are not proven independent random samples. Sequential mono measurements form the six-channel ring. Zero/weak responses remain in evaluation.

| Representation | Within N | Within FP | Cross distance | Wrong−positive gap | N→N % | FP→FP % | N→FP % | FP→N % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T | 0.00060 | 0.00044 | 0.00479 | 0.01464 | 41.67 | 41.67 | 22.92 | 13.54 |
| N | 0.00963 | 0.00862 | 0.01379 | 0.01501 | 23.96 | 33.33 | 19.79 | 20.83 |
| C | 0.05283 | 0.05506 | 0.08590 | 0.09946 | 27.08 | 22.92 | 26.04 | 18.75 |
| C_early | 0.05953 | 0.06289 | 0.09677 | 0.11090 | 23.96 | 21.88 | 26.04 | 18.75 |
| C_full | 0.04653 | 0.04755 | 0.07549 | 0.08817 | 26.04 | 22.92 | 22.92 | 19.79 |
| pre_ECN | 0.10135 | 0.08690 | 0.21676 | 0.15501 | 43.75 | 36.46 | 26.04 | 20.83 |
| init | 0.02306 | 0.02543 | 0.04541 | 0.04632 | 32.29 | 34.38 | 27.08 | 12.50 |
| best | 0.10187 | 0.08395 | 0.22949 | 0.50169 | 56.25 | 51.04 | 42.71 | 29.17 |
| epoch12 | 0.18083 | 0.16938 | 0.38873 | 0.43330 | 56.25 | 50.00 | 37.50 | 35.42 |
| nce_seed42 | 0.06068 | 0.06326 | 0.20048 | 0.43965 | 62.50 | 46.88 | 43.75 | 18.75 |
| align_seed42 | 0.00001 | 0.00001 | 0.00002 | 0.00002 | 29.17 | 32.29 | 26.04 | 19.79 |
| nce_align_seed42 | 0.03844 | 0.03937 | 0.14052 | 0.40047 | 56.25 | 44.79 | 41.67 | 20.83 |
| warm_nce_seed42 | 0.11532 | 0.09759 | 0.26685 | 0.47244 | 61.46 | 53.12 | 42.71 | 29.17 |
| warm_align_seed42 | 0.00003 | 0.00003 | 0.00006 | 0.00012 | 52.08 | 51.04 | 34.38 | 32.29 |
| warm_nce_align_seed42 | 0.07779 | 0.06526 | 0.17561 | 0.49095 | 54.17 | 54.17 | 50.00 | 37.50 |

These fixed20 success labels describe the inherited refined-output hypotheses; [matrix_analysis.json](artifacts.json) (`matrix_analysis.json`) also includes grid-pose success labels for the actually rendered reference positions, pooled reference-bank Top1/5/20, AUC, margin, GT probability mass, and exact/near-tie sensitivity. [domain_summary.json](evidence/domain_summary.json) provides scene bootstrap intervals. Repeat/seed observations are averaged within scene, not treated as independent samples. Raw descriptors, intermediate features and final embeddings are all retained. [cross_scene_analysis.json](artifacts.json) (`cross_scene_analysis.json`) compares cross-scene negatives only for globally shared historical encoders and raw features, never across independently trained OOF coordinate systems.

| Best encoder domain | Refined target Top1 % | Rendered grid Top1 % | Refined oracle20 % | Grid oracle20 % |
| --- | --- | --- | --- | --- |
| Native→Native | 56.25 | 56.25 | 100.00 | 100.00 |
| FP→FP | 51.04 | 51.04 | 100.00 | 100.00 |
| Native→FP | 42.71 | 42.71 | 100.00 | 100.00 |
| FP→Native | 29.17 | 29.17 | 100.00 | 100.00 |

| Native→FP representation | Exact positive vs other-scene negatives AUC % | Margin to hardest other-scene candidate |
| --- | --- | --- |
| T | 83.69 | -0.0018 |
| N | 90.13 | -0.0024 |
| C | 92.15 | 0.0169 |
| init | 96.50 | 0.0105 |
| best | 99.85 | 0.2727 |
| epoch12 | 98.40 | 0.1881 |

For the original best encoder, the paired Native-reference minus FP-reference Top1 difference is +13.54 percentage points, with exploratory scene-bootstrap 95% interval [-2.08, +30.21]. This interval includes zero. The 24-query diagnostic supports heterogeneous reference sensitivity, not a conclusive claim that domain mismatch is the dominant population-level bottleneck.

Cross-scene discrimination can be easier than within-scene pose discrimination and is not a substitute for it. Negative-scene and repeat multiplicity do not increase independent query count.

## 3. Historical trajectory: alignment must be evaluated relative to negatives

| Split | Checkpoint | Pool Top1 % | Top5 % | Top20 % | AUC % | Margin |
| --- | --- | --- | --- | --- | --- | --- |
| train | init | 8.20 | 26.82 | 53.65 | 76.32 | -0.0135 |
| train | best | 33.20 | 68.62 | 91.67 | 89.54 | -0.0352 |
| train | epoch5 | 31.25 | 65.89 | 93.62 | 88.93 | -0.0376 |
| train | epoch10 | 27.08 | 60.81 | 89.45 | 86.81 | -0.0602 |
| train | epoch12 | 27.60 | 59.64 | 85.42 | 85.75 | -0.0684 |
| val | init | 8.98 | 27.54 | 51.95 | 75.48 | -0.0153 |
| val | best | 29.10 | 62.70 | 87.50 | 88.66 | -0.0524 |
| val | epoch5 | 26.76 | 60.55 | 85.55 | 87.61 | -0.0617 |
| val | epoch10 | 21.48 | 52.15 | 79.69 | 85.03 | -0.0899 |
| val | epoch12 | 21.88 | 49.02 | 77.34 | 84.00 | -0.0978 |
| extra | init | 8.38 | 27.13 | 56.39 | 75.36 | -0.0128 |
| extra | best | 29.55 | 62.78 | 87.22 | 88.43 | -0.0513 |
| extra | epoch5 | 27.98 | 61.08 | 86.79 | 87.81 | -0.0535 |
| extra | epoch10 | 22.16 | 54.97 | 82.67 | 84.60 | -0.0799 |
| extra | epoch12 | 21.02 | 52.56 | 79.12 | 83.58 | -0.0889 |

The pooled bank is a GT-free union of Top20 candidates for 64 uniform queries per scene, roughly 1.5–1.7k reference indices per scene, not the full dense FP bank. Historical Train results include training exposure. Later checkpoints do not uniformly improve retrieval despite continued optimization. Positive/negative relative gaps and concentration distinguish useful separation from globally compressed embeddings. [checkpoint_bank_analysis.json](artifacts.json) (`checkpoint_bank_analysis.json`)

## 4. Matched objective controls and retention falsification

Fresh controls: NCE, positive-only (1−cos positive), and NCE + 1×positive-only, same architecture/init/order/eight identities/optimizer/two epochs. Four Train scene folds and full Train fit, two seeds: 30 models. Train9→held3 gives scene-held encoder evaluation; full Train models evaluate reused development scenes. FP32 pilot repeats NCE and positive-only on one held fold/seed. Warm-start controls load the same original best checkpoint, reset the optimizer equally, and train two more epochs under the three objectives and two seeds. Warm Train evaluation inherits original Train exposure; it is not scene-held representation validation.

| Start | Split | Objective | Pool Top1 % | Top20 % | Fixed20 Top1 % | Ambiguous pair % | FP variance |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fresh | train | nce | 21.55 | 79.56 | 30.86 | 71.74 | 0.803806 |
| fresh | train | align | 7.68 | 52.80 | 20.08 | 60.31 | 0.000047 |
| fresh | train | nce_align | 22.20 | 78.71 | 31.71 | 72.66 | 0.681869 |
| fresh | val | nce | 25.68 | 85.55 | 34.77 | 77.00 | 0.856171 |
| fresh | val | align | 7.32 | 51.76 | 21.68 | 62.17 | 0.000036 |
| fresh | val | nce_align | 25.59 | 85.64 | 36.18 | 77.08 | 0.731557 |
| fresh | extra | nce | 25.07 | 84.23 | 36.43 | 74.31 | 0.843925 |
| fresh | extra | align | 8.45 | 54.97 | 20.70 | 61.73 | 0.000032 |
| fresh | extra | nce_align | 26.78 | 84.16 | 37.64 | 75.33 | 0.712460 |
| warm | train | nce | 31.97 | 91.86 | 39.23 | 77.71 | 0.881720 |
| warm | train | align | 18.10 | 79.30 | 28.22 | 71.35 | 0.000243 |
| warm | train | nce_align | 34.18 | 92.71 | 42.45 | 78.38 | 0.831325 |
| warm | val | nce | 24.90 | 85.84 | 32.76 | 75.04 | 0.899023 |
| warm | val | align | 20.90 | 76.66 | 30.03 | 71.79 | 0.000254 |
| warm | val | nce_align | 27.64 | 86.13 | 36.28 | 77.85 | 0.851545 |
| warm | extra | nce | 26.78 | 85.65 | 38.25 | 74.22 | 0.874129 |
| warm | extra | align | 20.24 | 79.05 | 30.11 | 71.27 | 0.000229 |
| warm | extra | nce_align | 28.55 | 86.58 | 38.88 | 74.93 | 0.823215 |

Positive-only training is a deliberately discriminability-free ablation, not the existing contrastive objective. Its training slot-0 argmax can report 100% when all eight scores tie; tie-aware correctness is 12.5%. FP32 evaluation retains small nonzero signals, and FP32 training reproduces poor retrieval/low variance on the precision pilot, so neither “all information erased” nor “only BF16 caused it” is supported. [precision_bank_analysis.json](artifacts.json) (`precision_bank_analysis.json`) · [quantization_control.json](artifacts.json) (`quantization_control.json`) · [objective_paired_comparisons.json](artifacts.json) (`objective_paired_comparisons.json`) · [domain_summary.json](evidence/domain_summary.json) (per-query alignment/discrimination tradeoff quadrants).

## 5. Same stage with rigid objects removed

A separate four-scene/eight-query control performs 462 renders after removing all rigid objects before creating the audio sensor. Original architectural stage and poses remain; object counts before/after are asserted. This is not furniture-only semantic intervention: rigid objects may include other geometry. NativeNoObjects references are also diagnostic and unavailable in floorplan-only deployment.

| Representation | Reference | Top1 % | AUC % |
| --- | --- | --- | --- |
| T | Native | 50.00 | 74.99 |
| T | NativeNoObjects | 18.75 | 61.07 |
| T | FP | 40.62 | 67.34 |
| N | Native | 40.62 | 76.02 |
| N | NativeNoObjects | 34.38 | 69.87 |
| N | FP | 25.00 | 68.23 |
| C | Native | 31.25 | 77.28 |
| C | NativeNoObjects | 34.38 | 69.99 |
| C | FP | 31.25 | 72.95 |
| best | Native | 68.75 | 87.19 |
| best | NativeNoObjects | 25.00 | 77.93 |
| best | FP | 53.12 | 84.88 |
| warm_nce_seed42 | Native | 75.00 | 86.93 |
| warm_nce_seed42 | NativeNoObjects | 50.00 | 80.65 |
| warm_nce_seed42 | FP | 46.88 | 86.33 |
| warm_align_seed42 | Native | 65.62 | 88.42 |
| warm_align_seed42 | NativeNoObjects | 40.62 | 73.99 |
| warm_align_seed42 | FP | 68.75 | 81.33 |
| warm_nce_align_seed42 | Native | 81.25 | 87.26 |
| warm_nce_align_seed42 | NativeNoObjects | 59.38 | 83.87 |
| warm_nce_align_seed42 | FP | 62.50 | 85.87 |

The object-removal intervention does not monotonically recover floorplan discrimination. Its effect depends on representation and query. Native–FP discrepancy therefore cannot be equated to a single furniture effect; architectural geometry and reference suitability remain competing explanations. [object_analysis.json](artifacts.json) (`object_analysis.json`)

## 6. Does the acoustic change improve final localization?

Matched utility heads use original fixed20 candidates; nested B2 adapter-training candidates for head fitting; 64 uniform queries/scene; identical 172-column capacity, HistGradientBoosting 150 iterations / 15 leaves / min leaf 100 / L2 10, objective joint + 0.5×fine relative to candidate0. Two seeds, four head folds plus full fit. Base receives zero extra score slots; other arms receive eight pure acoustic score features. No GT enters features. Fixed switch thresholds are inherited (0 / 0.01). These experiments do not retrain B2 token exchange or evaluate end-to-end candidate extraction.

| Family | Arm | Joint % | Fine % | R@1m % | Δ strong GEOQ pp [95% CI] |
| --- | --- | --- | --- | --- | --- |
| fresh | original_GEOQ | 72.30 | 25.78 | 74.29 | +0.00 [+0.00, +0.00] |
| fresh | base | 69.89 | 24.64 | 72.30 | -2.41 [-3.91, -0.92] |
| fresh | best | 70.67 | 25.43 | 73.22 | -1.63 [-3.20, +0.07] |
| fresh | init | 70.67 | 25.21 | 72.94 | -1.63 [-3.06, -0.07] |
| fresh | nce | 70.60 | 25.36 | 73.01 | -1.70 [-3.12, -0.07] |
| fresh | align | 70.24 | 24.86 | 72.59 | -2.06 [-3.34, -0.92] |
| fresh | nce_align | 70.60 | 25.50 | 73.08 | -1.70 [-2.84, -0.50] |
| warm | original_GEOQ | 72.30 | 25.78 | 74.29 | +0.00 [+0.00, +0.00] |
| warm | base | 69.89 | 24.64 | 72.30 | -2.41 [-3.84, -0.92] |
| warm | best | 70.67 | 25.43 | 73.22 | -1.63 [-3.20, +0.00] |
| warm | warm_nce | 70.38 | 25.07 | 72.59 | -1.92 [-3.05, -0.85] |
| warm | warm_align | 70.81 | 25.64 | 73.15 | -1.49 [-2.70, -0.50] |
| warm | warm_nce_align | 70.67 | 25.28 | 73.08 | -1.63 [-3.41, +0.14] |

Original GEOQ is evaluated on exactly the same cohort and remains the strong reference. The matched base is weaker because its training cohort is smaller; beating that base is not a claimed GEOQ improvement. Original full-Val11 joint 71.818% and cohort joint 72.301% have different denominators. All tested variants fail to establish superiority over strong GEOQ. Query-level repair/harm, visual preservation and scene/seed results are in [discriminator_analysis.json](artifacts.json) (`discriminator_analysis.json`) and [warm_discriminator_analysis.json](artifacts.json) (`warm_discriminator_analysis.json`). Paired scene bootstrap intervals are exploratory and do not remove adaptive-selection bias.

## 7. Failure mechanisms and competing explanations

| Diagnostic group | Seed/repeat observations | Unique queries (overlapping groups) |
| --- | --- | --- |
| cross_fail_matched_success | 26 | 13 |
| both_fail | 29 | 13 |
| matched_fail_cross_success | 13 | 8 |
| audio_pool20_success_pool1_fail | 46 | 16 |
| small_distance_wrong_wins | 29 | 10 |
| large_distance_success | 22 | 9 |
| geoq_failure_with_coverage | 20 | 6 |
| has_visual_ambiguous_pair | 96 | 24 |

Matched-domain repairs coexist with both-domain failures and cross-domain successes when matched-domain fails. Small absolute positive distance can coexist with a higher-scoring wrong hypothesis; large distance can coexist with success. These cases establish reference sensitivity, not a unique physical cause of GEOQ error. Query-wise domain outcomes, positive/negative margins and GEOQ labels are retained in [domain_cases.json](artifacts.json) (`domain_cases.json`). Distance groups use the rendered Train-query median, only for diagnostics. Acoustic Top20-success/Top1-failure uses the explicitly limited reference bank.

Supported: acoustic evidence distinguishes many visually ambiguous pairs; relative positive/negative separation matters; positive-only contraction sacrifices useful ranking; scalar additions can be redundant with or inadequately used by GEOQ. Rejected as sufficient explanations: smaller positive distance guarantees better localization; all discrepancy is furniture; precision alone explains contraction; improved acoustic retrieval necessarily improves fusion. Still unresolved: whether a better candidate-conditioned learner, dynamically refreshed negatives, or physically richer reference simulation yields independent-scene gains. Existing static negatives overlap current candidate ambiguity imperfectly, but overlap audit alone is not causal evidence for a new mining policy.

## 8. Integrity, uncertainty and next investments

Val8/Val11 are reused development data. The visual upstream checkpoint has Val19 exposure; the acoustic best has Val3 selection exposure. Fresh encoders are scene-held for Train, but this does not erase upstream B2 exposure. Warm-start models inherit original training exposure. No independent real-world generalization is established; no Test set was used. All scene counts, query IDs, reference IDs, checkpoint hashes and inference-safe feature rules are retained.

Priority 1 — front end: preserve relative discrimination and audit current candidate-negative coverage; do not optimize absolute alignment alone. Priority 2 — cross-modal interaction: investigate residual errors conditional on strong GEOQ and reference reliability; scalar stacking tested here is insufficient. Priority 3 — back end: preserve GEOQ and improve only with paired repair/harm evidence against it, not a weakened matched control. Any next larger training run should preregister a candidate-conditioned or sampling contrast and evaluate independently frozen scenes. The present controls support reproducible knowledge rather than a deployable model upgrade.

## 9. Reproduction and artifacts

New controlled training comprises 38 encoder models and 90 utility heads; rendering comprises 3,198 RIR records. [resource_summary.json](artifacts.json) (`resource_summary.json`) records measured durations and their accounting limits; the user authorized local resources freely.

All new code/models/results are isolated in this experiment directory. Existing B2/GEOQ checkpoints and 4,927 earlier non-log artifacts were hash-checked unchanged; the user’s existing working-tree patch was preserved. See [verify_preservation_final.log](artifacts.json) (`verify_preservation_final.log`). Failures are retained: host python alias absent (python3 succeeds), and Docker /output sibling-path analysis failure (host analysis succeeds). No result is silently substituted for a failed run.

GPU scripts run with `bash run_script.sh SCRIPT.py`; numerical analysis scripts run using host `python3 SCRIPT.py` because they resolve neighboring experiment directories. Rendering requires the recorded SoundSpaces image and mounts. [REPRODUCE.sh](artifacts.json) (`REPRODUCE.sh`) supplies checked execution modes; [PROTOCOL.md](archive/PROTOCOL.md) records preregistered controls/extensions; [CONTINUATION_STATUS.json](artifacts.json) (`CONTINUATION_STATUS.json`) records execution state; [artifact_manifest.json](artifacts.json) (`artifact_manifest.json`) provides hashes. Preserve logs and receipts when resuming; inspect live renderer state before restarting. Render entrypoints skip existing checked receipts.

그림: Domain, objective and downstream comparisons — 외부 산출물 `research_summary.png` ([목록](artifacts.json)).

All numeric source files remain available alongside this report. No positive deployment improvement is claimed.

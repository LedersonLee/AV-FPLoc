> 역사적 연구 기록. 원본: `pairwise_acoustic_evidence_20261107_2100/REPRODUCE.md`. 원본 SHA256은 [산출물 목록](../artifacts.json)에 있습니다.
> 당시의 완료·재개 지시는 현재 실행 요청이 아닙니다. 최신 판정은 [현재 결론](../../../../current_findings.md)을 먼저 확인하세요.

# Reproduction

Run from `${WORKSPACE}`. Source datasets/checkpoints/prior experiments are read-only Docker mounts. New outputs go only to this experiment. Use the existing pinned local images `avfploc:cu124-clean` and `track1-soundspaces-echoloc:latest`; record image IDs in final verification. No network, no Test.

1. `python3 experiments/pairwise_acoustic_evidence_20261107_2100/verify_preservation.py`
2. Repeat the saved `repeat_plan.json` with `render_repeats.py` in the SoundSpaces image: CPU software EGL, `/opt/conda/envs/soundspaces/bin/python`, `LD_PRELOAD=/opt/conda/envs/soundspaces/lib/libstdc++.so.6`, source `${WORKSPACE}` and `/file2` read-only, this directory `/output` writable, workdir `/output`, CUDA_VISIBLE_DEVICES empty. Keep production acoustic settings in code. Existing successful renders are hash-checked and skipped; archive new outputs separately for a genuinely new stochastic replication.
3. `bash experiments/pairwise_acoustic_evidence_20261107_2100/run_script.sh repeat_features.py` then `repeat_ranking.py` through the same wrapper. Host Python3+NumPy: run `repeat_analysis.py`, `repeat_pooling.py`, `verify_repeats.py`.
4. Host Python3+NumPy: `pair_analysis.py --split train`, then `--split val`, `--split extra`; `pair_opportunity.py` for the same splits. Preserve `pair_protocol.json` and `pair_margin_protocol.json`; thresholds must not be selected on development outcomes.
5. Wrapper: `train_ranker.py --seed 42` and `--seed 43`; same for `train_pair.py` and `train_pair_broad.py`. First uses nine predeclared candidate arms. Pair scripts fit base/cue factorials with close-visual and mixed-pair training distributions, respectively. Completed `train_result.json` causes skipping; to rerun use an isolated output copy with models/results removed and required code/catalog/config intact. Do not overwrite source experiments.
6. Wrapper: `evaluate_models.py --seed 42 --arms C0_matched C1_candidate C2_pair C3_agreement C4_donor1 C4_donor2 C4_donor3 C5_basis C2_scalar P0_base P1_cues P0_broad P1_broad`; repeat seed43. These evaluate all registered arms, not development-selected winners.
7. Wrapper: `frozen_controls.py --seed 42 --split train` (then val/extra and seed43). Frozen extra-block interventions are distinct from retraining.
8. Host: `model_statistics.py --split train` then val/extra; `paired_statistics.py`; `environment_analysis.py`; `verify_baseline.py`; `verify_preservation.py`.
9. `build_report.py` on host; `plot_results.py` through wrapper (Matplotlib). Model/cache verification uses wrapper `final_verify.py`; manifest is generated last with `make_manifest.py` on host.

## Interpretation and artifacts

Train12 outer folds exclude held scenes from ranker fitting and use nested B2 adapter caches for training hypotheses. Upstream encoder/checkpoint exposure persists. Val8/Val11 are reused development sets. Thresholds are original seed42=0/seed43=.01. Model targets and active feature capacity differ only in explicitly named comparisons. P0/P1 comparisons separate objective/aggregation changes from cue additions. GT-free pair construction uses scores/coordinates/yaws, never labels. Labels provide training targets and evaluation only.

Pair metrics aggregate within query before scene/seed reporting; pair counts do not increase independent sample size. Scene-cluster bootstrap intervals average seeds first and do not correct exploration/multiple comparisons. No Test is accessed.

Training logs include the initial missing-`cache_access` import failure before any fitted model; dependency was copied from preserved source and subsequent fits completed. Repeat analysis initially lacked host scikit-learn; equivalent explicit Mann–Whitney AUC computation removes that dependency. These failures did not produce reported results.

Raw repeat manifests record SHA256, exact source/registry/stage, object count, source pose, renderer settings and elapsed time. Fresh distinct hashes are not proof of independent renderer RNG streams. Model checkpoints, thresholds, per-query scores, scene/group metrics, repair/harm cases, code, final manifest and report are retained in this directory.

## Same-pose domain follow-up

Use `domain_pair_plan.json` and `render_domain_pairs.py` with exactly the SoundSpaces container flags above. It generates32additional FP renders at8nativeObserved source/receiverposes; originals are not overwritten. Run wrapper `domain_pair_features.py`, then host `analyze_domain_pairs.py`. This contrast controls physicalpose, not allscenegeometry/material/furnishing factors. Across the main and follow-up repeat experiments there are192fresh renders and240feature records (including originalwaveform records, some reused across conditions). Run host `verify_alignment.py` to compare40originalwaveform embeddings to the priorcachedpureembeddings.

`pair_anchor.py --seed 42 --split train` and seed43 (wrapper) implement the frozen candidate0-only aggregation falsification. This branch was stopped after worse TrainOOF results; no development aggregation selection was performed. `plot_cases.py` renders explicitly selected repair/harm examples, with headings omitted to avoid implying a plotting convention.

Run host `decision_dependence.py` for C3repair/harm sensitivity to frozen removals/donors. `container_images.json` pins local image IDs; `final_verification.json` records library versions. The prior visualencoder monitored Val19 (Val8+Val11), as documented in the preserved coverage/discrimination reports; downstream OOF does not remove that upstream history.

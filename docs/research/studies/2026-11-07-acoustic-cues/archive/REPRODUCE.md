> 역사적 연구 기록. 원본: `acoustic_cues_20261107_1830/REPRODUCE.md`. 원본 SHA256은 [산출물 목록](../artifacts.json)에 있습니다.
> 당시의 완료·재개 지시는 현재 실행 요청이 아닙니다. 최신 판정은 [현재 결론](../../../current_findings.md)을 먼저 확인하세요.

# Reproduce acoustic evidence research

Run from `${WORKSPACE}`. Research output is `experiments/acoustic_cues_20261107_1830`; earlier experiment directories, `AV-FPLoc` and `/file2`/`/ssd1` are inputs. `run_script.sh` mounts inputs read-only and only the current research directory read-write. It uses the existing `avfploc:cu124-clean` image and existing sklearn dependencies. Do not overwrite prior experiment outputs. Completed per-scene computations generally skip existing verified outputs; a true fresh reproduction should use a copied output directory with the `study.py` output path updated consistently, not delete evidence.

## Data and signal audit

```bash
python3 experiments/acoustic_cues_20261107_1830/data_catalog.py
python3 experiments/acoustic_cues_20261107_1830/raw_audit.py
python3 experiments/acoustic_cues_20261107_1830/cue_features.py
python3 experiments/acoustic_cues_20261107_1830/raw_features.py
bash experiments/acoustic_cues_20261107_1830/run_script.sh verify_physics.py
python3 experiments/acoustic_cues_20261107_1830/verify_raw_subset.py
```

Do not use the abandoned `data_index.py` as the authoritative catalog. The catalog hard-checks that selected scenes do not belong to Test. Raw descriptors use64 deterministic query rows/scene and their original fixed20 union; no GT-driven sampling. The raw hash pass verifies only those actually analyzed waveforms, while the existence audit covers the complete indexed corpus.

## Frozen encoder interventions and corrected OOF

```bash
bash experiments/acoustic_cues_20261107_1830/run_script.sh query_screen.py
bash experiments/acoustic_cues_20261107_1830/run_script.sh evaluate_screen.py --seed 42 --split train
bash experiments/acoustic_cues_20261107_1830/run_script.sh correct_train_oof.py --seed 42
bash experiments/acoustic_cues_20261107_1830/run_script.sh evaluate_screen.py --seed 43 --split train
bash experiments/acoustic_cues_20261107_1830/run_script.sh correct_train_oof.py --seed 43
```

For each seed42/43 and split `val`/`extra`, run:

```bash
bash experiments/acoustic_cues_20261107_1830/run_script.sh evaluate_screen.py --seed 42 --split val --modes baseline no_time no_C no_freq early_EN late_EN
```

Substitute seed and split explicitly. Train analysis must read `oof_screen_seed*/train`, never the provisional Train GEOQ outputs in `screen_seed*/train`. `evaluate_screen_initial.py` archives the actual initial running Train source; on-disk code provenance caveat is documented in METHODS.md. `confirmation_reschedule.json` records stopping only obsolete orchestration shells while leaving correction workers running; independent development splits were then evaluated concurrently via `confirm_split.sh`.

For each split `train`/`val`/`extra`:

```bash
bash experiments/acoustic_cues_20261107_1830/run_script.sh paired_reference.py --split train
bash experiments/acoustic_cues_20261107_1830/run_script.sh analyze_paired.py --split train
```

## Matched retraining

For each arm `all no_C no_T no_N no_TN T N F C no_TNC` and each split `train val extra`, use:

```bash
bash experiments/acoustic_cues_20261107_1830/run_script.sh metric_cues.py --arm all --split train
```

Ridge alpha10, fold assignment and signal dimensions are fixed in source and model receipts. No development tuning is applied. The fully trained models and all four fold models are saved under `cue_metric/ARM`.

```bash
python3 experiments/acoustic_cues_20261107_1830/coherence_windows.py
bash experiments/acoustic_cues_20261107_1830/run_script.sh verify_late.py
bash experiments/acoustic_cues_20261107_1830/run_script.sh check_coherence_vector.py
```

Additional Train-only quantization falsification: run `late_quantization_control.py` through `run_script.sh` after C_late fold models and the raw subset exist.

For each arm `C_early C_full C_late C_real C_imag C_mag` and each split, run `coherence_metric.py --arm ARM --split SPLIT` through `run_script.sh`. Same nominal456 coordinates; the first three arms each have48 active dimensions. Protocol snapshots precede development evaluation.

For seeds42/43:

```bash
bash experiments/acoustic_cues_20261107_1830/run_script.sh cue_ranker.py --seed 42 --arms GEOQ_replay CUES no_C no_T no_N no_TN basis donor
bash experiments/acoustic_cues_20261107_1830/run_script.sh evaluate_cue_ranker.py --seed 42
```

The retained GEOQ input is acoustic-bearing; these are conditional added-block controls. Previous nested B2 training caches and fixed thresholds are required. Saved `cue_ranker_ARM_seedSEED/model.joblib` includes model/scaling/arm; `fold0..3.joblib` are held-scene fits. No final production model is replaced.

## Probes and simulator falsification

Run `cue_probes.py`, `geometry_probe.py`, `coherence_geometry_probe.py`, `raw_analysis.py` and `analyze_repeat.py` through `run_script.sh`. Run `raw_descriptives.py`, `plot_cases.py`, `plot_failure_cases.py` on host Python.

The repeated render uses a separate image and CPU software rendering. Existing dataset outputs are read-only:

```bash
docker run --rm --network none --cpus 4 --memory 8g --user 1000:1000 \
  --entrypoint /opt/conda/envs/soundspaces/bin/python \
  -e CUDA_VISIBLE_DEVICES=-1 -e EGL_PLATFORM=surfaceless -e LIBGL_ALWAYS_SOFTWARE=1 \
  -e LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  -v ${WORKSPACE}:${WORKSPACE}:ro -v /file2:/file2:ro \
  -v ${WORKSPACE}/experiments/acoustic_cues_20261107_1830:/output:rw \
  -w /output track1-soundspaces-echoloc:latest /output/repeat_render.py
```

Fresh renders are stochastic; byte-identical reproduction is not expected. Existing twelve raw repeats, metadata and analysis are retained. No explicit random seed is exposed by the diagnostic renderer. Rendering-order comparisons are confounded by new stochastic samples, so they are not isolated causal order interventions.

## Analysis, confidence intervals and report

Wait until all16 metric arms have31 scene outputs, both corrected Train screens have12, and both development screen seeds have8+11. Do not aggregate incomplete scenes as final evidence.

```bash
OPENBLAS_NUM_THREADS=2 python3 experiments/acoustic_cues_20261107_1830/analyze_metric.py
OPENBLAS_NUM_THREADS=2 python3 experiments/acoustic_cues_20261107_1830/metric_stats.py
OPENBLAS_NUM_THREADS=2 python3 experiments/acoustic_cues_20261107_1830/metric_comparisons.py
OPENBLAS_NUM_THREADS=2 python3 experiments/acoustic_cues_20261107_1830/evidence_stats.py
OPENBLAS_NUM_THREADS=2 python3 experiments/acoustic_cues_20261107_1830/paired_stats.py
OPENBLAS_NUM_THREADS=2 python3 experiments/acoustic_cues_20261107_1830/analyze_screen.py --split train --require-complete
OPENBLAS_NUM_THREADS=2 python3 experiments/acoustic_cues_20261107_1830/analyze_screen.py --split val --require-complete
OPENBLAS_NUM_THREADS=2 python3 experiments/acoustic_cues_20261107_1830/analyze_screen.py --split extra --require-complete
OPENBLAS_NUM_THREADS=2 python3 experiments/acoustic_cues_20261107_1830/screen_stats.py
OPENBLAS_NUM_THREADS=2 python3 experiments/acoustic_cues_20261107_1830/group_signal_stats.py
OPENBLAS_NUM_THREADS=2 python3 experiments/acoustic_cues_20261107_1830/case_evidence.py
python3 experiments/acoustic_cues_20261107_1830/verify_preservation.py
OPENBLAS_NUM_THREADS=2 python3 experiments/acoustic_cues_20261107_1830/baseline_branches.py
OPENBLAS_NUM_THREADS=2 python3 experiments/acoustic_cues_20261107_1830/environment_evidence.py
bash experiments/acoustic_cues_20261107_1830/run_script.sh verify_metric_replay.py
python3 experiments/acoustic_cues_20261107_1830/final_verify.py
python3 experiments/acoustic_cues_20261107_1830/build_report.py
```

Numeric evidence JSONs include per-scene/seed/group values, paired scene bootstrap summaries and count definitions. Repair/harm counts are query-seed pairs, not independent observations. AUC is conditional on mixed correct/incorrect candidate sets. Scientific figures are standalone PNG/PDF. Final verification and artifact manifest close the evidence trail after all workers finish; the manifest does not hash itself.

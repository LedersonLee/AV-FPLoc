# 재현과 원본 근거

원본 연구 폴더: `${WORKSPACE}/experiments/acoustic_cues_20261107_1830`. `${WORKSPACE}`는 로컬 track1 루트, `${HSSD_ROOT}`/`${HSSD_SSD_ROOT}`는 기존 HSSD 저장소를 뜻한다.

- [원본 재현 문서: REPRODUCE.md](archive/REPRODUCE.md)

## 원본 실행·분석 코드

- `analyze_metric.py`
- `analyze_paired.py`
- `analyze_repeat.py`
- `analyze_screen.py`
- `baseline_branches.py`
- `build_report.py`
- `cache_access.py`
- `case_evidence.py`
- `check_coherence_vector.py`
- `coherence_geometry_probe.py`
- `coherence_metric.py`
- `coherence_windows.py`
- `confirm_after_train.sh`
- `confirm_coherence.sh`
- `confirm_split.sh`
- `correct_train_oof.py`
- `cue_features.py`
- `cue_probes.py`
- `cue_ranker.py`
- `data_catalog.py`
- `data_index.py`
- `environment_evidence.py`
- `evaluate_cue_ranker.py`
- `evaluate_screen.py`
- `evaluate_screen_initial.py`
- `evidence_stats.py`
- `final_verify.py`
- `finish_analysis.sh`
- `fixed_features.py`
- `geometry_probe.py`
- `group_signal_stats.py`
- `late_quantization_control.py`
- `metric_comparisons.py`
- `metric_cues.py`
- `metric_stats.py`
- `paired_reference.py`
- `paired_stats.py`
- `plot_cases.py`
- `plot_failure_cases.py`
- `query_screen.py`
- `raw_analysis.py`
- `raw_audit.py`
- `raw_descriptives.py`
- `raw_features.py`
- `repeat_render.py`
- `run_script.sh`
- `runtime_config.py`
- `screen_stats.py`
- `study.py`
- `verify_late.py`
- `verify_metric_replay.py`
- `verify_physics.py`
- `verify_preservation.py`
- `verify_raw_subset.py`

전체 코드는 기존 외부 연구 폴더에 보존한다. Git에는 코드 위치·SHA256 및 선택된 결과를 기록했다. 원본 checkpoint/cache와 과거 연구 의존성이 필요하며 이 문서 저장은 코드의 본 라이브러리 이관을 의미하지 않는다.

## 검증 경계

[artifacts.json](artifacts.json)의 `sha256`은 변환 전 원본 해시, `git_copy_sha256`은 경로 치환/Markdown 변환 후 Git 사본의 해시다. 기존 원본 manifest는 중첩 checkpoint·예측·로그를 추적한다. 원본 디렉터리를 덮어쓰지 말고 별도 replica에서 재현한다.

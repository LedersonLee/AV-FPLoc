# 재현과 원본 근거

원본 연구 폴더: `${WORKSPACE}/experiments/hssd_scaling_20261108_1332`. `${WORKSPACE}`는 로컬 track1 루트, `${HSSD_ROOT}`/`${HSSD_SSD_ROOT}`는 기존 HSSD 저장소를 뜻한다.

- [원본 재현 문서: REPRODUCE.md](archive/REPRODUCE.md)

## 원본 실행·분석 코드

- `adapt_b2.py`
- `adapt_visual.py`
- `ambiguity_gate.py`
- `analyze_expanded.py`
- `analyze_full.py`
- `analyze_oof.py`
- `analyze_recipe_controls.py`
- `analyze_results.py`
- `analyze_scene_balanced.py`
- `audit_acoustic_geometry.py`
- `audit_assets.py`
- `audit_objects.py`
- `bootstrap_replica.py`
- `build_candidates.py`
- `build_full_candidates.py`
- `build_report.py`
- `build_training_candidates_seed42.py`
- `build_training_candidates_seed43.py`
- `build_training_full_candidates.py`
- `cost_audit.py`
- `estimate_future_cost.py`
- `evaluate_adapters.py`
- `evaluate_expanded.py`
- `evaluate_full.py`
- `evaluate_heads.py`
- `evaluate_oof.py`
- `evaluate_scene_balanced.py`
- `evidence_metrics.py`
- `expanded_report_section.py`
- `final_verify.py`
- `floorplan_check.py`
- `plot_dataset.py`
- `plot_expanded.py`
- `plot_oof.py`
- `plot_recipe_controls.py`
- `plot_results.py`
- `prepare_full_inputs.py`
- `prepare_inputs.py`
- `prepare_oof_alignment.py`
- `prepare_training_full_inputs.py`
- `query_alignment.py`
- `recipe_report_section.py`
- `render_known_pose.py`
- `replay_oof_checkpoints.py`
- `replay_recipe_models.py`
- `reproduce.sh`
- `reuse_full_training_inputs.py`
- `run_build_training_candidates_seed42.sh`
- `run_build_training_candidates_seed43.sh`
- `run_build_training_full_candidates.sh`
- `run_evaluation.sh`
- `run_expanded_evaluation.sh`
- `run_expanded_seed42.sh`
- `run_expanded_seed43.sh`
- `run_expanded_training.sh`
- `run_full_evaluation.sh`
- `run_oof_evaluation.sh`
- `run_oof_replay.sh`
- `run_oof_seed42.sh`
- `run_oof_seed43.sh`
- `run_prepare_training_full_inputs.sh`
- `run_scene_balanced.sh`
- `run_scene_balanced_evaluation.sh`
- `run_training.sh`
- `run_verify_inference.sh`
- `source_provenance.py`
- `summarize_resources.py`
- `supplementary_analysis.py`
- `train_expanded.py`
- `train_expanded_seed42.py`
- `train_expanded_seed43.py`
- `train_oof_aligned.py`
- `train_oof_seed42.py`
- `train_oof_seed43.py`
- `train_scaling.py`
- `train_scene_balanced.py`
- `verify_expanded.py`
- `verify_external.py`
- `verify_full_training_inputs.py`
- `verify_inference.py`
- `verify_oof.py`
- `verify_training.py`

전체 코드는 기존 외부 연구 폴더에 보존한다. Git에는 코드 위치·SHA256 및 선택된 결과를 기록했다. 원본 checkpoint/cache와 과거 연구 의존성이 필요하며 이 문서 저장은 코드의 본 라이브러리 이관을 의미하지 않는다.

## 검증 경계

[artifacts.json](artifacts.json)의 `sha256`은 변환 전 원본 해시, `git_copy_sha256`은 경로 치환/Markdown 변환 후 Git 사본의 해시다. 기존 원본 manifest는 중첩 checkpoint·예측·로그를 추적한다. 원본 디렉터리를 덮어쓰지 말고 별도 replica에서 재현한다.

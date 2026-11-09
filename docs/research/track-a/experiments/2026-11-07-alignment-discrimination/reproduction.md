# 재현과 원본 근거

원본 연구 폴더: `${WORKSPACE}/experiments/alignment_discrimination_20261107_2221`. `${WORKSPACE}`는 로컬 track1 루트, `${HSSD_ROOT}`/`${HSSD_SSD_ROOT}`는 기존 HSSD 저장소를 뜻한다.

별도 원본 재현 Markdown은 없다. 아래 실행/분석 스크립트, 프로토콜 및 결과 문서를 함께 사용한다. 이 Git 문서만으로 완전한 실행 환경이 복원된다고 주장하지 않는다.

## 원본 실행·분석 코드

- `REPRODUCE.sh`
- `analyze_checkpoint_banks.py`
- `analyze_checkpoint_banks_initial.py`
- `analyze_cross_scene.py`
- `analyze_discriminator.py`
- `analyze_domain_cases.py`
- `analyze_matrix.py`
- `analyze_matrix_initial.py`
- `analyze_object.py`
- `analyze_objective_banks.py`
- `analyze_precision_banks.py`
- `analyze_warm_discriminator.py`
- `analyze_warmstart_banks.py`
- `audit_sampling.py`
- `build_ranker_features.py`
- `build_warm_ranker.py`
- `cache_access.py`
- `checkpoint_audit.py`
- `compare_objectives.py`
- `encode_checkpoint_banks.py`
- `encode_matrix.py`
- `encode_matrix_initial.py`
- `encode_object.py`
- `encode_objective_banks.py`
- `encode_precision_banks.py`
- `encode_warmstart_banks.py`
- `expand_matrix_plan.py`
- `extension_repeat_plan.py`
- `extract_matrix.py`
- `extract_matrix_initial.py`
- `extract_object.py`
- `finish_matrix_analysis.py`
- `make_manifest.py`
- `make_report.py`
- `matrix_plan.py`
- `metrics.py`
- `quantization_control.py`
- `render_matrix.py`
- `render_matrix_extension1.py`
- `render_matrix_extension2.py`
- `render_object_removal.py`
- `renderer_capabilities.py`
- `replay_training_contract.py`
- `resource_summary.py`
- `run_script.sh`
- `study.py`
- `summarize_domain.py`
- `train_discriminator.py`
- `train_objective.py`
- `train_objective_fp32.py`
- `train_objective_v1.py`
- `train_warm_discriminator.py`
- `train_warmstart.py`
- `verify_final.py`
- `verify_preservation.py`
- `watch_followups.py`
- `watch_matrix.py`

전체 코드는 기존 외부 연구 폴더에 보존한다. Git에는 코드 위치·SHA256 및 선택된 결과를 기록했다. 원본 checkpoint/cache와 과거 연구 의존성이 필요하며 이 문서 저장은 코드의 본 라이브러리 이관을 의미하지 않는다.

## 검증 경계

[artifacts.json](artifacts.json)의 `sha256`은 변환 전 원본 해시, `git_copy_sha256`은 경로 치환/Markdown 변환 후 Git 사본의 해시다. 기존 원본 manifest는 중첩 checkpoint·예측·로그를 추적한다. 원본 디렉터리를 덮어쓰지 말고 별도 replica에서 재현한다.

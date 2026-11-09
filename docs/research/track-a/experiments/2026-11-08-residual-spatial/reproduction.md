# 재현과 원본 근거

원본 연구 폴더: `${WORKSPACE}/experiments/residual_spatial_20261108_1144`. `${WORKSPACE}`는 로컬 track1 루트, `${HSSD_ROOT}`/`${HSSD_SSD_ROOT}`는 기존 HSSD 저장소를 뜻한다.

별도 원본 재현 Markdown은 없다. 아래 실행/분석 스크립트, 프로토콜 및 결과 문서를 함께 사용한다. 이 Git 문서만으로 완전한 실행 환경이 복원된다고 주장하지 않는다.

## 원본 실행·분석 코드

- `REPRODUCE.sh`
- `actual_spatial_audit.py`
- `aggregate_results.py`
- `analyze_neural.py`
- `analyze_probes.py`
- `audit_requirements.py`
- `benchmark.py`
- `build_report.py`
- `case_analysis.py`
- `common_counterfactual.py`
- `common_frame.py`
- `common_spatial_pilot.py`
- `early_stopping.py`
- `evaluate_discrimination.py`
- `evaluate_reference_gate.py`
- `extract_reference_tokens.py`
- `extract_reference_tokens_full.py`
- `final_verify.py`
- `finalize_analysis.py`
- `manifest.py`
- `neural_capacity_full.py`
- `neural_full.py`
- `neural_pilot.py`
- `neural_queryonly.py`
- `neural_retrain.py`
- `probe.py`
- `probe_full.py`
- `probe_raw.py`
- `reference_probe.py`
- `reference_probe_full.py`
- `spatial_full.py`
- `spatial_pilot.py`
- `spatial_validity.py`
- `token_probe.py`
- `verify_additional_sources.py`
- `verify_capacity.py`
- `verify_feature_receipts.py`
- `verify_inference.py`
- `verify_sources.py`

전체 코드는 기존 외부 연구 폴더에 보존한다. Git에는 코드 위치·SHA256 및 선택된 결과를 기록했다. 원본 checkpoint/cache와 과거 연구 의존성이 필요하며 이 문서 저장은 코드의 본 라이브러리 이관을 의미하지 않는다.

## 검증 경계

[artifacts.json](artifacts.json)의 `sha256`은 변환 전 원본 해시, `git_copy_sha256`은 경로 치환/Markdown 변환 후 Git 사본의 해시다. 기존 원본 manifest는 중첩 checkpoint·예측·로그를 추적한다. 원본 디렉터리를 덮어쓰지 말고 별도 replica에서 재현한다.

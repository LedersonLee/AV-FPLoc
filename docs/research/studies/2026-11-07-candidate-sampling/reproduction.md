# 재현과 원본 근거

원본 연구 폴더: `${WORKSPACE}/experiments/candidate_sampling_20261107_2337`. `${WORKSPACE}`는 로컬 track1 루트, `${HSSD_ROOT}`/`${HSSD_SSD_ROOT}`는 기존 HSSD 저장소를 뜻한다.

별도 원본 재현 Markdown은 없다. 아래 실행/분석 스크립트, 프로토콜 및 결과 문서를 함께 사용한다. 이 Git 문서만으로 완전한 실행 환경이 복원된다고 주장하지 않는다.

## 원본 실행·분석 코드

- `REPRODUCE.sh`
- `analyze_audit.py`
- `analyze_balanced.py`
- `analyze_discriminator_all.py`
- `analyze_discriminator_primary.py`
- `analyze_domain_primary.py`
- `analyze_donor_contrasts.py`
- `analyze_factorial.py`
- `analyze_sampling_banks.py`
- `analyze_subgroups.py`
- `analyze_subgroups_all.py`
- `analyze_warm_sampling.py`
- `audit_competition.py`
- `audit_held_encoder.py`
- `audit_modal_pools.py`
- `audit_posterior_support.py`
- `audit_score_fields.py`
- `build_discriminator_features.py`
- `cache_access.py`
- `checkpoint_audit.py`
- `domain_interactions.py`
- `domain_reference_cases.py`
- `encode_checkpoint_banks.py`
- `encode_domain_alternatives.py`
- `encode_domain_primary.py`
- `evaluate_balanced.py`
- `evaluate_factorial.py`
- `evaluate_sampling.py`
- `evaluate_warm_sampling.py`
- `finish_analysis.py`
- `freeze_manifest.py`
- `freeze_subgroups.py`
- `inspect_manifest.py`
- `make_report.py`
- `materialize_bank.py`
- `materialize_objective_extra.py`
- `metrics.py`
- `objective_dataset.py`
- `policy_dataset.py`
- `policy_dataset_nfs.py`
- `prepare_balanced.py`
- `prepare_objective_policies.py`
- `prepare_policies.py`
- `replay_mining.py`
- `replay_training_contract.py`
- `run_script.sh`
- `study.py`
- `summarize_contrasts.py`
- `summarize_domain_primary.py`
- `summarize_posterior_support.py`
- `train_balanced.py`
- `train_discriminator_alternatives.py`
- `train_discriminator_primary.py`
- `train_objectives.py`
- `train_sampling.py`
- `train_sampling_fresh_v1.py`
- `verify_dataset.py`
- `verify_final.py`
- `verify_objective_dataset.py`
- `verify_preservation.py`

전체 코드는 기존 외부 연구 폴더에 보존한다. Git에는 코드 위치·SHA256 및 선택된 결과를 기록했다. 원본 checkpoint/cache와 과거 연구 의존성이 필요하며 이 문서 저장은 코드의 본 라이브러리 이관을 의미하지 않는다.

## 검증 경계

[artifacts.json](artifacts.json)의 `sha256`은 변환 전 원본 해시, `git_copy_sha256`은 경로 치환/Markdown 변환 후 Git 사본의 해시다. 기존 원본 manifest는 중첩 checkpoint·예측·로그를 추적한다. 원본 디렉터리를 덮어쓰지 말고 별도 replica에서 재현한다.

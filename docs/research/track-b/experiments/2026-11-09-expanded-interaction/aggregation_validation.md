# 확장 cohort 집계 준비 검증

- 상태: 집계 회귀 검증 완료, 112-Train/19-Val 주 실험은 진행 중. 신규 성능 결론 없음.
- 집계 코드 SHA: `e250aa59a0a2207b7629c4ff17d176b453a83965`.
- 변경: scene/query 수와 bootstrap 설명을 protocol에서 계산한다. 각 fit에서 세 평가 arm의 query ID가 같고, scene별 query 수가 설정과 같으며, 모든 mode/seed의 query ID 집합이 동일함을 검사한다. 질의 ID가 연속 정수라고 가정하지 않는다.
- 검증: 완료된 16-Train/6-Val, 9-fit 원본을 재집계했다. aggregate, per-seed summary, initialization validation은 기존 `foundation_registered_v1`과 JSON 전체 일치. 24개 paired bootstrap 비교도 설명 문자열을 제외하고 일치. 레지스트리 9행 모두 scene22/train1024/val384 확인. Python 구문 검사 통과.
- 원본 학습을 재실행하지 않았다. 외부 진단 출력: `/home/jeongeonlee/track1/experiments/track_b_interaction_20261109/dynamic_cohort_registration_replay_v1`.
- 이 진단 레지스트리는 역사적 fit의 실행 SHA가 없는 경우 집계 SHA를 fallback으로 쓰는 기존 동작 때문에 **공식 provenance로 사용하지 않는다**. 원래 검증된 historical registry를 유지한다. 확장 실험은 각 fit의 execution_code_sha가 실제 학습 SHA와 일치하는지 최종 등록 전에 별도 확인한다.
- 한계: 완료된 소규모 cohort에 대한 회귀 검증이다. 확장 실험의 모든 fit 완료/전체 질의 검증을 대신하지 않는다. 가설/데이터/평가/경쟁 설명은 같은 폴더 README를 따른다. 새로운 학습, ablation, 독립 Test 평가는 이번 집계 검증에 해당하지 않는다.

재현: `python -m track_b.scripts.summarize_interaction --run <interaction-root>/foundation_matched_v1 --out <new-diagnostic-dir> --code-sha e250aa59a0a2207b7629c4ff17d176b453a83965 --expected-parameters 2834908` (기존 avfploc:cu124-clean 환경).

## 후속 정정: 학습 출처 fallback 제거

앞선 진단에서 확인한 provenance 위험을 `cf6db637f4405a2678d9a7c3e35f3b2e3f8206da`에서 수정했다. 집계 SHA는 별도 `aggregation_provenance.json`에 저장한다. 학습 SHA가 없는 과거 fit은 `--historical-registry`로 제공한 기존 기록의 checkpoint SHA256 일치를 확인한 뒤 해당 학습 SHA를 사용한다. 출처 충돌, 누락, 형식 오류는 거부한다. SHA 형식 검사가 실제 실행을 독립적으로 증명하는 것은 아니며, 기존 실행 기록과 체크포인트 대응에 의존한다.

검증 출력 `provenance_registration_replay_v1`: 9개 fit의 학습 SHA와 checkpoint hash가 공식 역사적 registry와 모두 일치하고 aggregate도 기존과 동일했다. 음성 대조 `provenance_missing_negative_v1`: 역사적 registry 없이 실행하면 attention_42의 학습 SHA 누락으로 실패하고 registry 파일을 생성하지 않았다. 예상 실패 traceback을 외부 출력 폴더에 보존했다. 기존 `dynamic_cohort_registration_replay_v1`의 출처 한계 기록은 삭제하지 않는다.

재현은 위 명령에서 집계 코드 SHA를 후속 SHA로 변경하고, `--historical-registry <repo>/docs/research/track-b/experiments/2026-11-09-foundation-interaction/experiment_registry.csv`를 추가한다. 신규 출력 폴더를 사용한다. 확장된 주 학습의 코드는 변경하지 않았다.

## 본 학습 초기 상태의 추가 검증

본 학습 `matched_v1/attention_42_initial.pt`를 사용한 19개 Val 장면, 1,216개 질의 replay 완료: Joint/R@1m/fine/raw20/NLL의 기준 F1 대비 허용 오차 초과가 0건이었다(`rtol=1e-5`, `atol=2e-5`). [검증 결과](main_initial42_validation.json). 실제 학습 SHA는 `c083c376631c3f6ed8d89daf9d3a9d93c350fd3b`, replay 실행 checkout SHA는 `1e74884e514a74135e2630879482fb6dc1fe5e61`이며 replay script SHA256은 `6c7291434f761402ac5d0e506e419016f14a32f27841e49b0c9432823ac7df22`이다. 무관한 Track A 두 파일의 기존 dirty 상태는 유지했으며 해당 파일은 replay 경로에서 사용하지 않았다.

짧은 시험 실행과 본 학습 attention/concat의 seed42 초기 상태는 65개 tensor 모두 bitwise 동일했다([검증 결과](main_initial42_tensor_validation.json)). 저장 파일 해시는 직렬화 때문에 다를 수 있으므로 텐서 동일성과 구분했다. 이 결과는 아직 생성되지 않은 모든 seed/mode의 초기 상태를 증명하지 않으며, 최종 학습 성능에 대한 결론도 아니다. 상기 파일을 입력해 기존 `replay_initial_fusion` 명령을 새 출력 폴더에서 실행하면 재현할 수 있다.

## 본 학습 seed43 초기 상태 검증

`matched_v1/attention_43_initial.pt`도 동일한 19-Val/1,216-query 기준 F1 replay를 통과했다. Joint/R@1m/fine/raw20/NLL 허용 오차 초과0건([결과](main_initial43_validation.json)). 학습 SHA는 `c083c376631c3f6ed8d89daf9d3a9d93c350fd3b`, replay 실행 checkout은 `1c9195c9f9778cabb177815b290729a4d1ee970c`; replay script는 위 seed42 검증과 동일하다. 명령의 initial/out/seed를43에 대응하도록 변경해 실행했다. 기존 무관한 Track A dirty 파일 두 개는 보존하고 사용하지 않았다. 현재42·43 두 seed만 이 검증이 완료됐으며44 및 모든 mode의 초기 상태 검증은 최종 등록 시 별도 확인한다. 성능 비교나 독립 Test 검증으로 해석하지 않는다.

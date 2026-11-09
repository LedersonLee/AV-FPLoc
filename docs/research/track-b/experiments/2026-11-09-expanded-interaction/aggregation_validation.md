# 확장 cohort 집계 준비 검증

- 상태: 집계 회귀 검증 완료, 112-Train/19-Val 주 실험은 진행 중. 신규 성능 결론 없음.
- 집계 코드 SHA: `e250aa59a0a2207b7629c4ff17d176b453a83965`.
- 변경: scene/query 수와 bootstrap 설명을 protocol에서 계산한다. 각 fit에서 세 평가 arm의 query ID가 같고, scene별 query 수가 설정과 같으며, 모든 mode/seed의 query ID 집합이 동일함을 검사한다. 질의 ID가 연속 정수라고 가정하지 않는다.
- 검증: 완료된 16-Train/6-Val, 9-fit 원본을 재집계했다. aggregate, per-seed summary, initialization validation은 기존 `foundation_registered_v1`과 JSON 전체 일치. 24개 paired bootstrap 비교도 설명 문자열을 제외하고 일치. 레지스트리 9행 모두 scene22/train1024/val384 확인. Python 구문 검사 통과.
- 원본 학습을 재실행하지 않았다. 외부 진단 출력: `/home/jeongeonlee/track1/experiments/track_b_interaction_20261109/dynamic_cohort_registration_replay_v1`.
- 이 진단 레지스트리는 역사적 fit의 실행 SHA가 없는 경우 집계 SHA를 fallback으로 쓰는 기존 동작 때문에 **공식 provenance로 사용하지 않는다**. 원래 검증된 historical registry를 유지한다. 확장 실험은 각 fit의 execution_code_sha가 실제 학습 SHA와 일치하는지 최종 등록 전에 별도 확인한다.
- 한계: 완료된 소규모 cohort에 대한 회귀 검증이다. 확장 실험의 모든 fit 완료/전체 질의 검증을 대신하지 않는다. 가설/데이터/평가/경쟁 설명은 같은 폴더 README를 따른다. 새로운 학습, ablation, 독립 Test 평가는 이번 집계 검증에 해당하지 않는다.

재현: `python -m track_b.scripts.summarize_interaction --run <interaction-root>/foundation_matched_v1 --out <new-diagnostic-dir> --code-sha e250aa59a0a2207b7629c4ff17d176b453a83965 --expected-parameters 2834908` (기존 avfploc:cu124-clean 환경).

# Track B 초기 전체 후보 실험 결과

증거 수준: **Exploratory**. 프로토콜 전체 완료가 아닌 Stage 0 및 Stage 1–2 초기 개발 실험이다.

## 관찰

실데이터: 학습 4개 장면×16관측=64, 개발 평가 3개 장면×16관측=48. Seed42/43/44. 아래 수치는 seed 평균이며 독립 장면 수는3이다. Test는 사용하지 않았다. Final step을 사전 고정했으며 best epoch를 선택하지 않았다.

| Run | 모델 | 완료 seed | Joint % | R@1m % | R@0.1m % | Raw@20 % | Diverse@20 % | Audio shuffle Joint % |
|---|---|---|---|---|---|---|---|---|
| pilot_v1 | visual | 3 | 0.69 | 3.47 | 0.00 | 4.86 | 8.33 | 0.69 |
| pilot_v1 | acoustic | 3 | 0.00 | 1.39 | 0.00 | 1.39 | 4.86 | 0.00 |
| pilot_v1 | concat | 3 | 0.00 | 0.69 | 0.00 | 2.78 | 4.17 | 0.00 |
| pilot_v1 | noninteraction | 3 | 0.00 | 1.39 | 0.00 | 4.17 | 7.64 | 0.00 |
| pilot_v1 | joint | 3 | 0.00 | 0.69 | 0.00 | 2.78 | 4.86 | 0.00 |
| pilot_long_v1 | visual | 3 | 0.00 | 2.08 | 0.00 | 2.78 | 5.56 | 0.00 |
| pilot_long_v1 | acoustic | 3 | 0.00 | 0.69 | 0.00 | 8.33 | 10.42 | 0.69 |
| pilot_long_v1 | concat | 3 | 0.00 | 2.08 | 0.69 | 6.94 | 7.64 | 0.00 |
| pilot_long_v1 | noninteraction | 3 | 0.00 | 0.00 | 0.00 | 6.25 | 7.64 | 0.69 |
| pilot_long_v1 | joint | 3 | 0.00 | 3.47 | 0.69 | 8.33 | 9.03 | 0.00 |
| pilot_rays_v1 | acoustic | 3 | 0.69 | 2.08 | 0.00 | 3.47 | 5.56 | 0.00 |
| pilot_rays_v1 | joint | 3 | 0.00 | 0.00 | 0.00 | 10.42 | 13.19 | 0.69 |
| pilot_zero_v1 | acoustic | 3 | 0.69 | 2.08 | 0.69 | 8.33 | 9.03 | 0.69 |
| pilot_zero_v1 | joint | 3 | 0.00 | 2.78 | 0.69 | 6.94 | 12.50 | 0.00 |

완료 fit 수: `{'pilot_v1': 15, 'pilot_long_v1': 15, 'pilot_rays_v1': 6, 'pilot_zero_v1': 6}`. 모든 arm, seed별 결과는 [집계](pilot_aggregate_results.json), [registry](experiment_registry.csv)에 보존한다. [장면별 paired bootstrap](paired_scene_bootstrap.json)은 seed를 scene으로 세지 않는다. 3개 장면의 좁거나 0인 구간을 일반화 보장으로 해석할 수 없다.

## 해석과 경쟁 설명

120-step 실험은 Train Joint도 약0.5–1%로 낮았다. 합성8관측 암기는 성공했으므로 구현의 기본 미분 경로는 작동하지만 실제 구조 정합 능력은 확인되지 않았다. 저해상도 RGB, 첫170.7ms 음향만 사용, 작은 CNN, 제한된 학습 관측, 좁은 FP 수용영역, 최적화 실패가 모두 경쟁 설명이다. 단순히 audio shuffle에서 점수가 바뀐다는 사실은 유용성 증거가 아니다.

초기 local FP query의 5자리 반올림 특징은 한 장면에서 최대5,128개 후보가 겹쳤다. 반면 16방향 FP boundary ray는 pilot7장면의 후보를 각각 서로 다른 구조 signature로 만들었다. 이는 **표현 분별력 진단**이며 음향 정합 성공이나 성능 향상의 인과 증명이 아니다. 벽/외부/unknown이 혼합된 검은 픽셀을 반향의 물리적 벽으로 단정하지 않았고 RIR reference를 생성하지 않았다.

## 한계와 실패 보존

- 작은 from-scratch feasibility 모델이다. UnLoc/B2+GEOQ의 강한 동조건 재현은 아직 없다. 과거 성능을 비교표에 옮기지 않았다.
- Local visual/acoustic/concat/noninteraction/joint는 allocated parameter수52,324개가 같지만 활성 경로·연산량은 다르다. F3의 엄밀한 capacity-matched 비교로 주장하지 않는다.
- Ray 대 zero는 동일 모듈/초기값/관측 순서/학습 step의 기하 입력 대조다. 추가 FP 정보 효과와 물리적 acoustic mechanism은 구분해야 한다.
- Frozen/partial, variable camera intrinsics, late RIR, 직접파/후기파 재학습 대조, 시각 저하, furniture shift, calibration, full0.25m/36학습, 독립 Test, OOF discriminator는 미완료다.
- `nll`로 저장된 필드는 실제로 soft-target cross entropy다. 이 보고서에서는 categorical NLL로 사용하지 않으며 정식 NLL은 NA다.
- 0.5m/12bin pilot full bank의 fine oracle는12.5%; fine0%를 숨기지 않는다. 0.25m/36 audit의 fine upper bound는 별도 보고한다.
- 일부 실행은 같은 GPU에서 겹쳤다. 학습 wall time 및 warm timing은 정확도 검증의 비용 기록이며 방법 간 최종 speed ranking이 아니다.

## 판단 / 다음 반증

Dense retrieval의 낮은 정확도·coverage 때문에 후단 판별기로 진행할 근거가 부족하다. 현재 Stage2 성능 gate는 통과로 선언하지 않는다. 학습 길이만 늘린 결과와 ray-vs-zero 대조를 먼저 확인하고, 실제 Train fit 회복 여부로 최적화 실패와 scene generalization을 분리한다. 구조적으로 구분 가능한 후보도 query-RIR로 구분하지 못하면 acoustic encoder의 시간·위상 표현 및 보조 구조 supervision을 검사한다. H-B1–H-B6는 아직 검증되지 않았다.

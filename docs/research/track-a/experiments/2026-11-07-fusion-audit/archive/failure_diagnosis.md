> 역사적 연구 기록. 원본: `astra_fusion_20261107_0200/failure_diagnosis.md`. 원본 SHA256은 [산출물 목록](../artifacts.json)에 있습니다.
> 당시의 완료·재개 지시는 현재 실행 요청이 아닙니다. 최신 판정은 [현재 결론](../../../../current_findings.md)을 먼저 확인하세요.

# 실패 진단

- 실제 측정 기반; subgroup은 **GT를 이용한 사후 분석용**이며 gate inference feature가 아니다. 여기서 reliable/degraded는 성공영역에 들어오는지를 뜻한다. 밝기/노이즈를 개입한 물리적 degradation label이 아니다.
- 진단 증거: [baseline/samples.json](../artifacts.json) (외부 산출물: `baseline/samples.json`), [pilots/samples.json](../artifacts.json) (외부 산출물: `pilots/samples.json`), [analysis.json](../evidence/analysis.json).

| 시각/음향 단독 joint 상태 | n | fixed01 joint (%) | 복구/악화 |
|---|---:|---:|---:|
| 둘 다 성공 | 280 | 96.79 | 0/9 |
| 시각 실패·음향 성공 | 171 | 69.59 | 119/0 |
| 시각 성공·음향 실패 | 648 | 93.83 | 0/40 |
| 둘 다 실패 | 605 | 44.13 | 267/0 |

- 핵심: 두 모달리티의 MAP 실패가 정보 부재를 뜻하지 않는다. 둘 다 실패한605개 중267개가 융합으로 성공한다. 후보 posterior 내 상보 정보가 남아 있다는 관찰이며 원시 특징의 정보량을 완전히 증명하는 것은 아니다.
- 둘 다 성공한280개에서도 융합9개가 실패한다. 두 peak 사이의 score geometry가 새로운 잘못된 peak를 선택할 수 있다. 이것만으로 yaw/가구 원인을 특정하지 않는다.
- [case_studies.json](../artifacts.json) (외부 산출물: `case_studies.json`): `108736737_177263406:11,12,15`는 두 단독 실패→고정 융합 복구; `:497,503,667`은 시각 성공→고정 융합 악화; `:135,353,521`은 adaptive의 fixed 복구, `:34,89,354`는 반대. 파일에 GT, 각 방법 pose/error/entropy/NLL/weight를 포함한다.
- 제한된 oracle 메뉴(V,A,fixed01,equal,adaptive58)에서 query별 GT 최선 선택: R1=85.4460%, joint=83.3333%. **이 메뉴에 대한 oracle upper bound**이며 모든 연속 가중치/모델의 상한은 아니다. 배포 가능한 선택기가 아니고 inference에 사용하지 않았다. fixed01 대비 잠재 메뉴 joint 여유9.0962%p.

## A–G 원인 분류

- A 특징 정보 부족: 미확정. 셔플의 큰 퇴행은 기존 음향에 query 대응 신호가 있다는 증거다. raw/early RIR이 더 나은지는 frozen feature 실험만으로 판정 불가.
- B calibration/scale: T=1 점수는 확률 보정 안 됨. acoustic entropy가 시각보다 낮은 query94.78%인데 acoustic joint26.47%다. 전역 w 축소는 harm와 rescue를 함께 줄였다. 정확한 calibration 학습은 미실행.
- C gate: 기존 adaptive58의51 repair/50 harm, 순1개 이득. weight bin 결과는 `analysis.json/gate_bins`; .2~.5 구간692개에서27 repair/30 harm. weight는 신뢰도로 검증되지 않았다.
- D interaction: 현재 package에는 interaction이 없고 최신 direct_attention 구현에는 진짜 Q/K/V 교환이 있다. 이번에는 새 interaction 모델을 학습하지 않아 우열 원인은 판정하지 않는다.
- E acoustic domain mismatch: observed furnished vs FP proxy의 차이는 구조상 존재. 동일 query의 가구 제거 counterfactual이 없어 인과 크기는 미확정.
- F geometry/yaw: 음향 R1−joint gap9.33%p. yaw-marginal 개입은 개선 실패. XY 재정규화가 함께 바뀌어 순수 방향 ablation으로 해석하지 않는다.
- G confound: 오래된 E+C 점수 vs 현 E+C+N 혼합 위험을 발견해 피했다. 모든 새 비교는 동일 query/후보/평가. 과거 validation 재사용과 pretrained encoder의 nested OOF 부재는 해소하지 못했다.

## 각 반복의 5줄 기록

### B0 baseline
- Observation: 시각54.46%, fixed01 74.24%, adaptive74.30% joint.
- Hypothesis: 이미 음향 신호는 있으나 query-level 신뢰도 선택이 병목일 수 있다.
- Intervention: 검증된 동결 특징에서 동일 후보 점수/기준들을 재계산.
- Outcome: adaptive 순1개 이득, reliable-vision harm58 vs fixed49.
- Remaining uncertainty: 독립 scene 일반화, feature 손실, 가구 mismatch는 남는다.

### P1 calibration005
- Observation: equal과adaptive가 fixed보다 시각 성공을 더 자주 훼손.
- Hypothesis: 전역 acoustic trust 감소로 harm를 줄이면서 전체 성능을 유지할 수 있다.
- Intervention: w=.1→.05, 다른 입력/정규화 동일.
- Outcome: harm49→30이지만 rescue386→344, joint -1.35%p. 기각.
- Remaining uncertainty: query별 calibration을 학습하면 trade-off가 달라질 수 있다.

### P2 entropy_gate
- Observation: 낮은 acoustic entropy와 낮은 acoustic 정확도가 공존.
- Hypothesis: 단순 entropy 비교는 reliability 선택에 실패한다.
- Intervention: acoustic entropy가 더 낮을 때만 w=.1, 아니면0.
- Outcome: joint -0.41%p,3scene 모두 하락. 이 규칙 기각.
- Remaining uncertainty: 학습된 calibration/다른 uncertainty proxy 가능성은 남는다.

### P3 yaw_marginal
- Observation: acoustic의 R1 대비 joint 손실이 vision보다 크다.
- Hypothesis: audio yaw 정보 제거가 joint를 개선할 수 있다.
- Intervention: yaw logsumexp→XY zscore→yaw broadcast, w=.1.
- Outcome: joint -1.41%p, harm49→57. 이 구체적 개입 기각.
- Remaining uncertainty: yaw와XY scaling을 분리한 개입은 미실행.

### P4 shuffled
- Observation: fixed01이 시각 실패386개를 복구.
- Hypothesis: 이 개선이 query에 대응하는 음향 신호에 의존한다.
- Intervention: 동일 scene 내 fixed-point 없는 embedding derangement(seed42).
- Outcome: joint74.24→33.22%, harm427. 대응 신호 기여를 지지.
- Remaining uncertainty: permutation 반복 분산/정확한 물리 원인/독립 scene 효과는 남는다.

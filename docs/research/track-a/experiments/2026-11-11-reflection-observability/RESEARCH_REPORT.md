# Track A Experiment 1 — Geometry-guided Reflection Observability

2026-11-11 KST · **완료: 핵심 물리 시간 정렬 가설 음성, HSSD 조건부 에너지 연관성 양성** · 증거 수준 **Exploratory**. 독립 Test 검증 아님. Experiment 2–4, network 학습, localization scorer·reranking·accuracy 평가는 수행하지 않았다.

## 결론

**현재 SoundSpaces 생성 설정에서 c=343m/s로 계산한 좁은 first-order reflection window가 실제 벽 반사를 포착한다는 핵심 가설은 지지되지 않았다.** 단일 벽 제거·이동에 따라 변하는 음향 응답은 존재하지만, 예상 시간보다 늦게 나타난다. 2m·6m 단일 벽에서 반복 측정한 peak timing bias는 약+2.36ms·+6.15ms다. 반면 HSSD의 후보 기하 정렬 창은 같은 시간·거리 분포를 보존한 다른 질의 대조보다 평균 정규화 power +0.2619 [0.1671,0.3681] 높았다. 이 결과는 **기하와 관측 RIR의 조건부 연관성**이며, 정확한 벽 반사 대응이나 localization 개선의 검증이 아니다.

[물리 검증](PHYSICAL_VERIFICATION.md) · [원리와 pipeline](METHOD_AT_A_GLANCE.md) · [Interactive HTML](interactive.html) · [입력 감사](INPUT_AUDIT.md) · [재현과 비용](REPRODUCIBILITY.md) · [사전 protocol과 후속 개입](PROTOCOL.md).

## 가설·경쟁 설명·증거 구분

- 관찰: 벽 제거/이동으로 단일 벽 early response가 달라지고 dominant return이 늦어지는 경향이 있다. c343±0.5ms 창에는 attributable energy가 사실상 없다.
- 통제에 기반한 결론: 단일 벽 응답은 벽 의존적이지만, 현재 renderer의 좁은 예상 시간 대응은 실패한다. 신호의 peak를 벽 GT로 지정하지 않았다.
- 관찰: HSSD에서 정렬 창의 energy enrichment가 random/shift/matched-marginal 대조보다 높다. coherence는 matched 대조보다 낮다.
- 해석: 후보 기하가 관측의 시간별 에너지 구조와 연관될 가능성이 있다. 시각 후보 선택, 가구/천장/바닥·고차 반사·공통 early decay·FP 정합 오차가 대안 설명이다.
- 미검증 인과 가설: renderer 내부 반사 시간 생성 또는 합성 방식이 지연을 만든다. 공개 API에는 path GT/내부 travel-time trace가 없어 특정 backend 원인을 확정하지 못했다. 지연을 학습·보정해 HSSD 창을 이동시키지 않았다.

## Dataset / Evaluation Protocol

기존 directional-structure의12 Train 장면에서 순수 visual NMS Top20을 그대로 사용했다. 장면별128 질의 중 균등32개를 사전 선택해 총384질의·7680후보를 분석했다. 후보의 XY/yaw만으로 반사 경로를 계산했고, query GT XY/yaw를 시간창 선택·filter에 사용하지 않았다. 질의의 채널 회전은 acquisition schedule의 관측 좌표 convention이다. 센서 높이1.5m는 알려진 acquisition 가정이다.

FP extrusion의 유한 수직 segment를 image-source로 반사하고, specular point가 벽 내부에 있으며 두 leg가 다른 FP 벽에 가리지 않은 경로만 남겼다. 반사점이 평행한 native architectural vertical slice에서0.20m 이내이면 `supported`; 나머지는 `unknown/closure`로 분리했다. 이는 architectural geometry의 추가 확인을 사용한 **감사된 FP 경로**로, FP-only 배포 성능 주장이 아니다. Stage fixed built-in도 포함될 수 있어 semantic wall GT와 같지 않다.

전체 유효 경로28902개 중 공통 분석 범위(각 채널 중심2–79ms)에22295개 supported,4749개 unknown/closure가 있었고1858개는 범위 밖이었다. 실패한 finite/visibility proposals는 생성 단계에서 제거됐으며 그 전체 개수는 따로 계수하지 않았다. Raw RIR 길이는15751–40835 samples (328.15–850.73ms), 모두80ms 분석에 충분했다. 기존 HSSD를 재렌더할 필요가 없었다.

Bootstrap10000회, seed20261111, query 내 candidate 평균→scene 평균→paired scene 재표집. 경로·채널·seed는 독립 표본 수가 아니다. 세 random/donor seed42/43/44를 평균했다. Teacher Train116/Val19 노출과 재사용 연구 장면을 명시한다. Scene ID 및 family 기준으로 Test 제외를 확인했으며 Test payload/결과는 열지 않았다. CI는 다중 비교 보정 없는 탐색적 scene 변동성이다.

## Baseline / Ablation과 정량 결과

평균 squared amplitude를 각 관측 full early1–80ms 평균으로 나눈 **power ratio**를 비교한다. Total energy는 창 길이가 달라 full early와 직접 비교하지 않는다. c343,반폭0.5ms가 primary이며0.25/1ms는 사전 민감도 분석이다. No-guidance full early는 power ratio=1로 정의된다. Peak energy,sharpness=peak/mean,aligned waveform absolute correlation,inter-channel peak residual도 함께 계산했다. 약한 창의 argmax는 noise maximum일 수 있어 detection으로 해석하지 않는다.

### 모든 visual 후보, architecture-supported FP paths

| 대조군 | 정렬 창 평균 | 대조군 평균 | paired scene 차이 [95% CI] |
|---|---:|---:|---:|
| random | 2.7965 | 1.0189 | 1.7776 [1.4962, 2.0655] |
| shifted | 2.7965 | 2.0415 | 0.7550 [0.5943, 0.9038] |
| other_candidate | 2.7981 | 2.7949 | 0.0032 [-0.0016, 0.0101] |
| matched_marginal | 2.7965 | 2.5346 | 0.2619 [0.1671, 0.3681] |
| full_early | 2.7965 | 1.0000 | 1.7965 [1.5149, 2.0854] |

다른 후보는 rank를 한 칸 순환한다. 모든20후보 평균에서는 이 대조가 대수적으로 거의 상쇄되므로 작은 차이를 후보 구분의 증거로 해석하지 않는다. 일부 후보의 경로 부재 때문에 paired subset 평균이 살짝 다르다.

### 사전 고정된 Visual Top1 후보만

| 대조군 | 정렬 창 평균 | 대조군 평균 | paired scene 차이 [95% CI] |
|---|---:|---:|---:|
| random | 3.0202 | 0.9956 | 2.0246 [1.6925, 2.3954] |
| shifted | 3.0202 | 1.8985 | 1.1217 [0.8175, 1.5020] |
| other_candidate | 3.0265 | 2.6088 | 0.4177 [0.1412, 0.6993] |
| matched_marginal | 3.0202 | 2.5432 | 0.4770 [0.2446, 0.7860] |
| full_early | 3.0202 | 1.0000 | 2.0202 [1.6975, 2.3806] |

Top1 vs 다음 후보 power 차이는 양수지만, 실제 정답/오답 분별률이나 localization 성능을 계산한 결과가 아니다. 후보 rank 선택은 visual-only이며 best acoustic rank를 택하지 않았다.

Matched 대조 대비 coherence 차이는 -0.0056 [-0.0112, -0.0005]; timing residual 표준편차 차이는 -0.0046 [-0.0070, -0.0022]ms다. 낮은 timing residual은 좁은 창 내부 argmax의 기술통계이고 실제 wall timing GT 정확도가 아니다. **에너지 이득이 위상 일관성 이득을 동반하지 않았다.**


| 지표 | 정렬 평균 | 분포보존 대조 평균 | 차이 [95% CI] |
|---|---:|---:|---:|
| power_ratio | 2.7965 | 2.5346 | 0.2619 [0.1671, 0.3681] |
| peak_ratio | 41.3703 | 36.1145 | 5.2559 [3.3288, 7.5926] |
| sharpness | 9.2195 | 8.8328 | 0.3867 [0.2734, 0.5136] |
| coherence | 0.2816 | 0.2872 | -0.0056 [-0.0112, -0.0005] |
| timing_residual_ms | 0.1945 | 0.1991 | -0.0046 [-0.0070, -0.0022] |

### 거리·방향·중첩별 실패 조건

| 조건 | 정렬−분포보존 대조 power 차이 [95% CI] |
|---|---:|
| body_bearing_quadrant: 0 | 0.4251 [0.1758, 0.6759] |
| body_bearing_quadrant: 180 | 0.0153 [-0.1179, 0.1771] |
| body_bearing_quadrant: 270 | 0.4839 [0.2104, 0.7664] |
| body_bearing_quadrant: 90 | 0.2407 [0.1076, 0.3836] |
| distance: 1-3m | 0.0773 [-0.0092, 0.1600] |
| distance: <1m | 0.5104 [-0.2041, 1.4963] |
| distance: >=3m | 0.0295 [0.0154, 0.0444] |
| overlap: False | 0.1933 [0.1143, 0.2962] |
| overlap: True | 0.4969 [0.3399, 0.6484] |

방향은 candidate-relative X/Z bearing quadrant이며 yaw 오차·시각 후보 편향과 혼합된다. 물리적 센서 방향성으로 일반화하지 않는다. <1m·1–3m strata CI는0을 포함한다. ≥3m는 작은 양의 차이이며 near/far 비교는 서로 다른 path/query 구성이다. Strata는 path→query→scene 평균으로 계산해 primary candidate-balanced 평균과 수치가 같지 않을 수 있다. 중첩 조건의 더 큰 에너지는 여러 응답의 합일 수 있으며 특정 벽 식별이 더 쉽다는 뜻이 아니다. 대표 scene별 값은 [전체 집계](hssd_summary.json)에 있다.

Supported matched-marginal power 차이는 반폭0.25ms +0.2333 [0.1128,0.3753],0.5ms +0.2619 [0.1671,0.3681],1ms +0.3672 [0.2938,0.4501]다. Unknown/closure 반폭0.5ms는+0.1374 [−0.0172,0.2714]로 불확실하며1ms에서 양수다. 따라서 넓은 창의 양성 결과만으로 물리 wall support를 증명할 수 없다.

## 필수 질문에 대한 답

1. **실제 벽 반사의 acoustic evidence가 있는가?** 통제 renderer에서 벽 의존적 return은 있다. 그러나 좁은 c343 first-order window와의 대응은 실패했다. HSSD에서는 wall identity GT가 없어 첫 반사라고 확정하지 못한다.
2. **어느 거리·방향·장면에서 관측 가능한가?** 단일 벽0.3/2/6m 모두 벽 의존적 response가 있고 timing mismatch도 있다. HSSD12장면과 위 strata에서 조건부 power 차이를 측정했다. 장면 밖 일반화는 미검증이다.
3. **약하거나 겹치면?** 수치 noise floor에도 argmax·큰 dB ratio가 생긴다. 룸/복도에서는 다른 벽·고차 응답 때문에 wall removal이 예상 창 power를 오히려 높일 수 있고 특정 벽 귀속이 불가능하다. Sequential mono 렌더의 phase 변동도 남는다.
4. **단순 시간 선택보다 geometry 정보가 추가되는가?** 같은 시간/거리 marginal을 보존한 query permutation보다 energy가 높아 단순 pooled early-time prior만으로 모두 설명되지는 않는다. 다만 시각 후보와 실내 구조의 상관, broad reverberation을 배제하지 못해 specular mechanism은 미검증이다.
5. **후보별 reflection consistency 가능성?** Top1-vs-next energy 차이는 진단적 가능성이다. 정확한 reflection consistency로 쓰려면 backend timing/path 검증과 FP-wall registration이 먼저 해결돼야 한다. 이번 결과로 scorer를 개발하거나 다음 실험을 시작하지 않았다.

## 실패·반증·일반화 한계

통제36rings와 후속60rings 전부 보존했다. 단일 벽 성공 사례만 고르지 않았으며9 geometry×distance조건의 실패 수치를 모두 공개한다. Trimesh 미설치,PLY loader/degenerate OBJ 오류는 로컬 로그에 보존하고 GLB+원격 바닥으로 수정했다. 최초 direction subgroup에 floating modulo로360°가 별도 bin이 된 오류는 원본 JSON을 보존한 뒤0° bin으로 정정했다. 이는 primary contrast를 바꾸지 않는다. Random sensitivity 창이 최대0.15ms 분석 경계를 넘을 수 있던 구현도 수정해 전 대조를 재계산했고, 수정 전 결과를 보존했다. Non-random contrasts는 전부 동일함을 검증했다.

물리1차 검증이 실패했으므로 HSSD 단계는 '통제 환경에서 검증 완료된 검출기 적용'으로 부를 수 없다. 사용자 요청의 feasibility 비교는 모두 실행하되 **미검증 물리 시간창의 적용 실험**으로 격하한다. Raw6ch RIR는 현장 녹음이 아니라 SoundSpaces 합성이다. 재질·transmission을 비활성화했고 실제 음속/표면 scattering/path export를 직접 확인할 공개 binding이 없다. 반사 경로 GT 대조는 미수행이며 이를 위해 내부 path length/order/wall ID와 합성 전 energy histogram export가 필요하다. wall 제거/이동 개입으로 가능한 대체 검증을 수행했다.

## 종료 판단

Step0 감사,Step1 세 환경/개입·반복·timing falsification,Step2 HSSD 실제 visual 후보·6대조군,Step3 정량/실패/가능성 분석과 모든 필수 산출물이 완료됐다. 실패한 물리 가설을 성공으로 바꾸는 것이 종료 조건은 아니다. 불가능한 path-GT/backend 원인 규명은 위 조건과 함께 명시했다. 이 보고서 작성만으로 종료하지 않았으며 재현 검증·문서 검증 후 지정 branch에 Commit/Push한다. Docs Commit SHA는 commit 이후 최종 응답에만 기록한다.

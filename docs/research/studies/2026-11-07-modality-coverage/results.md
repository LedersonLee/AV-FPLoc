> 역사적 연구 기록. 원본: `modality_coverage_20261107_1700/RESEARCH_REPORT.md`. 원본 SHA256은 [산출물 목록](artifacts.json)에 있습니다.
> 당시의 완료·재개 지시는 현재 실행 요청이 아닙니다. 최신 판정은 [현재 결론](../../current_findings.md)을 먼저 확인하세요.

# AV-FPLoc: 관측 증거, 후보 coverage, 실제 판별의 분리

이번 연구는 새 성능 최적화가 아니라 높은 Top-K coverage의 원인과 투자 우선순위를 검증했다. **관측 기반 acoustic 점수는 정보 없는 NMS보다 훨씬 높은 coverage를 제공한다. 그러나 그 후보를 실제 정답으로 선택하는 판별 격차가 크다.** Bank의 존재, Top-K 포함, Top-1 선택은 서로 다른 양이다. 아래 수치는 모두 반복 사용된 개발 데이터에 대한 탐색적 증거이며 새로운 일반화 성능이 아니다.

## 1. 구현 감사와 평가 범위

- 기존 visual/acoustic 점수는 B2 exchange **후** branch였다. Transformer의 첫 블록 뒤 latent 평균 교환(route1), 두 번째 블록, 잔차 및 acoustic decoder adaptation을 거친다. Pure visual은 원래 캐시 mu/scale, pure acoustic은 원래 observed embedding으로 같은 frozen FP bank와 계산했다. Pure는 단일 모달 **추론**을 뜻한다. B2와 학습 목적/decoder가 일치하는 통제군은 아니다.
- 동일 bank의 float64 visual ray residual와 acoustic dot product를 각각 population z-normalization했다. B2 fusion은 V.85+A.15. Pose index=cell×36+yaw/10. NMS는 XY≤1m 및 yaw≤30°를 동시에 억제한다. 모든 modality에 같은 NMS와 .4m/±10° support(T=.05), local-score refinement를 적용했다.
- Center와 refinement를 별도 저장했다. NMS separation은 center에 대한 제약이며 refinement 후에는 유지되지 않을 수 있다. 이 차이를 ranking 개선으로 부르지 않는다.
- Train12 OOF 6,387관측, Val8 4,190관측, 재사용 Val11 5,775관측, seed42/43. 두 seed 결과는 같은 관측에 대해 평균한 뒤 장면 단위 paired bootstrap10,000회로 평가했다. Pure 출력을 두 seed에서 반복 기록해도 독립 표본이 늘지는 않는다.
- 후보 생성/모델 입력에는 GT가 없다. GT는 성공 판정, coverage/질량 진단 및 회고적 실패 그룹에만 사용했다. 기존 learned ranker는 고정했고 새 validation tuning은 하지 않았다. Test 관측은 접근하지 않았다.
- Upstream visual checkpoint에는 이전 Val19 노출이 있고, Train OOF도 encoder까지 완전히 nested된 학습을 뜻하지 않는다. 모든 CI는 개발셋 기술 통계이며 여러 비교에 대한 확증적 검정이 아니다.

## 2. Experiment A: modality별 coverage

Joint=XY≤1m AND circular yaw≤30°, fine=XY≤.1m. 엄격한 부등호 그대로 사용하며 허용 오차를 추가하지 않았다. 표의 Top1은 해당 후보열 첫 pose의 실제 성공률, TopK는 GT를 사후 확인한 oracle이다. AUC는 K=1..20 선형 축의 trapezoid 면적/19이다. 모든 값은 두 seed 평균(%).

### Val8

| 방법 | Top1 | Top5 | Top10 | Top20 | Fine Top20 | Joint AUC |
|---|---:|---:|---:|---:|---:|---:|
| Pure visual | 47.73 | 71.22 | 80.41 | 87.73 | 28.42 | 77.35 |
| Pure acoustic | 22.12 | 55.63 | 69.98 | 82.12 | 18.95 | 65.44 |
| Exchanged visual | 49.81 | 74.13 | 82.04 | 89.21 | 22.73 | 79.31 |
| Exchanged acoustic | 26.24 | 62.36 | 75.97 | 86.25 | 20.69 | 71.02 |
| Pure-score fixed fusion | 66.37 | 88.59 | 93.32 | 95.61 | 31.53 | 90.85 |
| B2 score-NMS centers | 71.01 | 90.93 | 94.21 | 95.91 | 29.19 | 92.17 |
| B2 support centers | 72.78 | 91.31 | 94.32 | 95.97 | 26.47 | 92.53 |
| B2 support + refinement | 72.60 | 91.30 | 94.27 | 95.99 | 30.13 | 92.49 |

### 재사용 Val11

| 방법 | Top1 | Top5 | Top10 | Top20 | Fine Top20 | Joint AUC |
|---|---:|---:|---:|---:|---:|---:|
| Pure visual | 44.48 | 70.42 | 81.26 | 89.35 | 32.76 | 77.58 |
| Pure acoustic | 20.81 | 54.10 | 70.46 | 83.41 | 18.82 | 65.35 |
| Exchanged visual | 46.81 | 72.65 | 82.81 | 89.80 | 24.33 | 79.13 |
| Exchanged acoustic | 25.52 | 61.53 | 76.58 | 87.52 | 20.36 | 71.25 |
| Pure-score fixed fusion | 64.07 | 89.11 | 94.63 | 97.25 | 34.06 | 91.70 |
| B2 score-NMS centers | 68.08 | 91.10 | 95.39 | 97.56 | 30.52 | 92.91 |
| B2 support centers | 70.19 | 92.14 | 95.57 | 97.55 | 28.51 | 93.43 |
| B2 support + refinement | 70.08 | 91.92 | 95.31 | 97.34 | 31.22 | 93.21 |

Joint 개선이 fine 개선을 보장하지 않는다. Exchanged visual의 fine Top20은 pure visual보다 낮고, support/refinement는 Top1 joint·fine·coverage를 서로 다르게 바꾼다. 고정 temperature를 각 modality에 동일 적용했으며 modality별 최적 support bandwidth를 주장하지 않는다.
전체 K=1,2,3,5,10,20, seed별/scene별, fine curve, XY/yaw 다양성은 summary.json에 있다. 성공 후보의 최초 순위와 GT에 가장 가까운 성공 후보 순위는 analysis/*.npz와 diagnostics.json에 저장했다.

## 3. Experiment B: candidate availability와 정보 없는 대조

Random은 장면당100개의 uniform-score NMS 후보열, 관측당10개 반복이다. Diverse는 동일 SE2 suppression 아래 farthest-XY 후보열20개에서10회 샘플링한다. Donor-score는 같은 장면의 다른 관측 후보열을 가져온다. Donor는 장면/위치 분포 편향을 보존하지만 현재 관측과의 대응은 끊는다. Random SD는 고정 pool에 조건부인10회 sampling SD이며 장면 bootstrap CI와 다르다.

| 개발셋 | Random Top1 / Top20 | Random Top20 SD | Diverse Top20 | Donor acoustic Top20 | 관측 acoustic Top20 |
|---|---:|---:|---:|---:|---:|
| val | 0.40 / 6.91 | 0.40 | 4.95 | 7.98 | 86.25 |
| extra | 0.45 / 8.92 | 0.37 | 6.59 | 11.00 | 87.52 |

| 비교: 관측−대조 Top20 | Val8 Δ [95% CI], pp | Val11 Δ [95% CI], pp |
|---|---:|---:|
| ex_a_nms − random | 79.34 [71.55, 85.49] | 78.59 [74.88, 82.06] |
| ex_v_nms − random | 82.30 [74.93, 86.92] | 80.88 [78.02, 83.11] |
| fusion_nms − random | 89.00 [82.65, 93.23] | 88.63 [85.18, 91.24] |
| ex_a_nms − donor_ex_a_nms | 78.27 [71.92, 83.18] | 76.51 [72.98, 79.56] |

### NMS의 공간 분산 기여: 같은 점수, 같은 K

Stable tie rule(동점 시 낮은 원래 index)을 맞춘 ordinary Top20과 NMS Top20을 추가 비교했다. Top1 인덱스는 모든 관측에서 같다. 따라서 아래 차이는 score/학습 변경이 아니라 추출의 분산 효과다.

| 개발셋 / score | Ordinary20 | NMS20 | Δ [95% CI], pp |
|---|---:|---:|---:|
| val / ex_v | 82.24 | 89.21 | 6.97 [5.75, 8.23] |
| val / ex_a | 75.10 | 86.25 | 11.16 [9.21, 12.89] |
| val / fusion | 93.02 | 95.91 | 2.89 [1.38, 5.03] |
| extra / ex_v | 82.02 | 89.80 | 7.78 [6.60, 9.27] |
| extra / ex_a | 74.50 | 87.52 | 13.01 [11.44, 14.79] |
| extra / fusion | 93.57 | 97.56 | 3.99 [3.03, 4.97] |

이 차이는 NMS가 coverage에 실질적으로 기여한다는 증거다. 그러나 random NMS와 관측 NMS의 큰 격차가 남으므로 공간 분산과 관측 정보는 함께 작용한다. NMS의 oracle 이득은 ranking/Top1 개선이 아니다.

### Bank reachability와 공동 density/yaw perturbation

| 개발셋 / bank | Joint bank oracle | Fine bank oracle | Fusion Top20 |
|---|---:|---:|---:|
| val / .2m /10° | 100.00 | 71.48 | 95.91 |
| val / .4m /20° | 100.00 | 23.22 | 94.80 |
| val / .8m /40° | 99.62 | 4.70 | 85.55 |
| extra / .2m /10° | 100.00 | 72.07 | 97.56 |
| extra / .4m /20° | 100.00 | 21.14 | 96.43 |
| extra / .8m /40° | 100.00 | 6.29 | 87.40 |

동일 subset에서 modality를 비교했고 물리 단위 NMS separation과 K는 유지했다. 각 장면의 pose 수·공간 bounds는 diagnostics.json/density에 있다. XY와 yaw를 함께 바꾼 실험이므로 각각의 독립 효과로 해석할 수 없다. B2 fusion만 subset 내 branch normalization을 다시 계산했으며 pure fusion은 원래 full-bank normalization의 부분집합이다. 따라서 density 표는 B2와 개별 modality 중심으로 해석한다. Fusion 변화에는 subset 정규화 변화도 포함되어, 격자만의 독립 효과로 단정할 수 없다. 개별 modality의 순위는 양의 affine 정규화에 불변이며 이들에서도 coarse subset의 coverage 감소가 관찰됐다.
수치 경계도 감사했다(boundary_audit.json). .1m+1e-9를 진단용으로만 적용하면 원래 bank fine oracle은 Val8 74.27%, Val11 73.73%이며, strict 기준보다 각각2.79pp/1.66pp 높다. 약 절반의 관측이 .1m 경계에 놓이므로 fine 지표는 부동소수점 비교에 민감하다. 본문 지표는 요청한 strict 기준을 유지했고, 이 민감도를 물리적 해상도 손실 전체로 해석하지 않는다. 작은 fine 개선의 견고성을 주장하지 않는다.
Joint 성공 pose가 거의 항상 남아 있어도 coarse subset에서 Top20이 감소한다. Reachability만으로 coverage 변화가 설명되지 않는다. 반면 fine bank oracle의 큰 하락은 격자 표현력 자체의 제한이다. 원래 bank/NMS의 공간 분산은 영향을 주지만 높은 관측 기반 coverage를 단독으로 설명하지 못한다.

## 4. Experiment C: visual ambiguity와 상보성

Entropy T=.05/.2/1, top2 score margin, 최고값−.2z 안의 mode 수, 상위20 XY 분산을 비교했다. Low/medium/high 경계는 Train OOF tertile로 고정했다. 이 지표들은 보정된 정답 확률이 아니며 temperature와 장면 규모에 민감할 수 있다. ambiguity_control.json은 장면 내 rank 표준화 상관도 제공한다.

| 개발셋 / entropy(.05) ambiguity | Visual Top20 | Acoustic Top20 | Fusion Top20 |
|---|---:|---:|---:|
| val / low | 98.63 | 88.97 | 98.76 |
| val / medium | 95.05 | 82.76 | 95.49 |
| val / high | 74.80 | 86.04 | 93.20 |
| extra / low | 98.55 | 88.91 | 98.91 |
| extra / medium | 94.65 | 85.67 | 97.88 |
| extra / high | 70.59 | 87.66 | 95.11 |

장면 내 rank 표준화 뒤에도 높은 ambiguity와 낮은 visual Top20 coverage의 음의 연관성이 유지됐다. 다음은 두 seed 평균 상관계수다. 이는 단순한 장면 크기 차이만으로 설명되지 않는다는 증거이며 인과 효과나 calibration의 증거는 아니다.

| 개발셋 | Entropy .05 | Entropy .2 | Entropy1 | Negative margin | Mode count | Dispersion |
|---|---:|---:|---:|---:|---:|---:|
| train | -0.401 | -0.329 | -0.160 | -0.335 | -0.293 | -0.133 |
| val | -0.368 | -0.279 | -0.092 | -0.347 | -0.282 | -0.169 |
| extra | -0.369 | -0.308 | -0.154 | -0.322 | -0.311 | -0.143 |

| 개발셋 | V만 성공 | A만 성공 | 둘 다 | 둘 다 실패 | V20∪A20 (최대40) | V10∪A10 (최대20) | F20 새 복구 | F20 손실 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| val | 10.73 | 7.77 | 78.48 | 3.02 | 96.98 | 93.90 | 0.78 | 1.85 |
| extra | 10.41 | 8.12 | 79.39 | 2.08 | 97.92 | 94.94 | 0.94 | 1.31 |

합집합의 최대40후보를 F20과 동일 예산으로 해석하지 않았다. V10∪A10은 최대20후보 대조이며 동일 pose가 겹치면 실제 고유 후보 수는 더 적다. Fusion은 둘의 Top20에 없던 성공도 찾고, 합집합에 있던 성공을 잃기도 한다. 한 modality의 coverage 실패와 정보 자체의 부재는 같지 않다.

Acoustic evidence의 GT 인접 질량은 uniform reachable-pose fraction과 비교했다. 아래는 관측별 질량비의 중앙값이며 T는 z-score의 진단적 temperature다.

| 개발셋 | Pure A mass lift(.05/.2/1) | Exchanged A mass lift(.05/.2/1) |
|---|---:|---:|
| val | 17.50× / 19.03× / 3.39× | 20.88× / 26.14× / 3.83× |
| extra | 11.28× / 16.12× / 3.27× | 13.95× / 22.30× / 3.72× |

높은 acoustic coverage를 단순히 넓은 영역에 흩어진 점수로만 설명할 수 없다. 다만 mass lift는 GT를 쓰는 회고 진단이고 posterior calibration 또는 배포 가능한 confidence가 아니다.

## 5. Experiment D: coverage와 discrimination의 격차

| 개발셋 | Exchanged A Top1→Top20 | B2 support/refine Top1→Top20 | 고정 utility ranker Top1 |
|---|---:|---:|---:|
| val | 26.24 → 86.25 | 72.60 → 95.99 | 73.50 |
| extra | 25.52 → 87.52 | 70.08 → 97.34 | 70.68 |

기존 utility ranker는 같은20 support/refinement 후보와 고정 checkpoint/threshold를 사용했다. 후보 coverage는 변하지 않는다. 단일 V/A/fusion score만으로 동일 후보를 재정렬하는 대조도 summary.json의 fixedset_*에 있다. 기존 모델의 개선보다 남은 oracle gap이 훨씬 크다. 이것은 성능 달성 보장이 아니라 현재 판별의 병목을 나타낸다.

### Acoustic 후보열 고정, ranking score만 변경

| 개발셋 / seed | Acoustic 자체 | Visual score로 선택 | Fused score로 선택 |
|---|---:|---:|---:|
| val / 42 | 26.23 | 39.38 | 43.53 |
| val / 43 | 26.25 | 38.47 | 41.46 |
| extra / 42 | 25.25 | 38.18 | 41.61 |
| extra / 43 | 25.78 | 38.44 | 41.52 |

정답/오답이 함께 있는 acoustic 후보열에서 query 내 점수 AUC를 계산했다. 또한 acoustic Top1은 틀렸지만 Top20은 성공한 관측의 최고 정답 후보와 첫 후보 사이 acoustic z-score deficit을 기록했다. 아래 quantile/AUC는 각 seed 통계의 평균이다.

| 개발셋 | Visual-score AUC | Acoustic-score AUC | Fused-score AUC | A-score deficit p10 / median / p90 |
|---|---:|---:|---:|---:|
| val | 0.818 | 0.752 | 0.834 | 0.032 / 0.158 / 0.391 |
| extra | 0.813 | 0.743 | 0.826 | 0.035 / 0.178 / 0.430 |

Val8 seed42에서 visual Top20이 실패하지만 acoustic Top20에 성공 후보가 있는 관측은 332개였다. Acoustic Top1 실패를 같은 acoustic 후보열의 fused-score ranking으로 복구한 관측은 1145개다. 한편 성공 후보가 있는데 V/A/fusion 단일 점수의 argmax 모두 실패한 관측도 1353개였다.
후보를 포함한 상태에서 scalar V/A/fusion 점수가 모두 실패하는 경우가 있어도, 이를 표현력의 불가능성이나 Bayes error로 해석할 수 없다. 더 강한 학습기와 새로운 representation의 기여는 현재 실험만으로 완전히 분리되지 않는다.
정답/오답 후보의 각 점수에 대한 query 내 AUC와 성공 후보 score deficit은 diagnostics.json/failures에 있다. 이 AUC는 GT 사후 그룹 비교이지 새 classifier의 정확도가 아니다. Scalar ranking 실패나 특징 분포 중첩만으로 “현재 특징으로 절대 구별 불가”를 증명할 수 없다. 후보별 구조화된 ray residual·acoustic profile·교차 일치 증거가 필요한지 확인할 후속 평가군으로 사용할 수 있다.
사례는 각 subgroup의 사전 정의된 정렬상 첫 관측을 선택했다(cases.json, failure_cases.png). 정답 표시는 분석용이며 후보 생성에 사용하지 않았다.

## 6. 증거에 기반한 후속 실험: exchange의 실제 관측 정렬 기여

Pure→B2 차이는 residual/decoder adaptation도 포함한다. 이를 분리하기 위해 동일 가중치·decoder에서 (a) latent exchange 비활성화, (b) 같은 장면의 다른 관측을 상대 모달 donor로 사용, (c) 원래 짝을 재실행했다. Matched의 모든 후보 인덱스가 기존 extraction과 일치해야 한다. No-exchange는 별도 재학습한 unimodal 모델이 아니므로 분포 이동 가능성이 있다. Donor도 하나의 결정적 permutation으로 측정했다.

| 개발셋 / 비교 | V Top1 Δ, pp | A Top1 Δ, pp | Fusion Top1 Δ [95% CI], pp | Fusion Top20 Δ, pp |
|---|---:|---:|---:|---:|
| val / matched−noexchange | 0.00 | 0.33 | 0.19 [-0.07, 0.50] | -0.05 |
| val / matched−donor | -0.01 | 0.31 | 0.18 [0.00, 0.40] | -0.05 |
| extra / matched−noexchange | -0.04 | 0.41 | 0.10 [-0.02, 0.23] | -0.03 |
| extra / matched−donor | -0.02 | 0.17 | 0.04 [-0.13, 0.25] | -0.01 |

Exchange를 끈 대조 대비 fusion Top1 차이는 Val8 +.19pp, Val11 +.10pp이며 두95% CI 모두0을 포함한다. Visual branch의 Top1 변화는 거의0이다. Acoustic branch에는 작지만 양의 차이(+.33pp/+.41pp)가 있으므로 “교환이 아무 역할도 없다”는 결론도 과도하다. Pure→B2의 더 큰 차이를 generic latent exchange가 모두 설명하지는 못한다.
이 실험은 현행 B2 고정 가중치에 대한 조건부 대조다. 다른 cross-modal architecture의 잠재력을 부정하지 않는다. Pure→B2 개선 전체를 observation-specific cross-modal exchange의 효과로 돌리지 않는다.

## 7. H1~H4 판정과 남은 한계

- **H1 지지:** acoustic Top1은 낮아도 Top5/10/20은 크게 증가하며 random/donor-score보다 높다. 순수 acoustic에서도 나타난다.
- **H2 지지, 확률 해석은 제한:** visual coverage는 장면마다 다르고, 여러 GT-free ambiguity 지표와 coverage의 음의 연관성은 장면 내 통제 뒤에도 유지된다. 다만 ambiguity 지표는 temperature·margin·규모에 따라 의미가 다르다. 전체 지표/장면 내 상관을 확인해야 하며 단일 entropy를 보정된 confidence로 사용할 근거는 없다.
- **H3 지지, 범위 제한:** fusion은 단일 modality의 실제 Top1 및 coverage를 개선한다. 단순 합집합과 fused ranking은 다르다. 기존 B2의 observation-specific exchange 기여는 별도 frozen intervention 결과만큼만 주장한다.
- **H4 부분 지지/강한 설명은 반증:** bank 밀도와 NMS 분산은 coverage에 영향을 준다. 그러나 같은 bank/separation의 정보 없는 대조군은 훨씬 낮으므로 높은 coverage 전체가 공간 분산 산물이라는 설명은 맞지 않는다.
- **경쟁 가설:** pure→B2 차이에 decoder/잔차 adaptation이 크게 포함될 가능성, entropy에 scene 규모가 섞이는 가능성, refinement가 coverage와 fine precision을 서로 다르게 바꾸는 현상을 각각 별도 대조/통계로 확인했다.
- **미해결:** simulator/furniture mismatch의 인과 효과는 이 캐시만으로 분리할 수 없다. 장면 편향은 within-scene donor/장면 통계로 조사했지만 실제/시뮬레이션 paired 관측이나 가구 상태를 바꾼 실험을 대신하지 않는다. 새로운 데이터의 외부 검증도 없다.

## 8. 투자 우선순위와 일반 원리

1. **Back-end hypothesis discrimination 우선.** 같은 후보에서 V/A 교차 점수로 복구 가능한 사례와 여전히 잘못 고르는 사례를 분리한 평가군을 고정한다. 기존 utility ranker의 작은 개선과 큰 잔여 gap을 출발점으로, 후보별 구조화된 residual/프로파일 증거를 검증한다. 단순 후보 수 증가나 전역 scalar gate를 판별 개선으로 부르지 않는다.
2. **Cross-modal interaction은 후보별 검증 가능한 증거에 투자.** 현재 generic latent exchange의 정렬 효과와 pure→B2 전체 효과를 구분한다. 새 architecture를 학습한다면 같은 후보·같은 encoder/decoder 조건에서 donor/no-exchange 대조와 독립 검증 자료를 포함한다.
3. **Acoustic front-end는 선택적 개선.** Pure acoustic이 이미 높은 coverage를 제공하므로 전면 교체를 우선할 근거는 약하다. A20이 실패하거나 높은 점수의 잘못된 후보에 질량이 몰리는 하위군, simulator mismatch 자료 수집을 우선한다. 현재 결과가 acoustic 표현이 충분하다는 증명은 아니다.
4. **Fine precision 목표에는 bank 해상도를 별도 관리.** .1m oracle의 격자 한계를 먼저 확인하고 finer/sub-grid 위치 추정을 비교한다. Joint 1m 정확도의 ranking 병목과 혼동하지 않는다.
다른 multimodal localization에도 적용할 원리는 (i) full-bank oracle로 표현 가능성을 먼저 측정, (ii) 같은 bank/NMS의 정보 없는 대조로 관측의 추가 기여 측정, (iii) 동일 총 후보 예산에서 modality 합집합과 fusion 비교, (iv) 후보 고정 ranking 대조로 판별 분리, (v) temperature·scene 규모·checkpoint 적응을 인과 기여와 구분하는 것이다.

## 9. 산출물과 재현

- results.png/pdf: coverage·정보 없는 대조·discrimination gap·ambiguity·density·exchange 실험.
- summary.json + analysis/*.npz: 모든 K/scene/seed/fine/AUC/다양성/paired CI, ambiguity 경계와 그룹 크기.
- diagnostics.json: 고정 후보 score ranking, 최초 성공 순위 분포, within-query evidence AUC, density counts/bounds, exchange 대조.
- ambiguity_control.json: 장면 규모를 통제한 ambiguity 연관성. cases.json 및 failure_cases.png/pdf: 정의된 실패 하위군과 사례.
- PROTOCOL.md, environment.json, reproduce.md, 각 scene *.json: 설정·코드/입력/출력 해시와 재현 명령. final_verification.json: 기존 결과/사용자 변경 보존, NMS/strict metrics/인덱스 재현 검증.
- 최초 v1 worker는 큰 장면에서 완료되지 않고 종료됐으며, 직접적인 원인 로그는 확보하지 못했다. v2는 큰 query×pose 배열을 없애는 동등한 Cartesian factorization으로 재개해 메모리 부담을 줄였다. 기존 성공 산출물은 유지하고 명시적 pose enumeration과 비교 검증했다.

## 10. Seed / scene별 요약

| 개발셋 / seed | V1 / V20 | A1 / A20 | F1 / F20 | Support Top1 | Ranker Top1 |
|---|---:|---:|---:|---:|---:|
| val / 42 | 49.86 / 88.93 | 26.23 / 86.63 | 71.38 / 95.87 | 72.89 | 74.01 |
| val / 43 | 49.76 / 89.50 | 26.25 / 85.87 | 70.64 / 95.94 | 72.32 | 72.98 |
| extra / 42 | 46.91 / 89.78 | 25.25 / 87.10 | 67.93 / 97.58 | 69.77 | 70.44 |
| extra / 43 | 46.72 / 89.82 | 25.78 / 87.93 | 68.23 / 97.54 | 70.39 | 70.91 |

| Scene (두 seed 평균) | V20 | A20 | F20 | Random20 | Ranker Top1 |
|---|---:|---:|---:|---:|---:|
| 102344457 | 94.03 | 92.37 | 99.17 | 6.05 | 81.71 |
| 103997478_171030528 | 66.51 | 64.60 | 75.82 | 6.58 | 45.07 |
| 104348082_171512994 | 95.65 | 90.81 | 98.62 | 13.05 | 77.67 |
| 105515448_173104512 | 86.35 | 83.12 | 98.65 | 3.87 | 73.23 |
| 108294798_176710428 | 91.96 | 91.96 | 99.87 | 7.90 | 78.95 |
| 108736635_177263256 | 95.51 | 90.85 | 99.74 | 7.69 | 77.11 |
| 108736737_177263406 | 95.00 | 97.57 | 100.00 | 6.40 | 86.99 |
| 108736800_177263517 | 87.82 | 76.06 | 95.97 | 3.78 | 64.51 |
| 102344115 | 93.59 | 92.34 | 96.56 | 28.25 | 56.72 |
| 102344307 | 85.58 | 93.27 | 97.24 | 7.33 | 73.08 |
| 104862573_172226682 | 90.45 | 92.19 | 98.61 | 5.95 | 78.12 |
| 105515286_173104287 | 84.71 | 73.33 | 98.77 | 2.69 | 68.75 |
| 106366134_174226362 | 86.82 | 84.25 | 96.49 | 7.49 | 67.72 |
| 106366173_174226431 | 92.33 | 87.34 | 96.84 | 9.52 | 59.35 |
| 106879005_174887124 | 95.62 | 92.27 | 100.00 | 10.09 | 81.88 |
| 108294624_176710203 | 86.64 | 77.66 | 93.28 | 6.05 | 70.00 |
| 108294765_176710386 | 94.39 | 91.53 | 98.73 | 14.36 | 75.32 |
| 108294783_176710410 | 83.91 | 94.57 | 97.54 | 7.19 | 63.63 |
| 108736611_177263226 | 92.71 | 86.88 | 99.79 | 6.36 | 79.69 |

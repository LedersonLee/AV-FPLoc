> 역사적 연구 기록. 원본: `pairwise_acoustic_evidence_20261107_2100/RESEARCH_REPORT.md`. 원본 SHA256은 [산출물 목록](artifacts.json)에 있습니다.
> 당시의 완료·재개 지시는 현재 실행 요청이 아닙니다. 최신 판정은 [현재 결론](../../../current_findings.md)을 먼저 확인하세요.

# AV-FPLoc: 반복 음향과 모호한 후보 쌍의 판별 가치

**결론:** 시각적으로 모호한 후보를 구별하는 음향 정보는 확인됐지만, 이번 통제 실험에서는 GEOQ 이후의 재현 가능한 추가 localization 이득을 확인하지 못했다. Val11 pair accuracy는 visual 51.46%, pure acoustic embedding 76.02%, GEOQ 89.61%였다. 최종 joint는 GEOQ 71.818%, C3 71.714%(차이 −0.104pp, scene CI [−0.334,+0.114])였다. 같은 물리적 pose의 native/FP 차이도 반복 변동을 넘어서 남았다. 기존 GEOQ를 유지하며 reference 정합과 반복 신뢰도 검증을 우선한다.

이번 연구는 동일 후보 20개에서 음향의 존재, 반복 안정성, GEOQ 이후 추가 가치, 최종 선택을 분리했다. 기존 B2/GEOQ 및 이전 산출물을 보존했으며 Test는 사용하지 않았다. 단순 cue 추가 이후 후보 상대 특징, 세 donor 재학습, frozen 제거, 쌍별 utility 학습, 반복 점수 평균화까지 후속 반증을 수행했다.

## 평가 범위와 감사

Train12 6,387 queries, Val8 4,190, Val11 5,775; seed42/43. Val8/11은 이미 반복 사용한 development data다. 이전 감사에서 시각 encoder의 Val19(Val8+Val11) 모니터링/노출이 기록됐고, 이 upstream 노출은 남으며, downstream scene OOF를 전체 pipeline의 독립 평가로 해석하지 않는다. 학습 fold의 후보는 기존 nested adapter의 9-scene cache를 사용했다. 기존 비로그 산출물 4,038개의 보존 검증과 C0의 전체 32,704 query-seed 예측값 정확 재현을 확인했다.

GEOQ는 fused score, acoustic scalar, audio 영향을 받은 후보를 사용한다. 따라서 acoustic-free 기준이 아니다. 이 연구의 C1–C5는 추가 block의 조건부 가치이며, 전체 acoustic contribution은 기존 pure/exchanged modality coverage 결과와 구분한다. 모든 구간은 paired scene-cluster bootstrap 10,000회, scene 내 seed 평균 후 계산했다. pair 수를 독립 표본 수로 취급하지 않는다. 구간은 탐색·다중 비교·development 재사용 편향을 제거하지 않는다.

## 1–2. Cue 반복성 및 동일/상이 위치 분리

각 Train fold의 첫 scene, 각 scene의 처음/마지막 query를 GT 없이 선택했다. 4 scenes × 10 poses × 4 fresh renders = 160개; 기존 waveform 40개를 포함해 200개 특징 record를 생성했다. 32 FP poses와 8 native furnished Observed poses다. FP 비교는 nearby, 구조적으로 다른 위치, visual score가 유사하면서 공간적으로 떨어진 위치다. 마지막 분류는 전역 ambiguous threshold를 반드시 만족하는 집합이 아니며 실제 gap을 저장했다.

48 kHz, indirect rays 20,000, source rays 512, ray depth50, 5cm ring, 동일 source/receiver와 production 설정을 유지했다. native furnished 객체 수를 원본과 대조했다. 각 pose의 4 waveform hash는 모두 다르지만 명시적인 renderer RNG seed 제어가 없어 통계적 독립 RNG stream을 보장하지 않는다. 6-channel sequential mono의 한계도 유지된다.

| Cue | Nearby AUC | Structurally different AUC | Visually similar AUC | FP exact-pose retrieval |
|---|---|---|---|---|
| T | 0.927 | 1.000 | 0.985 | 94.01% |
| N | 0.768 | 0.999 | 0.863 | 79.43% |
| C | 0.765 | 1.000 | 0.837 | 77.08% |
| C_early | 0.762 | 1.000 | 0.828 | 76.56% |
| C_full | 0.765 | 0.999 | 0.844 | 77.60% |
| F | 0.919 | 0.985 | 0.971 | 88.02% |
| intermediate | 0.949 | 1.000 | 0.974 | 96.61% |
| embedding | 0.934 | 1.000 | 0.989 | 96.88% |

표의 AUC는 두 pose의 within-repeat cosine distance와 between-pose distance를 비교하며, FP retrieval은 동일 scene의 8 poses에서 다른 repeat를 reference로 사용한다. 가까운 두 위치를 정확히 구분하는 지표이므로 1m localization과 다르다. normalized within-pose variance, original-to-new distance 및 scene별 수치는 `repeat_verification.json`, `repeat_analysis.json`에 있다.

Native Observed를 다시 렌더링하고 기존 FP bank 및 원래 20개 후보를 고정한 결과는 다음과 같다. cue Ridge는 해당 scene을 제외한 기존 fold 모델이며 embedding은 순수 frozen encoder다. 이 표는 작은 8-query/4-scene 사례 연구로, 전체 성능 추정이 아니다.

| Evidence | Repeat choice agreement % | 2+2 agreement % | Single joint % | 2-repeat joint % | 4-repeat joint % |
|---|---|---|---|---|---|
| embedding | 50.00 | 70.83 | 68.75 | 73.96 | 68.75 |
| all | 81.25 | 83.33 | 40.62 | 41.67 | 43.75 |
| T | 78.12 | 87.50 | 26.56 | 26.04 | 31.25 |
| N | 41.67 | 72.92 | 7.81 | 6.25 | 0.00 |
| C | 44.79 | 60.42 | 29.69 | 36.46 | 37.50 |
| no_C | 82.29 | 83.33 | 21.88 | 21.88 | 25.00 |
| no_TN | 57.29 | 75.00 | 35.94 | 39.58 | 37.50 |

평균화는 frozen score를 평균한 실험이며 새로운 학습이 아니다. 더 안정된 선택이 반드시 정답을 의미하지 않는다. 특히 N-only 결과는 안정성만으로 유용성을 판단하는 설명의 반례다. FP 내 재검색과 furnished→FP 판별은 서로 다른 조건이므로 두 차이를 furnishing의 인과 효과로 단정할 수 없다.

같은 위치의 domain 대조를 추가했다. 8 native Observed source/receiver poses를 정확한 yaw와 production ray 설정으로 FP stage에서 4회씩 추가 생성했다(32 renders). 위치를 고정했지만 stage geometry/material/furnishing 차이는 함께 변하므로 가구만의 인과 효과가 아니다. `domain_pair_analysis.json`의 scene/pose별 결과를 확인할 수 있다.

| Feature | Within-domain cosine distance | Same-pose cross-domain distance | Distance AUC |
|---|---|---|---|
| T | 0.0005 | 0.0064 | 0.9147 |
| N | 0.0095 | 0.0144 | 0.7897 |
| C | 0.0499 | 0.0727 | 0.7428 |
| C_early | 0.0574 | 0.0830 | 0.7428 |
| C_full | 0.0430 | 0.0632 | 0.7422 |
| intermediate | 0.0229 | 0.0510 | 0.9479 |
| embedding | 0.1042 | 0.2714 | 0.9622 |

이 대조는 관측/reference 조건 차이가 동일 pose에서도 남는다는 근거다. 거리의 절대 크기는 feature별 정규화 표현에 의존하며, domain distance가 곧 localization 오류나 추가 정보량은 아니다. 전체 원인 중 domain mismatch의 기여 비율은 규명하지 못했다.

## 3–4. 시각적으로 모호한 후보 쌍과 acoustic 판별

원래 20개의 190 unordered pairs 중 XY 거리>1m 또는 yaw 차이>60°, 적어도 한 후보의 visual rank≤5, |visual score gap|≤0.049938을 사용했다. gap은 Train의 GT-free competitive-pair 25th percentile이다. Train 전체의 비지도 분포를 사용했으므로 strict outer-fold threshold는 아니다. GT는 mixed joint-success label 판정에만 사용한다. 아래 pair accuracy는 query 내 평균 후 집계하며 최종 Top1과 다르다. Candidate AUC는 20개 전체의 positive-negative pairs에서 별도로 계산했다.

| Evidence | Train % | Val8 % | Val11 % | Val11 Δ vs visual [95% CI] pp |
|---|---|---|---|---|
| visual_scalar | 52.823 | 51.895 | 51.460 | +0.000 [+0.000, +0.000] |
| pure_acoustic_embedding | 77.976 | 76.916 | 76.019 | +24.559 [+22.066, +27.432] |
| acoustic_scalar | 79.021 | 78.279 | 77.796 | +26.336 [+23.907, +29.231] |
| fusion_scalar | 78.569 | 78.171 | 77.481 | +26.021 [+23.622, +28.804] |
| learned_cue_T | 65.335 | 68.538 | 67.133 | +15.672 [+13.248, +18.314] |
| learned_cue_N | 62.705 | 63.160 | 63.233 | +11.773 [+9.428, +14.562] |
| learned_cue_C | 62.239 | 62.496 | 62.314 | +10.854 [+9.009, +13.277] |
| learned_cue_C_early | 61.799 | 61.820 | 61.814 | +10.354 [+8.562, +12.625] |
| learned_cue_C_full | 61.754 | 61.715 | 61.842 | +10.382 [+8.725, +12.589] |
| learned_cue_C_late | 55.909 | 57.220 | 57.088 | +5.628 [+2.406, +8.599] |
| learned_cue_F | 51.990 | 52.119 | 52.674 | +1.214 [-0.776, +3.218] |
| learned_cue_all | 69.004 | 69.437 | 69.879 | +18.419 [+15.994, +20.941] |
| GEOQ | 90.256 | 90.058 | 89.615 | +38.155 [+36.512, +40.244] |

Acoustic scalar는 exchanged score이며 pure embedding과 구분된다. Learned cue는 기존 Train-supervised query→reference Ridge의 held-scene 결과다. T/N/C의 방향성이 있는 판별력은 존재하지만, 그 자체가 기존 GEOQ 이후의 추가 정보임을 증명하지 않는다. 물리적 wall peak, DoA, first-order reflection 또는 cue 제거의 인과량을 주장하지 않는다.

Train에서 고정한 GEOQ pair-margin 하위25%(|margin|≤0.053967)의 모호한 쌍은 추가 탐색 동기였다. Train에서 GEOQ52.32%/all-cue57.83%, Val11에서54.33%/56.05%였다. 이는 일부 residual opportunity의 기술 통계이며, 기존 입력을 조건으로 학습 가능한 추가 gain이라는 증명이 아니어서 아래 matched 학습과 반증으로 확인했다(`pair_opportunity_*.json`).

## 5–6. 조건부 정보와 matched/donor/removal 통제

C0_matched는 원래 GEOQ를, C1은 이전 CUES 모델의 모든 예측값을 정확히 재현했다. 이는 동일 설정의 재실행이며 독립적인 새 데이터 검증은 아니다. C1은 후보별 T/F/C/N/M consistency summary, C2는 시각적으로 가장 가까운 분리 후보에 대한 query-conditioned squared residual 차이의 고정 projection, C3은 C1+C2+visual/acoustic agreement다. 선형 signed residual끼리 빼면 query가 소거되므로 C2는 squared residual을 사용했다. C4는 같은 scene에서 query cue를 fixed-point 없는 세 permutation으로 교체해 재학습했다. C5는 기존 입력만의 고정 basis expansion, C2_scalar는 기존 입력만의 challenger context다.

모두 동일 150-iteration/15-leaf histogram trees, 876 nominal columns, 동일 utility U=joint+0.5fine 및 scene-balanced weight를 사용했다. 활성 입력 차원과 표현의 효과적 용량까지 같다는 뜻은 아니다. C3 frozen 제거와 donor는 OOD 입력 개입이며, cue 없이 재학습한 C0/C4와 구분한다.

| Model/control | Train Δjoint pp | Val8 Δjoint pp | Val11 Δjoint pp |
|---|---|---|---|
| C0_matched | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| C1_candidate | -0.266 [-0.456, -0.089] | +0.048 [-0.043, +0.176] | -0.277 [-0.550, -0.017] |
| C2_pair | -0.070 [-0.263, +0.155] | -0.131 [-0.417, +0.141] | -0.251 [-0.550, -0.000] |
| C3_agreement | -0.352 [-0.694, +0.025] | -0.203 [-0.480, +0.066] | -0.104 [-0.334, +0.114] |
| C4_donor1 | -0.258 [-0.608, +0.045] | -0.095 [-0.291, +0.137] | -0.147 [-0.438, +0.135] |
| C4_donor2 | -0.431 [-0.656, -0.163] | +0.024 [-0.122, +0.175] | -0.156 [-0.406, +0.043] |
| C4_donor3 | -0.415 [-0.743, -0.090] | +0.000 [-0.241, +0.240] | -0.165 [-0.346, -0.000] |
| C5_basis | +0.000 [-0.394, +0.367] | -0.263 [-0.497, -0.026] | -0.121 [-0.418, +0.173] |
| C2_scalar | -0.399 [-0.727, -0.091] | -0.024 [-0.296, +0.240] | -0.043 [-0.281, +0.209] |
| F_drop_added | -0.219 [-0.426, -0.021] | +0.048 [-0.154, +0.249] | +0.052 [-0.233, +0.342] |
| F_keep_absolute | -0.266 [-0.446, -0.063] | -0.060 [-0.245, +0.132] | +0.009 [-0.273, +0.256] |
| F_donor1 | -0.579 [-0.872, -0.300] | -0.382 [-0.622, -0.175] | -0.147 [-0.523, +0.182] |
| F_donor2 | -0.485 [-0.699, -0.229] | -0.382 [-0.673, -0.125] | -0.165 [-0.590, +0.199] |
| F_donor3 | -0.384 [-0.753, -0.039] | -0.298 [-0.554, -0.069] | -0.156 [-0.400, +0.057] |

Val11의 C3는 99 query-seed 실패를 복구하고 111 성공을 오답으로 바꿨다. Frozen added-block 제거에서는 복구 42개가 사라지고 harm 51개가 복원됐다. Donor 세 permutation에서는 복구 33/36/39개가 사라졌다. 추가 특징의 결정 민감도는 있지만, repair의 존재가 순이득이나 정보 필요성의 증명은 아니다. OOD 개입이라는 한계와 모든 사례는 `decision_dependence.json`에 기록했다.

## 7. 학습 목표와 최종 Top1

후속 P0/P1은 입력과 tree budget을 맞춘 쌍별 utility 회귀다. query마다 GT 없이 visual gap이 가장 작은 separated 24 pairs를 선택하고 양방향 U_i−U_j를 학습했다. 추론은 모든 190 pairs의 antisymmetric prediction을 평균하여 후보 점수를 만든다. P0는 기존 입력, P1은 C1과 동일 후보별 cue를 포함한다. objective/aggregation 효과는 P0−C0, feature 효과는 P1−P0로 읽는다. 추가로 P0_broad/P1_broad는 close-visual 12개와 나머지 중 query-keyed 무작위 12개를 학습하여 train/inference pair-support 차이를 반증한다. 이는 Train P0 저하 후 등록한 후속 실험이다. 원래 candidate0 대비 frozen switch threshold(seed42=0, seed43=.01)를 유지했다.

| Model | Val11 R@1m % | Joint % | Fine % | Ambiguous pair % | Candidate AUC % | Δjoint vs GEOQ pp |
|---|---|---|---|---|---|---|
| GEOQ | 73.195 | 71.818 | 23.446 | 89.615 | 90.071 | +0.000 [+0.000, +0.000] |
| C1_candidate | 73.013 | 71.541 | 23.403 | 89.649 | 90.005 | -0.277 [-0.550, -0.017] |
| C2_pair | 73.004 | 71.567 | 23.506 | 89.598 | 90.037 | -0.251 [-0.550, -0.000] |
| C3_agreement | 73.177 | 71.714 | 23.472 | 89.527 | 89.957 | -0.104 [-0.334, +0.114] |
| P0_base | 71.290 | 69.732 | 22.935 | 88.476 | 89.349 | -2.087 [-3.656, -0.752] |
| P1_cues | 71.325 | 69.766 | 22.926 | 88.603 | 89.361 | -2.052 [-3.624, -0.708] |
| P0_broad | 71.896 | 70.450 | 23.273 | 88.682 | 89.600 | -1.368 [-2.605, -0.315] |
| P1_broad | 71.913 | 70.494 | 23.281 | 88.702 | 89.630 | -1.325 [-2.491, -0.300] |

| Contrast | Train Δjoint pp | Val8 Δjoint pp | Val11 Δjoint pp |
|---|---|---|---|
| P0_base − C0_matched | -1.816 [-2.614, -1.027] | -2.220 [-3.528, -0.654] | -2.087 [-3.656, -0.752] |
| P1_cues − P0_base | -0.039 [-0.156, +0.077] | +0.036 [-0.142, +0.184] | +0.035 [-0.063, +0.144] |
| P1_cues − C1_candidate | -1.589 [-2.404, -0.768] | -2.232 [-3.557, -0.662] | -1.775 [-3.181, -0.539] |
| P0_broad − C0_matched | -1.503 [-2.117, -0.927] | -1.790 [-2.666, -0.834] | -1.368 [-2.605, -0.315] |
| P1_broad − P0_broad | -0.078 [-0.162, -0.000] | +0.060 [-0.103, +0.234] | +0.043 [-0.112, +0.208] |

추가 frozen 반증에서 190-pair 평균 대신 candidate0와의 19개 직접 비교만 사용했다. Train OOF joint는 P0 약74.22%, P1 약74.16%로 Borda의 약75.12%/75.08%보다 낮았다. 따라서 평균 aggregation만이 실패의 원인이라는 설명은 지지되지 않았다. 이 분기는 Train에서 중단했고 development에서 유리한 aggregation을 고르지 않았다(`pair_anchor/`).

| Model | Val11 oracle20 % | Top1 given oracle % | Oracle20−Top1 pp | Pure-visual-success preservation % |
|---|---|---|---|---|
| GEOQ | 97.342 | 73.779 | 25.524 | 93.889 |
| C3_agreement | 97.342 | 73.673 | 25.628 | 94.025 |
| P1_cues | 97.342 | 71.671 | 27.576 | 93.499 |
| P1_broad | 97.342 | 72.418 | 26.848 | 93.655 |

C1의 Val11 ambiguous-pair 점추정은 GEOQ보다0.034pp 높지만 최종 joint는0.277pp 낮았다. Mixed-label 쌍에서의 정확도와 실제20개 중 하나를 고르는 정확도를 교환해 해석할 수 없다.

전체 original20 oracle와 conditional Top1, seed별 수치, scene별 repair/harm는 `model_statistics_*.json` 및 `paired_statistics.json`에 저장했다. 후보를 새로 추출하지 않았으므로 end-to-end 개선 또는 coverage 증가를 주장하지 않는다.

## 8. Rescue/harm 및 실패 조건

| Group | Model | Query-seed n | Δjoint pp | Repair | Harm |
|---|---|---|---|---|---|
| high_ambiguity | C3_agreement | 3087 | -0.065 [-0.615, +0.469] | 64 | 66 |
| high_ambiguity | P1_cues | 3087 | -0.939 [-2.803, +0.517] | 167 | 196 |
| high_ambiguity | P1_broad | 3087 | -0.518 [-2.098, +0.737] | 146 | 162 |
| pure_visual20_fail_audio20_success | C3_agreement | 934 | -0.535 [-2.304, +1.368] | 27 | 32 |
| pure_visual20_fail_audio20_success | P1_cues | 934 | -0.321 [-4.158, +2.747] | 59 | 62 |
| pure_visual20_fail_audio20_success | P1_broad | 934 | -1.178 [-4.582, +1.524] | 49 | 60 |
| pure_visual1_fail_audio1_success | C3_agreement | 1160 | -0.517 [-1.171, +0.000] | 13 | 19 |
| pure_visual1_fail_audio1_success | P1_cues | 1160 | -0.259 [-3.937, +2.973] | 66 | 69 |
| pure_visual1_fail_audio1_success | P1_broad | 1160 | +0.862 [-1.418, +2.917] | 56 | 46 |
| both_pure1_fail_fusion20_success | C3_agreement | 4993 | -0.260 [-0.672, +0.195] | 68 | 81 |
| both_pure1_fail_fusion20_success | P1_cues | 4993 | -4.286 [-6.201, -2.404] | 221 | 435 |
| both_pure1_fail_fusion20_success | P1_broad | 4993 | -3.024 [-4.464, -1.585] | 197 | 348 |
| fusion20_oracle_GEOQ_fail | C3_agreement | 2975 | +3.328 [+2.529, +4.357] | 99 | 0 |
| fusion20_oracle_GEOQ_fail | P1_cues | 2975 | +13.546 [+12.139, +15.235] | 403 | 0 |
| fusion20_oracle_GEOQ_fail | P1_broad | 2975 | +11.664 [+10.522, +13.149] | 347 | 0 |
| strong_wrong_acoustic | C3_agreement | 2648 | -0.189 [-0.763, +0.357] | 19 | 24 |
| strong_wrong_acoustic | P1_cues | 2648 | -3.663 [-5.688, -1.994] | 74 | 171 |
| strong_wrong_acoustic | P1_broad | 2648 | -3.059 [-4.437, -1.859] | 62 | 143 |

Both-pure-Top1-fail/fusion-Top20-success는 이전 fusion-Top1-success 집합과 다르며 새로 계산했다. GEOQ-failure 집합의 baseline 성공률은 정의상 0이므로 repair만 보고 전체 이득으로 해석하면 안 된다. Pure/exchanged 시각 그룹을 분리했고, 상세 pair 통계에는 각 cue와 high ambiguity도 포함했다.

`repair_harm_cases.png`는 같은 scene의 high-ambiguity repair/harm 예시를 보여준다. GT는 평가에만 표시하며 position만 그렸다. 사례별 query ID, GT, baseline/new pose, scalar scores와 subgroup을 `model_statistics_*.json`의 cases에 저장했다. 각 scene/seed/type의 처음 10개이며 대표 사례의 빈도를 추정하는 표본이 아니다. 반복 불안정 사례는 `repeat_ranking.json`의 모든 선택/정답 기록으로 확인할 수 있다. Geometry 관련 탐색은 Train 고정 free-grid-area/depth-CV tercile로 집계했다(`environment_analysis.json`). Global free area는 room size가 아니며 방 형태·가구·재질의 인과 설명을 제공하지 않는다.

## 9. 검증된 사실, 경쟁 가설, 반증 범위

- 시각적으로 비슷한 후보 사이에서 음향 score/embedding 및 T/N/C 기반 predictor가 올바른 방향을 지지하는 정보는 존재한다. GEOQ가 이미 사용하는 acoustic score와 새 정보의 가치는 별개다.
- 반복성은 cue와 평가 조건에 따라 다르다. FP 내 exact-pose retrieval의 높은 성능만으로 furnished observation의 안정적인 후보 판별을 보장하지 않는다.
- 평균화로 선택을 안정시켜도 정확도가 반드시 높아지지 않는다. 불안정성만이 실패의 유일한 원인이라는 설명은 지지되지 않는다.
- 단순 후보 summary의 한계를 challenger-relative 특징, agreement, scalar-context control 및 matched pair objective로 추가 검증했다. 해당 결과가 모든 acoustic representation이나 learner에 대한 불가능성 증명은 아니다.
- GEOQ와의 중복, residual information을 학습하지 못하는 문제, reference mismatch, finite renderer variation이 경쟁 설명으로 남는다. Donor 결과는 추가 block에만 적용되며 전체 acoustic을 제거하지 않는다.

## 10. 후속 투자 우선순위와 재개 조건

우선순위는 Front-end의 반복 관측 신뢰도 및 query/reference 대응 품질 검증, 그 다음 Cross-modal Interaction의 실제 모호한 후보 조건부 정보 보존이다. Back-end는 기존 GEOQ와 작은 통제 모델을 유지하면서 새로운 정보가 확인될 때만 확장한다. 독립 scene 확보와 사전 고정 평가가 현재의 추가 development sweep보다 정보 가치가 높다.

다음 재개 실험은 더 많은 사전 선정 poses/scenes에서 양 domain을 함께 반복 생성하고, native furnishing을 통제한 paired 조건과 신뢰도 예측을 분리한다. 동일 위치 domain 차이를 이미 작은 subset에서 확인했으므로, 새로운 관측/참조를 확보하면 먼저 안정성·pair discrimination을 확인하고 Train-only 조건부 gain이 재현될 때 최종 ranking을 평가한다. 임의의 architecture 확대나 현재 development 세트에서의 추가 threshold 검색은 우선하지 않는다. 실환경 음향 일반화, 특정 반사 경로의 물리적 해석, 인과적 furniture 효과는 미해결이다.

## 재현 자료

과학 그림은 `research_summary.png`/`.pdf`, 사례 지도는 `repair_harm_cases.png`에 있다. 실행 절차는 `REPRODUCE.md`, 사전 규칙과 후속 결정 시점은 `PROTOCOL.md`를 참조한다. 모델/코드/설정/실패 로그와 query/scene 통계를 보존했다. 최종 파일 manifest와 검증 결과는 `final_verification.json`, `artifact_manifest.json`에 있다.

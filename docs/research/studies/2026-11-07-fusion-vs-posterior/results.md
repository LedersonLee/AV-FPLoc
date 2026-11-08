> 역사적 연구 기록. 원본: `fusion_vs_posterior_20261107_1300/RESEARCH_REPORT.md`. 원본 SHA256은 [산출물 목록](artifacts.json)에 있습니다.
> 당시의 완료·재개 지시는 현재 실행 요청이 아닙니다. 최신 판정은 [현재 결론](../../current_findings.md)을 먼저 확인하세요.

# Fusion vs. Posterior Decision — 연구 결과

## 결론과 검증 범위

고정된 B2 점수에서 support 후 국소 정밀화를 적용한 사전 지정 방법의 Val11 동시 성공률은 68.078% → 70.078%이다. 변화는 +2.000 pp, 장면 bootstrap95% CI [+1.258 pp, +2.839 pp]이다.
이 비교는 feature·fusion weight·candidate bank를 바꾸지 않은 Back-end 결정 규칙의 개입이다. 개선을 학습형 adaptive fusion의 성과로 부르지 않는다.
Val8은 반복 탐색한 개발 자료다. Val11(5,775개 관측)은 이번 fusion/decoder 학습·선택에 사용하지 않고 방법·모델·코드를 고정한 뒤 평가했다. 그러나 기존 시각 인코더는 전체 Val19를 ray-NLL 모니터링과 학습률 조정에 사용했다. 따라서 Val11은 고정 인코더 위의 decoder에 대한 조건부 확인이며, 전체 시스템의 완전히 미사용 데이터 일반화 검증이 아니다. Test 결과는 열지 않았다. `validation_scope_audit.json`에 원본 학습 계약과 해시가 있다.

## 구현 감사와 동일 조건

- Front-end: 고정 시각/음향 인코더, B2 cross-modal exchange, 전체 SE(2) 후보 점수와 fusion weight. B2는 modality별 Transformer 뒤 latent를 교환·평균하는 구조이며 직접적인 audio-query/visual-key-value attention으로 설명하지 않는다.
- Back-end: z-score된 두 점수의 고정 결합을 exp(score/T)로 변환하고 XY0.4m·yaw±10°의 질량을 합산해 중심을 선택한다. refinement는 그 중심의 같은 이웃에서 원래 점수가 가장 높은 pose를 선택한다. 후보 추가나 GT 기반 filtering은 없다.
- 고정 fusion V0.85/A0.15, B2 epoch8, 두 seed42/43, 동일 관측·후보·checkpoint·strict≤1m/30° 및≤0.1m 평가. T=.05. Train12 gate 입력은 장면별4fold B2 OOF; gate CV까지 upstream과 완전히 nested되지는 않았으므로 Train CV는 독립 일반화 추정이 아니다.
- 기존 v2 산출물255개와 원래 git diff를 보존했다. 기존5방법8380개 및 확장11방법8380개의 예측 index를 정확히 재현했다. 입력 생성도 원래 batch64 acoustic/batch4 vision에서6개 tensor가 정확히 일치했다. 실패한 batch4 acoustic smoke는 약9e-5 차이를 보여 원인과 로그를 남겼고 허용오차를 완화하지 않았다.

## 동일 입력의 2×2 및 학습 목표 개입

### Val8 개발

| 방법 | R@1m | 동시 성공 | R@0.1m | 시각 성공 손실 수/seed |
|---|---:|---:|---:|---:|
| vision | 50.668% | 49.809% | 13.365% | 0.0 |
| acoustic | 36.611% | 26.241% | 6.480% | 1445.5 |
| fixed_map | 72.566% | 71.014% | 22.136% | 149.5 |
| fixed_support | 74.415% | 72.780% | 19.690% | 144.5 |
| fixed_refine | 74.224% | 72.601% | 22.279% | 147.5 |
| adaptive_map | 73.079% | 71.480% | 22.399% | 96.0 |
| adaptive_support | 73.819% | 72.220% | 19.535% | 118.0 |
| adaptive_refine | 73.687% | 72.076% | 22.327% | 119.0 |
| support_trained_map | 73.341% | 71.730% | 22.351% | 101.0 |
| support_trained_support | 74.368% | 72.721% | 19.535% | 115.0 |
| support_trained_refine | 74.189% | 72.554% | 22.315% | 116.0 |

### Val11 decoder 확인

| 방법 | R@1m | 동시 성공 | R@0.1m | 시각 성공 손실 수/seed |
|---|---:|---:|---:|---:|
| vision | 47.879% | 46.814% | 13.636% | 0.0 |
| acoustic | 34.788% | 25.515% | 6.043% | 1932.0 |
| fixed_map | 69.810% | 68.078% | 22.762% | 207.5 |
| fixed_support | 71.740% | 70.190% | 21.403% | 188.5 |
| fixed_refine | 71.706% | 70.078% | 23.247% | 190.0 |
| adaptive_map | 70.900% | 69.117% | 23.074% | 122.5 |
| adaptive_support | 71.636% | 70.035% | 20.952% | 140.0 |
| adaptive_refine | 71.558% | 69.913% | 23.022% | 140.5 |
| support_trained_map | 70.701% | 68.918% | 22.935% | 129.0 |
| support_trained_support | 71.714% | 70.173% | 21.100% | 143.0 |
| support_trained_refine | 71.662% | 70.043% | 22.970% | 144.0 |

MAP-trained gate와 support-trained gate는 같은22개 GT-free MAP 후보 특징, 여덟 가중치, HistGradientBoosting 구조, Train OOF 입력, 임계값0.1을 사용한다. 바꾼 것은 학습 정답(MAP 성공 → support 성공)이다. 각 gate가 선택한 동일 가중치를 MAP/support/refinement에 공통 적용했다. 모델 선택에 Val11을 쓰지 않았다.

**confirmation_dev의 paired contrast** (동시 성공; 장면 bootstrap95% CI):

- old_gate_interaction: -1.026 pp [-1.390 pp, -0.657 pp].
- target_change_on_support: +0.501 pp [+0.057 pp, +0.857 pp].
- target_change_on_map: +0.251 pp [-0.077 pp, +0.562 pp].
- new_gate_vs_fixed_support: -0.060 pp [-0.454 pp, +0.437 pp].

**confirmation의 paired contrast** (동시 성공; 장면 bootstrap95% CI):

- old_gate_interaction: -1.195 pp [-1.667 pp, -0.742 pp].
- target_change_on_support: +0.139 pp [-0.040 pp, +0.350 pp].
- target_change_on_map: -0.199 pp [-0.366 pp, -0.027 pp].
- new_gate_vs_fixed_support: -0.017 pp [-0.463 pp, +0.448 pp].

개발 자료에서 MAP-trained 선택기와 support의 효과는 단순히 더해지지 않았다. 정답만 바꾸자 두 seed의 support 성능이 좋아졌지만 MAP도 함께 좋아졌다. 따라서 decoder와 학습 목표의 불일치가 일부 원인이라는 증거는 있으나, support 정답이 더 안정적인 supervision을 제공한다는 경쟁 설명도 남는다. 분류기의 출력값을 보정된 성공확률로 주장하지 않는다.

## 병목: 가중치 oracle과 hypothesis oracle

다음은 Val8 개발 자료의 진단 전용 oracle이다. GT는 제안 생성 이후 성공 판정에만 사용하며 실제 추론으로 유입하지 않았다.

| 진단 | 두 seed 평균 동시 성공 포함률 |
|---|---:|
| continuous_weight_oracle | 84.248% |
| fixed_separated_top5 | 90.907% |
| fixed_separated_top10 | 94.212% |
| fixed_separated_top20 | 95.883% |
| support_separated_top20 | 95.967% |

연속 가중치 oracle은 양의 길이를 갖는 모든 MAP-winner interval과 양끝을 포함하고, 고립된 내부 tie 소유권은 제외한다. 각 interval의 중간에서 실제 argmax를 검증했다. 기존8점 grid oracle보다 넓은 진단이다. Separated Top-K는 ≤1m AND≤30° 이웃을 순차 억제해 생성했으며 GT를 쓰지 않는다.
성공 pose가 점수 상위의 서로 다른 가설에 이미 상당수 존재한다. 현재 점수에서 scalar weight만 바꾸는 것보다 hypothesis 선택의 진단 여지가 크다. 이는 구현 가능한95% 성능을 보장하지 않으며, feature 개선의 가능성을 배제하지 않는다. 보통 Top-K에서 support가 약해 보이던 현상의 상당 부분은 같은 공간 mode에 후보가 중복되는 효과였다.

## 성공·실패 메커니즘과 ablation

Val11 확인의 동시 성공률:

| 개입 | 동시 성공 |
|---|---:|
| xy_support | 69.784% |
| yaw_support | 68.208% |
| degree_support | 70.026% |
| shuffled_map | 29.541% |
| shuffled_refine | 29.359% |
| visual_refine | 47.290% |

같은 점수를 고정하고 decoder만 바꾸는 비교는 feature 개선이나 가중치 변경을 원인에서 제외한다. XY/yaw 분리와 이웃 수 정규화는 공간 집계와 후보 밀도 prior를 구분한다. 음향 관측을 GT 없이 derangement한 control과 visual-only refinement는 개선이 올바른 음향 정보를 필요로 하는지 검사한다. 이 control은 확인 실행 시작 후, Val11 지표를 보기 전에 지정했으며 새 방법 선택에 사용하지 않았다.
개발 자료에서 기존 선택기는 한 모달리티가 단독 성공하는 집단을 개선했지만 둘 다 단독 실패하는 집단에서는 고정 fusion의 복구를 잃었다(두 seed MAP 기준 net−56/−58, support 기준−67/−61). 이는 단독 모달리티 신뢰와 상보적 결합의 성공이 다를 수 있음을 보여준다. GT 기반 집단은 사후 분석용이다. 확인 집단별 수치와 장면별 결과는 summary JSON에 모두 있다.

## Calibration과 결정 품질의 구분

Train42 OOF에서13개 온도의 nearest-cell categorical point NLL을 비교했으나 최적 온도는 기존.05였다. 이 가설에서는 추가 개선을 얻지 못했다. GT-neighborhood NLL은 성공영역 사건의 질량이고 point NLL과 목적이 다르다. ECE는 선택 pose 중심의≤1m/30° 영역 확률과 실제 성공을 비교한10-bin 값이다. normalized support는 합이1인 별도 공간 분포이며, refinement pose의 calibrated posterior가 아니다.

| Val11 posterior | point NLL | GT영역 NLL | ECE10 |
|---|---:|---:|---:|
| fixed_posterior | 5.039857 | 1.200261 | 0.076596 |
| normalized_support | 5.334688 | 1.185426 | 0.089790 |

support NLL은 확률0 방지를 위해1e-300 floor를 사용한다. 위치 성공률·GT영역 질량·point NLL·ECE는 각각 다른 성질을 평가하므로 하나의 개선으로 다른 항목의 개선을 주장하지 않는다.

## 부작용, 강건성, 일반화 한계

사전 지정 fixed_refine의 R@1m 변화 +1.896 pp [+1.213 pp, +2.674 pp], R@0.1m 변화 +0.485 pp [+0.291 pp, +0.734 pp].
시각 단독 성공 사례에서 baseline 대비 성공률 변화 +0.647 pp, CI [+0.252 pp, +1.101 pp]. 사전 비열등성 margin을 지정하지 않았으므로 형식적 비열등성을 주장하지 않는다.
두 seed는 같은 관측을 반복 평가한 것이며 독립 표본 수를 두 배로 세지 않았다. CI는 관측별 seed 평균 후 장면을10,000회 paired resampling한 것이다. 장면 내 상관은 반영하지만11개 장면과2seed의 한계, 장면 모집단 대표성은 남는다. 개발 CI는 반복 탐색의 선택 편향을 제거하지 않는다.
Euclidean0.4m 이웃은 벽을 가로지를 수 있으며 connectivity를 강제하지 않는다. 작은 위치·yaw 허용 범위와 full-bank 정규화, 후보 밀도에 결과가 의존할 수 있다. 시간적 연속성이나 실측 음향 도메인 이동은 평가하지 않았다. 외부 Test 확인은 수행하지 않았다.

## 일반 원리와 다음 반증 실험

1. 먼저 후보가 없는 문제와 후보를 고르지 못하는 문제를 분리한다. scalar-weight oracle과 separated-hypothesis oracle을 함께 측정하면 투자 방향을 좁힐 수 있다.
2. loss가 감독하는 결정과 실제 decoder의 결정을 맞추되, label 변화가 다른 decoder까지 개선하는지 검사해 단일 원인 설명을 경계한다.
3. 국소 질량을 최대화하는 결정은 point MAP와 목적이 다르다. 질량 기반 중심 선택 후 원점수 정밀화는 coarse 성공과 fine 정밀도를 분리해서 다루는 방법이며, 지표별 tradeoff를 보고해야 한다.
4. 분포를 정규화했다고 확률 보정이 된 것은 아니다. calibration과 ranking/decision은 별도 검증해야 한다.
5. 다음 반증: 벽 연결성을 보존한 이웃에서 gain이 남는가; 같은 특징·capacity로 separated 후보의 성공영역을 학습하면 weight-only gate보다 낫는가; label smoothing을 통제해 support-target 이득의 원인을 분리할 수 있는가; 완전히 미사용 데이터에서 frozen decoder 이득과 시각 보존이 재현되는가. 새 검증 자료를 확보하기 전에는 이미 본 Val11을 다시 untouched라고 부르지 않는다.

## 산출물과 재현

`confirmation_frozen.json`, `expanded_confirmation_replay.json`, `confirmation_summary.json`, `supplementary_effects.json`, `controls_summary.json`, `frontier_summary.json`, `validation_scope_audit.json`, `preservation_audit.json`이 핵심 근거다. `reproduce.md`에 실행 순서와 의존성, 실패 smoke의 원인이 있다. 기존 portable decoder는 `../astra_fusion_v2_20261107_0225/inference.py`와 그 frozen config를 사용한다. 이번 연구는 기존 baseline을 덮어쓰지 않는다.

[Research results](artifacts.json) (외부 산출물: `${WORKSPACE}/experiments/fusion_vs_posterior_20261107_1300/research_results.png`)

## 확인 결과의 해석과 장면별 부작용

Val11에서 support-target 학습으로 바꾼 효과는 support decoder에서 +0.139pp(CI가0을 포함), MAP에서 −0.199pp였다. 개발 자료에서의 gate 개선이 모든 결정 방식에 일반화된다는 주장은 반증됐다. 두 gate 모두 fixed support보다 우월하다는 증거는 없다. 그러나 gate×decoder의 음의 상호작용은 새 장면에서도 유지됐다.
원래 support 중심을 refinement하면 Val11 동시 성공은0.113pp 감소하는 대신 R@0.1m를 회복한다. 따라서 joint만 최적화하는 목적에는 raw support가 더 낫고, 사전 지정 refined 방법은 coarse/fine 성능을 함께 개선하는 선택이다. 확인 결과를 보고 방법을 바꾸지는 않았다.

| Scene | fixed MAP joint | refined joint | 변화pp | 시각 성공 손실 수 변화/seed |
|---|---:|---:|---:|---:|
| 102344115 | 54.688% | 56.406% | +1.719 | -2.5 |
| 102344307 | 69.231% | 72.957% | +3.726 | -2.5 |
| 104862573_172226682 | 75.521% | 77.431% | +1.910 | -1.5 |
| 105515286_173104287 | 63.728% | 68.527% | +4.799 | -2.5 |
| 106366134_174226362 | 66.524% | 66.695% | +0.171 | -0.5 |
| 106366173_174226431 | 57.947% | 58.579% | +0.633 | +1.5 |
| 106879005_174887124 | 80.234% | 81.875% | +1.641 | -2.0 |
| 108294624_176710203 | 66.875% | 69.297% | +2.422 | -2.0 |
| 108294765_176710386 | 74.576% | 75.318% | +0.742 | +0.0 |
| 108294783_176710410 | 60.041% | 62.090% | +2.049 | -0.5 |
| 108736611_177263226 | 75.208% | 78.750% | +3.542 | -5.0 |

두 seed 평균에서는11개 장면 모두 joint가 개선됐다. 이 문장은 각 seed의 각 장면이 모두 개선됐다는 뜻이 아니다. 평균 시각 성공 손실 수는207.5→190으로 줄었지만 개별 장면의 손실은 위 표처럼 따로 확인해야 한다.

### Seed와 실패 집단

| Seed | fixed MAP joint | refined joint |
|---|---:|---:|
| 42 | 67.931% | 69.766% |
| 43 | 68.225% | 70.390% |

| 단독 모달리티 성공 집단 | 관측 수/seed | fixed MAP joint | refined joint |
|---|---:|---:|---:|
| both | 771.5 | 96.760% | 97.343% |
| vision_only | 1932.0 | 90.553% | 91.226% |
| audio_only | 702.0 | 71.949% | 74.920% |
| neither | 2369.5 | 39.272% | 42.523% |

확인 control에서는 XY-only가69.784%, yaw-only가68.208%였다(fixed MAP68.078%). 공간 집계가 대부분의 gain을 설명하며 yaw 집계만으로는 같은 이득을 얻지 못했다. 이웃 수로 정규화해도70.026%이므로 후보 수 prior만으로 전체 개선을 설명하기 어렵다. 올바른 음향 대신 shuffle을 적용한 refined 결과는29.359%로 낮아졌고, visual-only refinement는47.290%였다. 결합 정보의 필요성은 유지된다.

# 현재 지지되는 결론과 미해결 질문

최종 근거 범위: 2026-11-08 HSSD 후속 연구까지. 모든 수치는 해당 실험 조건 안에서 해석한다. 독립 Test 성능 또는 전체 pipeline 미노출 일반화를 주장하지 않는다.

## 1. 후보 geometry를 이용한 점수 해석은 일관적인 개선 근거가 있다

[구조적 증거 연구](track-a/experiments/2026-11-07-geometry-discrimination/results.md)는 후보 형상과 기존 multimodal 점수의 결합으로 ranking을 개선했다. Candidate-only 및 donor 대조 때문에 이를 새로운 관측 기반 physical structural consistency의 발견으로 해석하지 않는다. GEOQ는 이미 acoustic/fused score와 음향 영향을 받은 후보를 사용한다.

[HSSD 후속 연구](track-a/experiments/2026-11-08-hssd-scaling/results.md)에서 OOF 입력과 장면별 동일 가중치를 맞춘24full GEOQ−score-only의 joint 차이는 **+1.218%p [95% CI +0.510,+2.051]**였다. 과거 threshold를 공통 적용한 보조 비교도 **+1.120%p [+0.476,+1.871]**였다. 두 시드 모두 양수다. 이는 기존 GEOQ보다 좋은 새 모델을 찾았다는 뜻이 아니다. [원수치](track-a/experiments/2026-11-08-hssd-scaling/evidence/recipe_control_results.json)

## 2. 데이터 확대 이득은 학습 가중치에 민감하다

HSSD의 OOF/query-weight 비교에서는12full→24full GEOQ가 **+0.506%p [+0.271,+0.742]**였다. 하지만 기존 장면별 가중치와 threshold를 복원하면 **+0.211%p [−0.147,+0.524]**로 불확실했다. 원본12full GEOQ의 점수와 선택을 정확히 재현한 뒤 비교한 결과다. 총 query를 고정한 장면 다양성 확대도 안정적인 이득을 확인하지 못했다.

따라서 초기 pilot 및 확대 단계의 양의 결과를 삭제하지 않되, **학습 방식에 불변한 scaling 효과**라는 해석은 채택하지 않는다. 장면 수, 관측 수, 장면별 loss 가중치는 분리해서 비교해야 한다. [OOF 결과](track-a/experiments/2026-11-08-hssd-scaling/evidence/oof_results.json) · [조건 복원 결과](track-a/experiments/2026-11-08-hssd-scaling/evidence/recipe_control_results.json)

## 3. 음향 정보의 유용성과 명시적 interaction의 필요성은 다르다

[Modality coverage](track-a/experiments/2026-11-07-modality-coverage/results.md)와 [pairwise 연구](track-a/experiments/2026-11-07-pairwise-acoustic/results.md)는 음향이 시각적으로 모호한 후보를 구분하는 정보를 제공함을 지지한다. 높은 Top20 coverage는 정답 선택 성능과 다르다.

[방향성 연구](track-a/experiments/2026-11-08-directional-interaction/results.md)의 A→V는 일부 visual ranking을 개선했지만 동일 정보의 concat/scalar보다 안정적으로 우수하지 않았다. HSSD OOF24full에서도 D2−concat은 **−0.163%p [−0.437,+0.108]**였다. Matched−donor가 양수여도 gate-disabled 대비 이득은 확인되지 않았다. 올바른 audio에 의존한다는 것만으로 interaction 구조의 추가 기여가 입증되지는 않는다.

실제 visual residual adapter의 depth 품질 변화와 최종 pose 개선도 구분한다. 이번 비교에서는 acoustic-conditioned adapter의 최종 localization 우위를 확립하지 못했다. 모든 fine-tuning이나 모든 cross-modal 학습이 불가능하다는 결론은 아니다.

## 4. 초기 개선 신호가 후속 조건에서 유지됐는지 확인해야 한다

| 이전 관측 | 후속 검증과 현재 판정 |
|---|---|
| [v2 Support+refinement](track-a/experiments/2026-11-07-support-refinement/results.md)가 B2보다 Val8 joint +1.587%p | [Fusion/posterior 분리](track-a/experiments/2026-11-07-fusion-vs-posterior/results.md)에서 고정 decoder 개입으로 재검토. 학습형 gate 성공이나 pipeline 독립 일반화로 부르지 않는다. |
| [Directional E0+GEOQ](track-a/experiments/2026-11-08-directional-interaction/results.md)의 개발 Val11 +0.442%p | HSSD11의128query pilot +0.675%p 신호가 전체6,072query에서는 +0.345%p [−0.081,+0.821]. 확인 기준 실패, 새 강한 baseline으로 확정하지 않는다. |
| [Residual R3 pilot](track-a/experiments/2026-11-08-residual-spatial/results.md)의 +0.228pp | 전체query에서 −0.321pp. Pilot 신호를 확정 개선으로 보존하지 않는다. |
| HSSD query-weight scaling 양의 결과 | 기존 scene-weight 복원 후 CI가0 포함. 데이터 다양성만의 효과라는 해석을 제한한다. |

HSSD 단일 ambiguity gate도 실패한 결과를 보존했다. Threshold를 추가 탐색해 양의 결과를 만들지 않았다.

## 5. 음향의 물리적 해석에는 입력 계약이 필요하다

[음향 단서 연구](track-a/experiments/2026-11-07-acoustic-cues/results.md)의 시간/envelope와 초기 복소 채널 관계는 판별 정보를 가진다. 그러나 중앙 source·5cm ring·정규화 입력에서 절대 음량, 특정 반사 또는 정확한 DoA를 읽는다는 주장은 지지되지 않았다. C의 두 번째 구간은 late-only가 아니라 full window다.

[Alignment 연구](track-a/experiments/2026-11-07-alignment-discrimination/results.md)는 positive distance 감소가 negative와의 분리나 최종 pose 성공을 보장하지 않음을 보였다. [Sampling 연구](track-a/experiments/2026-11-07-candidate-sampling/results.md)는 실제 경쟁 후보와 학습 negative의 차이를 확인했지만 sampling 변경만으로 GEOQ를 개선하지 못했다.

[Residual/spatial 연구](track-a/experiments/2026-11-08-residual-spatial/results.md)의 좌표·Fourier·angular 대조는 수학적 정합과 실제 판별 향상을 구분한다. 10° 각도 격자가10° 물리 분해능을 뜻하지 않는다. HSSD FP acoustic reference는 실제 stage-only mesh가 아닌2D map extrusion이며 native furnished 관측과 수직구조 근사도 다르다.

## 남은 질문과 후속 연구 조건

- 추가 정보가 기존 GEOQ 이후에도 남는지: matched capacity, donor/no-audio, 실제 후보 reference를 함께 통제해야 한다.
- 더 많은 장면이 도움이 되는지: 동일 query 예산 및 scene weighting, 새로운 독립 평가 장면을 함께 확보해야 한다.
- 완전히 새 공간에 일반화되는지: visual teacher 노출을 피한 평가 또는 upstream 재학습이 필요하다. 기존 Test를 탐색 자료로 전환하지 않는다.
- 48/96scene dense-bank 확대는 아직 수행하지 않았다. HSSD 감사에서 실제 준비된 자산 범위와 생성 비용을 확인했다.

이 질문들은 가능한 다음 연구이며, 이 아카이브를 읽는 것만으로 실행 승인을 받은 것으로 해석하지 않는다.

## 2026-11-09 Track A 후속: Directional Evidence / Tri-branch Pilot

[후속 결과](track-a/experiments/2026-11-09-directional-tribranch/RESULTS.md)는 음향만으로 coarse near/far77.67%(prior62.00%)를 예측하지만 GEOQ 이후 추가 localization 이득을 확인하지 못했다. Attention joint +0.065pp는 불확실하고 fine -1.204pp이며, 실제 부분 음향 학습과 시각 어댑터·저학습률 내부 검증 후에도 B2+GEOQ를 유지했다.12개 reused/teacher-exposed scenes,2seeds의 탐색 결과다. 기존 B2 Top20/FP-RIR reference 의존 Pilot이므로 Track A이며 새 FP-RIR-free Track B의 검증 결과로 읽지 않는다. 이번 문서 이관으로 실험을 재실행하거나 기존 결론을 삭제하지 않았다.

## 2026-11-09 Track A: Directional Structural Evidence

[새 연구](track-a/experiments/2026-11-09-directional-structure/README.md)는 raw·ECN·frozen embedding·회전·donor·동일용량 비방향 대조와73새checkpoint를 검증했다. 고정 pureV후보 재정렬이 New12 joint +3.125pp, frozen Transfer11 +3.267pp로 개선됐다. 기존cosine/GEOQ보다 낮으며 fullbank fine은 악화됐다. 따라서 거친 방향 단서의 모호성 해소를 지지하지만 metric depth/DoA나 강한 baseline 대체는 입증하지 않는다. 모두 reused/teacher-exposed Exploratory이며 기존Test는 사용하지 않았다.

## Track B 추가 근거 — 2026-11-09 encoder 업데이트 범위

[27개 matched 모델 비교](track-b/experiments/2026-11-09-foundation-regimes/completed_results.md)에서 frozen/partial도 초기 독립F1보다 Joint6.77–7.73%p 낮았다. 모든 초기 텐서와 고정 텐서 보존을 확인했으므로 인코더 변화만으로 저하를 설명할 수 없다. 새 결합부와 학습 가능한 점수 스케일은 여전히 바뀌므로 어느 쪽이 원인인지는 미검증이다. 이는16Train/6재사용Val의 Exploratory 결과이며, 규모 확대·독립평가·기작 성공을 뜻하지 않는다.

[고정 모델의 점수 개입](track-b/experiments/2026-11-09-frozen-score-controls/completed_results.md)에서 초기 스케일 복원은 Joint2.34–2.78%p, 학습 게이트/잔차 제거는5.21–5.99%p를 회복했다. 둘을 복원하면 초기F1가 재현된다. 따라서 해당 고정 모델·재사용6장면에서는 두 변경 그룹이 모두 저하에 기여한다. NLL은 원래 결합 모델이 더 낮아 지표 간 tradeoff가 확인되지만, 학습 경로의 원인·물리 기작·일반화는 확정하지 않는다. 결합부 제거로 얻은 회복은 새로운 interaction의 성공이 아니다.

[Track B 독립F1의 규모×학습량 비교](track-b/experiments/2026-11-09-expanded-cohort/completed_factorial_results.md)는36fits와19Val의4조건을 완료했다. 같은업데이트에서16→112장면은 Joint+7.24/+8.58%p를 보였지만, 업데이트만7배 늘리면16장면에서는−2.14%p,112장면에서는불확실했다. 이 nested cohort에서는 추가최적화보다 장면구성 확대가 유리했다는 근거이며, 장면수·내용다양성의 보편적인과효과나 M1결합성공·독립일반화를 증명하지 않는다.

## 2026-11-11 추가 — Track A Experiment 1의 물리 시간 정렬과 energy 연관성을 구분

[Reflection observability](track-a/experiments/2026-11-11-reflection-observability/RESEARCH_REPORT.md)는96개 통제/진단 RIR와HSSD12개 개발 장면의실제visual Top20을분석했다. 벽 제거·이동에따른음향응답은존재하지만c343의좁은first-order시간창대응은실패했다. 단일벽2m/6m에서peak bias약+2.36/+6.15ms가sampling rate·diffraction·depth·ray count·mesh형식변경에도남았다. HSSD에서는time/distance marginal을보존한query permutation보다power+0.2619[0.1671,0.3681]이나coherence는개선되지않았다. 이는조건부energy연관성이며확인된wall identity또는localization gain이아니다. 증거수준Exploratory;Test payload·새모델/scorer·Experiment2–4는수행하지않았다.

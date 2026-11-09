> 역사적 연구 기록. 원본: `candidate_sampling_20261107_2337/RESEARCH_REPORT.html`. 원본 SHA256은 [산출물 목록](artifacts.json)에 있습니다.
> 당시의 완료·재개 지시는 현재 실행 요청이 아닙니다. 최신 판정은 [현재 결론](../../../current_findings.md)을 먼저 확인하세요.

# 음향 후보 판별: 기존 근거와 통제 후속 실험

## 결론

**학습 negative와 실제 경쟁 후보의 차이는 크지만, candidate-aware sampling만으로 강한 GEOQ를 개선하지 못했다.** 100개 새 encoder와 210개 matched utility heads를 평가했다. Multi-positive 목표는 candidate-region exact 목표 대비 Train held-scene pooled Top1을 5.47pp 회복했지만, 같은 reference bank의 원래 sampling보다 여전히 4.36pp 낮았다. 이는 허용 범위 충돌의 일부 완화를 보여주며 최종 개선과 다르다.

동일 Val11 표본에서 original GEOQ joint는 72.30%, 가장 높은 새 encoder 점수 추가 모델(acoustic_safe)은 71.16%였다. 차이 −1.14pp의 scene CI [−2.41,+0.28]는 개선 근거를 제공하지 않는다. 기존 GEOQ를 유지하며 end-to-end 교체나 Test 평가로 진행하지 않는다. 아래 수치는 기존 full Val11의 71.818%와 평가 표본이 다르다.

## 연구 범위와 평가 정직성

음향 정보의 존재, 반복 안정성, GEOQ 이후 조건부 가치, 최종 선택 기여를 분리했다. 앞선 pairwise 연구의 전체 query 평가와 이번 candidate-learning 후속 실험을 구분한다. 이번 후속 모델은 동일 64 query/scene 표본을 사용하며, full-data GEOQ도 정확히 같은 표본에서 비교한다. Train12 outer folds, 두 encoder seeds, 기존 두 proposal seeds를 사용했다. Val8/11은 재사용 development, acoustic Val3 및 visual Val19의 과거 노출은 남는다. Test는 사용하지 않았다. 신뢰구간은 scene 내 seeds를 평균한 paired scene bootstrap이며 탐색·다중 비교 편향을 제거하지 않는다.

## 1–4. 반복성과 모호한 후보의 음향 판별

기존 [pairwise 연구](../2026-11-07-pairwise-acoustic/results.md)는 반복 RIR, T/N/C 및 intermediate/final embedding, visually ambiguous pairs, donor 및 removal을 평가했다. Val11 ambiguous-pair accuracy는 visual 51.46%, pure acoustic embedding 76.02%, GEOQ 89.61%였다. 이는 음향 판별 근거의 존재를 보이지만 GEOQ 이후 추가 정보의 증명이 아니다. 동일 pose의 반복 변동과 Native/FP 차이도 분리했다. 현재는 기존 12 scenes×2 queries의 2,736 renders를 새 encoder로 다시 평가했다. 렌더링의 seed 독립성이나 실환경 일반화를 주장하지 않는다.

[모든 domain·repeat·query 결과](artifacts.json) (`matrix_analysis.json`), [거리·상대 분리·검색 집계](artifacts.json) (`domain_summary.json`), [paired domain interaction](artifacts.json) (`domain_interactions.json`), [reference 조건 변화에 따른 모든 선택 사례](artifacts.json) (`domain_reference_cases.json`). 두 query/scene의 작은 표본이므로 domain mismatch를 지배적 원인으로 확정할 수 없다.

## Native/FP 조건을 바꾼 동일 후보 판별

| Encoder | Native→Native % | FP→FP % | Native→FP % | FP→Native % |
| --- | --- | --- | --- | --- |
| best | 56.25 | 51.04 | 42.71 | 29.17 |
| nce | 57.29 | 48.96 | 41.67 | 17.71 |
| bank_original | 51.04 | 46.88 | 38.54 | 21.88 |
| bank_safe | 51.56 | 48.96 | 40.10 | 22.92 |
| fusion_safe | 57.81 | 47.92 | 28.12 | 25.52 |
| acoustic_safe | 50.00 | 40.10 | 26.04 | 18.23 |
| mixed_safe | 51.04 | 41.15 | 34.90 | 22.92 |
| balanced_safe | 54.69 | 45.83 | 28.65 | 26.04 |
| bank_multi | 50.52 | 44.79 | 39.06 | 20.83 |
| region_exact | 52.60 | 39.06 | 29.17 | 21.88 |
| region_multi | 55.73 | 44.79 | 29.69 | 31.25 |

동일 24 native query poses와 동일 후보 위치를 두 domain에서 반복 렌더링한 진단이다. Native→Native는 배포 가능한 reference 조건으로 간주하지 않는다. Fusion sampling의 bank_original 대비 변화는 Native→Native +6.77pp, Native→FP −10.42pp였으나 각 신뢰구간은 0을 포함한다. Balanced와 region-multi도 domain에 따라 다른 점추정을 보였지만, 작은 표본으로 원인을 확정할 수 없다. 거리·분리·repeatability와 paired scene 구간은 연결된 JSON에 있다.

## 학습 구현과 실제 경쟁 후보 감사

6-channel E/C/N 입력, normalized 128D 출력, temperature 0.07, exact-pose/yaw positive를 확인했다. Historical best는 Val3로 선택된 epoch4이며 validation replay의 loss/Top1/Top5/MRR 차이는 모두 0이었다. Visual epoch20 기반 mining은 이미 존재하던 기능이며 신규 기여로 제안하지 않는다. 역사적 positive+7 negatives 중 4 spatial 및 2 orientation mining을 6,387 queries 전체에서 정확히 재현했다. Random draw의 원래 실행 소스는 복원하지 못했으며 immutable records와 protocol을 사용한다. 기존 negatives의 strict 1m/30° false-negative는 0/44,709, epsilon boundary는 별도 집계했다. 실제 support의 오답은 기존 negatives보다 양성을 더 자주 이겼다. 이것은 학습 후보와 추론 후보 차이의 근거이며 곧바로 mining 병목이라는 결론은 아니다. 64-query/scene full-posterior 대조에서도 실제 오답 후보의 .4m/±10° 주변 log mass는 기존 negatives보다 평균 6.335 높았다(scene CI [5.907, 6.727]). 기존 positive 열은 이 비교에서 제외했다.

원래 recording을 common dense bank에 투영하는 bridge는 최대 20cm 이동과 reference recording 변경을 포함한다. 이후 sampling-only 비교는 bank_original과 비교한다. 투영으로 일부 원래 negative가 tolerance-equivalent가 되었으므로 bank multi-positive 개선은 원래 역사적 데이터의 false-negative 문제를 증명하지 않는다. 초기 파라미터, batches, steps, 2 epochs, AdamW, 7 negative slots, NCE temperature, 모델 용량을 맞췄다. 모든 source/checkpoint/policy 해시와 실패 로그를 보존했다. [mining 재현](artifacts.json) (`mining_replay.json`), [validation 재현](artifacts.json) (`training_contract_replay.json`), [경쟁 후보 감사](artifacts.json) (`audit_distribution_analysis.json`), [held-scene encoder 반증](artifacts.json) (`held_encoder_audit.json`).

## 5–7. 조건부 가치와 최종 localization

동일 fixed20, 동일 utility joint+0.5fine, 172 columns, 150-iteration/15-leaf learner로 재학습했다. GEOQ는 이미 acoustic scalar/fused score/음향 영향을 받은 후보를 사용하므로 acoustic-free baseline이 아니다. 점수 추가는 B2 encoder 교체나 end-to-end 개선으로 해석하지 않는다. Matched base 개선과 강한 original GEOQ 개선을 함께 표시한다.

| Model | Joint % | Fine % | R@1m % | Δ original GEOQ pp [95% CI] | Δ matched base pp [95% CI] | Visual preservation % | Repair/harm |
| --- | --- | --- | --- | --- | --- | --- | --- |
| original_GEOQ | 72.30 | 25.78 | 74.29 | +0.00 [+0.00, +0.00] | +2.41 [+0.92, +3.91] | 93.49 | 0/0 |
| base | 69.89 | 24.64 | 72.30 | -2.41 [-3.84, -0.92] | +0.00 [+0.00, +0.00] | 93.28 | 19/53 |
| best | 70.67 | 25.43 | 73.22 | -1.63 [-3.20, +0.07] | +0.78 [-0.43, +1.99] | 93.29 | 23/46 |
| legacy | 70.60 | 25.36 | 73.01 | -1.70 [-3.12, -0.14] | +0.71 [-0.64, +1.85] | 93.17 | 24/48 |
| bank_original | 70.31 | 25.21 | 72.66 | -1.99 [-3.34, -0.50] | +0.43 [-0.50, +1.28] | 93.00 | 19/47 |
| bank_safe | 70.60 | 25.07 | 73.15 | -1.70 [-2.98, -0.57] | +0.71 [-0.43, +1.92] | 92.83 | 17/41 |
| fusion_safe | 70.81 | 25.07 | 73.44 | -1.49 [-2.70, -0.07] | +0.92 [-0.07, +1.92] | 92.79 | 17/38 |
| acoustic_safe | 71.16 | 25.64 | 73.58 | -1.14 [-2.41, +0.28] | +1.28 [-0.07, +2.49] | 93.15 | 25/41 |
| mixed_safe | 69.82 | 25.43 | 71.95 | -2.49 [-4.12, -0.85] | -0.07 [-1.42, +1.42] | 93.09 | 20/55 |
| balanced_safe | 69.89 | 25.28 | 72.23 | -2.41 [-3.62, -1.07] | +0.00 [-1.21, +1.21] | 92.27 | 17/51 |
| bank_multi | 70.24 | 25.36 | 72.51 | -2.06 [-3.27, -0.71] | +0.36 [-1.07, +1.63] | 93.13 | 20/49 |
| region_exact | 70.17 | 25.07 | 72.66 | -2.13 [-3.84, -0.50] | +0.28 [-1.14, +1.70] | 93.17 | 20/50 |
| region_multi | 70.60 | 25.78 | 72.59 | -1.70 [-2.84, -0.28] | +0.71 [-0.64, +1.92] | 93.05 | 22/46 |

그림: Paired downstream contrasts — 외부 산출물 `research_summary.png` ([목록](artifacts.json)).

[Train/Val8/Val11 및 scene/seed 통계](artifacts.json) (`discriminator_all_analysis.json`), [seed별 집계](evidence/downstream_seed_summary.json). 전체 cohort에서 재현된 양의 근거 없이 기존 모델을 교체하거나 Test를 열지 않는다.

## 학습된 표현의 Train12 held-scene 결과

| Encoder | Pool Top1 % | Top5 % | Top20 % | AUC % | Fixed20 acoustic Top1 % | Ambiguous pair % | Variance | Effective rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| acoustic_safe | 10.35 | 29.36 | 58.53 | 77.67 | 25.16 | 68.22 | 0.529 | 12.26 |
| bank_original | 20.83 | 49.80 | 78.45 | 86.10 | 29.95 | 71.05 | 0.820 | 15.89 |
| bank_safe | 21.03 | 50.13 | 79.04 | 86.12 | 30.37 | 71.67 | 0.815 | 15.71 |
| best | 33.20 | 68.62 | 91.67 | 89.54 | 40.95 | 78.99 | 0.893 | 19.56 |
| fusion_safe | 15.04 | 37.17 | 63.74 | 80.84 | 28.94 | 70.61 | 0.658 | 13.45 |
| legacy | 21.55 | 51.82 | 79.56 | 86.54 | 30.86 | 71.74 | 0.804 | 15.09 |
| mixed_safe | 15.89 | 39.39 | 69.47 | 82.42 | 27.21 | 68.41 | 0.568 | 14.24 |
| balanced_safe | 17.71 | 44.27 | 72.85 | 84.28 | 29.00 | 70.34 | 0.622 | 14.24 |
| bank_multi | 21.16 | 49.67 | 78.78 | 86.09 | 30.96 | 71.04 | 0.813 | 15.80 |
| region_exact | 11.00 | 31.97 | 58.85 | 77.85 | 25.46 | 68.56 | 0.592 | 12.59 |
| region_multi | 16.47 | 39.84 | 67.51 | 82.36 | 28.97 | 71.71 | 0.584 | 12.11 |

역사적 best는 Train scenes에 노출된 비교점이며 held-scene encoder가 아니다. Pool은 GT 없이 고정한 기존 supports의 합집합으로, full dense-bank retrieval과 다르다. Fixed20 acoustic Top1은 음향 단독 선택이며 GEOQ와 결합한 위 표의 downstream joint와 다르다. Scene/encoder seed/proposal seed를 동일하게 평균했다.

## Sampling × objective의 분리

| Contrast | Metric | Train Δ pp [95% CI] |
| --- | --- | --- |
| region_multi − region_exact | pool_top1 | +5.47 [+2.47, +8.40] |
| region_multi − region_exact | fixed_joint | +3.52 [+0.65, +6.28] |
| (region_multi-region_exact)-(bank_multi-bank_original) | pool_top1 | +5.14 [+2.08, +8.14] |
| (region_multi-region_exact)-(bank_multi-bank_original) | fixed_joint | +2.51 [-0.46, +5.50] |

Multi-positive의 회복 효과는 candidate-region에서 더 크다. 원래 historical negatives의 strict false-negative가 0이었다는 사실과 충돌하지 않는다. Candidate-region과 투영된 bank에서 새로 생긴 tolerance-equivalent 후보를 다루는 실험이기 때문이다.

## 추가 음향 block의 donor 재학습 대조

| Split | Real-query arm | Donor | Real − donor joint pp [95% CI] |
| --- | --- | --- | --- |
| train | fusion_safe | fusion_safe_donor1 | +0.59 [-0.26, +1.43] |
| train | fusion_safe | fusion_safe_donor17 | +0.00 [-0.85, +0.78] |
| train | fusion_safe | fusion_safe_donor31 | +0.26 [-0.59, +1.04] |
| train | balanced_safe | balanced_safe_donor1 | -0.72 [-1.37, -0.07] |
| train | balanced_safe | balanced_safe_donor17 | -0.07 [-0.78, +0.65] |
| train | balanced_safe | balanced_safe_donor31 | -0.33 [-1.04, +0.33] |
| train | region_multi | region_multi_donor1 | -0.13 [-0.78, +0.46] |
| train | region_multi | region_multi_donor17 | +0.07 [-0.78, +0.91] |
| train | region_multi | region_multi_donor31 | -0.52 [-1.11, +0.13] |
| extra | fusion_safe | fusion_safe_donor1 | +0.57 [+0.00, +1.14] |
| extra | fusion_safe | fusion_safe_donor17 | +0.28 [-0.78, +1.21] |
| extra | fusion_safe | fusion_safe_donor31 | +0.57 [-0.43, +1.63] |
| extra | balanced_safe | balanced_safe_donor1 | -0.50 [-1.63, +0.50] |
| extra | balanced_safe | balanced_safe_donor17 | -0.92 [-2.70, +0.64] |
| extra | balanced_safe | balanced_safe_donor31 | -0.78 [-2.06, +0.57] |
| extra | region_multi | region_multi_donor1 | +0.07 [-0.92, +1.07] |
| extra | region_multi | region_multi_donor17 | +0.36 [-0.92, +1.63] |
| extra | region_multi | region_multi_donor31 | +0.36 [-0.85, +1.63] |

세 donor permutation의 일관성을 함께 확인하며 한 permutation의 유리한 값만 선택하지 않는다. Existing GEOQ의 acoustic 정보를 제거한 대조가 아니다.

## 통제·반증과 학습 목표

bank_safe는 허용 범위 양성 제거, fusion_safe/acoustic_safe는 실제 support 경쟁 후보, mixed_safe는 orientation/random을 유지한 혼합, balanced_safe는 기존 spatial 다양성을 더 유지한 낮은 hard-negative 비율이다. Warm-start의 동일 초기 checkpoint 대조는 처음부터 학습하는 어려움만으로 성능 저하가 설명되는지 검사한다. 2×2 factorial은 bank/candidate-region sampling과 exact/multi-positive objective를 분리한다. Multi-positive는 최소 한 개 실제 negative를 유지한다. Loss 하락만 보지 않고 retrieval, margin, variance/effective rank, pair accuracy, fine precision을 함께 평가한다.

[paired encoder 및 factorial interaction](artifacts.json) (`paired_encoder_contrasts.json`), [두 seed warm pilot](artifacts.json) (`warm_sampling_bank_analysis.json`). Same-scene donor shifts 1/17/31은 query embedding을 교체하고 recipient reference에 다시 scoring하며 score-slot shuffle이 아니다. Fusion, balanced, region-multi에 대해 동일 재학습 대조를 유지한다. 이전 연구의 frozen removal과 새 donor 재학습은 서로 다른 개입이다.

## 8. Rescue, harm와 환경별 실패

Ambiguity threshold는 전체 Train의 GT-free 분포에서 고정했으므로 strict outer-fold threshold는 아니다.

[Train에서 고정한 ambiguity, visual20 failure/acoustic success, both unimodal failure, strong wrong acoustic, GEOQ-failure 분석](artifacts.json) (`subgroup_all_analysis.json`)과 [모든 query 선택 및 성공 labels](artifacts.json) (`discriminator_all_queries/`)를 제공한다. 실패 조건으로 정의한 subgroup의 repair만 전체 이득으로 해석하지 않는다. 기존 fullbank unimodal 결과와 새 restricted pooled acoustic bank를 명확히 구분한다. 기존 pairwise 연구의 rescue/harm 지도와 decision-dependence 결과도 보존했다.

## 검증한 설명과 남은 경쟁 가설

| 설명 | 근거 | 판정 |
| --- | --- | --- |
| 기존 negatives가 실제 경쟁 후보를 충분히 대표한다 | 낮은 overlap, 실제 wrong 후보의 높은 hardness 및 posterior support; held encoder 대조 | 분포 차이 확인 |
| Hard 후보로 바꾸면 최종 localization이 개선된다 | 기본·mixed·balanced 정책, 두 seed·4 folds, strong GEOQ 비교 | 시험한 정책/예산에서 지지되지 않음 |
| Scratch 초기화만이 hard 정책 실패의 원인이다 | 동일 held-scene warm checkpoint에서 2 epoch 추가, 두 seed fold0 pilot | Pilot에서 충분한 설명이 아님 |
| Tolerance 목표가 candidate-region의 충돌을 줄인다 | Region multi − exact Train pooled Top1 +5.47pp; factorial interaction | 일부 회복 확인; bank_original/strong GEOQ 개선은 아님 |
| Hard 후보 비율을 낮추고 다양성을 유지하면 해결된다 | Balanced − bank_original Train pooled Top1 −3.13pp | 이번 low-dose 정책에서 지지되지 않음 |
| Domain mismatch가 유일하거나 지배적인 원인이다 | Native/FP 조건에 따라 다른 변화, 작은24query표본과 넓은CI; 기존 alignment 실험도 판별력 저하 | 미해결; 단일 원인 단정 불가 |
| 새 score block의 이득이 안정적인 조건부 음향 정보다 | 세 donor 재학습 대조와 strong baseline 비교에서 일관된 우위 없음 | 현재 learner에서 재현 가능한 추가 이득 미확인; 정보 부재 증명 아님 |

## 9–10. 해석의 범위와 후속 투자

음향 정보는 존재하지만 기존 GEOQ 이후 추가 가치와 동일하지 않다. 더 어려운 negatives, tolerance targets, warm curriculum, 다양성 유지, donor, four-domain 대조로 경쟁 설명을 검증했다. 개별 실패가 모든 representation이나 learner의 불가능성을 증명하지는 않는다. Domain 차이는 geometry/material/furnishing이 함께 달라져 가구만의 인과 효과가 아니다. 거리 감소만을 alignment 개선으로 인정하지 않는다.

우선순위는 Front-end의 관측/reference 대응과 안정적 상대 분리, 다음은 실제 competing hypotheses에 대한 조건부 정보 보존이다. Back-end 용량 확대는 강한 GEOQ 대비 재현 가능한 gain이 확인될 때 진행한다. 추가 104 Train scenes의 계획은 존재하지만 upstream visual teacher의 116-scene 학습 범위 안이다. 독립적인 전체 pipeline 일반화로 부르지 않는다. 이번 실험은 짧은 동일 2-epoch 비교와 fold0 warm curriculum이며 모든 training schedule의 불가능성을 증명하지 않는다. 긴 사전 등록 curriculum, encoder feature의 cross-fitting/재보정, 새로운 reference 조건에서의 검증은 미해결이다. 같은 development 데이터에서 추가 policy/threshold sweep을 반복할 정보 가치는 현재 낮다. 새 scene 평가에는 모델·후보·threshold를 먼저 동결해야 한다.

## 재현·보존

[검증 결과](evidence/final_verification.json), [이전 8,554 산출물 보존](artifacts.json) (`prior_preservation_verification.json`), [등록 및 결정 기록](archive/PROTOCOL.md), [재현 명령](artifacts.json) (`REPRODUCE.sh`), [최종 artifact manifest](artifacts.json) (`artifact_manifest.json`). 원본 체크포인트 및 user 작업 파일을 유지했다. 실패한 I/O, shared-memory, 경로, precision 검증 로그도 삭제하지 않았다.

> 역사적 연구 기록. 원본: `directional_disambiguation_20261108_0123/RESEARCH_REPORT.html`. 원본 SHA256은 [산출물 목록](artifacts.json)에 있습니다.
> 당시의 완료·재개 지시는 현재 실행 요청이 아닙니다. 최신 판정은 [현재 결론](../../current_findings.md)을 먼저 확인하세요.

# 후보별 음향 증거의 효용과 명시적 교환의 한계

AV-FPLoc · Directional Cross-modal Interaction · 2026-11-08 · Reused development data, no Test

## 검증된 결과와 채택 결정

후보별 음향 증거를 제공하면 audio-conditioned visual 판별은 개선된다. 그러나 명시적 방향성 modulation이 같은 정보를 사용한 concat이나 단순 score 결합보다 유리하다는 근거는 확보되지 않았다. 실제 visual 중간 특징을 직접 보정하는 cross-attention도 올바른 audio에 특이적인 개선을 입증하지 못했다.

전체 query 후속 검증에서 교환 없는 E0+GEOQ는 Val11 71.818%→72.260%(+0.442%p, scene 95% [+0.167,+0.678])로 개선됐고 Val8도 +0.453%p였다. 반면 Train scene OOF는 −0.110%p로 개선을 확인하지 못했다. 이를 개발셋상 후보 개선으로 보존하며 독립 일반화나 명시적 교환의 성과로 주장하지 않는다. 기존 encoder·B2·Support-NMS·GEOQ는 유지한다. 첫 실패 후 실제 visual-feature 교환, audio-free 후보 검증, scalar 대조, 내부 scene 교차검증, 전체 query 학습까지 후속 실험했다. 명시적 교환에 대한 결론은 이 구조·목적·데이터 범위의 재현 가능한 반증이며 모든 cross-modal 학습의 불가능성을 뜻하지 않는다.

그림: Four-panel research comparison — 외부 산출물 `research_summary.png` ([목록](artifacts.json)).

## 1. 방향별 교환: 동일 Support-NMS20

| 방법 | Visual joint % | Acoustic joint % | Late fused joint % | V Top5 % | 높은 V모호성 % | 모호한 pair % |
| --- | --- | --- | --- | --- | --- | --- |
| E0: no exchange + late fusion | 48.08 | 37.22 | 56.11 | 78.48 | 38.65 | 60.84 |
| E1: V→A | 47.87 | 47.51 | 55.26 | 78.76 | 38.66 | 59.76 |
| E2: A→V | 55.97 | 37.00 | 56.82 | 83.95 | 46.67 | 65.56 |
| E3: bidirectional | 56.68 | 50.36 | 57.95 | 85.01 | 42.92 | 66.08 |
| E4: same-feature concat | 56.39 | 50.14 | 58.88 | 84.52 | 45.80 | 64.76 |
| A→V global message | 48.37 | 36.08 | 55.40 | 78.12 | 37.14 | 60.61 |
| E5: existing B2 reference | 50.71 | 39.06 | 67.12 | 78.20 | 40.23 | 62.34 |

64 query/scene, 2 seed, Val11의 동일 704개 query. E0–E4/global은 각각 30,466개의 실제 사용 파라미터, 동일 초기 seed·40 epochs·batch64·손실이다. E5는 보존된 기존 모델의 reference이며 학습 예산이 동일한 새 fit이 아니다. 수신 경로가 없는 branch도 다른 branch loss의 역전파로 재학습 후 달라질 수 있다. Frozen donor 불변성과 재학습 효과를 구분한다. 위 visual은 후보 집합에 B2 음향 영향이 남아 있는 제한된 ranking 평가다. Pure→B2 차이는 exchange 인과 효과로 해석하지 않는다.

## 2. 올바른 acoustic observation의 효과

| 후보 | A→V 비교 대상 | Train scene OOF Δpp [95%] | Val11 Δpp [95%] |
| --- | --- | --- | --- |
| fixed | primary_E0_real | +11.00 [+7.75, +14.32] | +7.88 [+4.62, +11.22] |
| fixed | primary_E4_real | -3.06 [-5.53, -0.85] | -0.43 [-1.99, +1.21] |
| fixed | primary_global_real | +10.94 [+7.16, +15.10] | +7.60 [+4.55, +10.58] |
| fixed | primary_E2_donor1 | +9.96 [+6.90, +13.15] | +11.72 [+8.59, +15.06] |
| fixed | primary_E2_donor17 | +11.00 [+7.55, +14.45] | +10.80 [+7.74, +14.28] |
| fixed | primary_E2_donor31 | +10.16 [+6.77, +13.41] | +10.16 [+6.82, +13.85] |
| fixed | primary_E2_noaudio | +7.68 [+4.88, +10.55] | +8.45 [+5.26, +12.07] |
| fixed | primary_E2_disabled | +9.31 [+6.12, +12.57] | +10.44 [+7.32, +13.78] |
| pure | primary_E0_real | +9.96 [+6.77, +13.35] | +5.75 [+3.20, +7.95] |
| pure | primary_E4_real | -0.46 [-1.50, +0.65] | -1.35 [-2.77, -0.14] |
| pure | primary_global_real | +9.83 [+6.90, +12.83] | +5.89 [+2.77, +8.81] |
| pure | primary_E2_donor1 | +11.39 [+9.18, +13.74] | +9.23 [+6.46, +12.00] |
| pure | primary_E2_donor17 | +11.26 [+8.79, +13.93] | +8.24 [+5.04, +11.43] |
| pure | primary_E2_donor31 | +10.81 [+8.92, +12.89] | +7.17 [+4.47, +9.94] |
| pure | primary_E2_noaudio | +8.72 [+6.25, +11.33] | +5.82 [+3.69, +8.03] |
| pure | primary_E2_disabled | +8.98 [+6.58, +11.52] | +7.60 [+5.54, +9.94] |

Pure 후보는 원래 pure-visual full-bank NMS20이며 audio/fused score로 생성하지 않았다. 동일 모델을 동결한 채 이 후보에 평가했으므로 후보 분포 변화도 포함한다. 각 donor는 같은 scene의 다른 query이며 recipient 후보의 acoustic reference와 다시 비교했다. Donor score의 후보 slot만 바꾸지 않았다. Frozen donor/removal, donor로 재학습, audio 제거 재학습은 별도 결과로 보존했다. 예측 순위와 정답·오답 confidence를 함께 분석했으며 entropy 감소만으로 성공을 판단하지 않았다.

## 3. 실제 visual feature 보정은 별도 검증

| 후보 | Frontend | Joint % | Top5 % | 높은 모호성 % | 모호한 pair % |
| --- | --- | --- | --- | --- | --- |
| fixed | self | 46.95 | 76.42 | 35.53 | 55.07 |
| fixed | cross | 46.31 | 75.07 | 35.45 | 55.15 |
| fixed | global | 46.73 | 76.78 | 36.35 | 54.73 |
| fixed | concat | 47.09 | 76.92 | 37.09 | 55.76 |
| pure | self | 43.75 | 70.03 | 35.91 | 58.94 |
| pure | cross | 42.97 | 69.25 | 35.17 | 57.61 |
| pure | global | 44.32 | 70.03 | 37.22 | 59.39 |
| pure | concat | 44.25 | 69.96 | 36.86 | 60.53 |

Frozen 40×128 visual feature에 제한된 residual을 더하고 기존 고정 depth head를 통과시켰다. Scale은 유지했다. Cross-attention의 Q는 visual40 tokens, K/V는 E/C/N52 tokens이며 폭32·4heads다. Global은 projected acoustic tokens의 평균, concat은 visual token과 acoustic 평균을 MLP에 전달한다. Self는 visual 특징에서 source를 구성하며 audio를 사용하지 않는다. Adapter는 FP geometry를 입력받지 않고, 기존 FP matching은 학습 손실과 평가에만 사용한다.

Cross/global/self 20,512개, concat 20,491개의 파라미터다. Frontend concat은 구조별 RNG 소비량 때문에 후속 공통 FFN 초기값도 다르며, 동일 파라미터 초기값의 주 대조는 candidate E0–E4 및 frontend self/cross/global이다. Audio matched−donor 및 matched−noaudio 결과가 일관된 양의 효과를 보이지 않았다. 따라서 feature 변화 자체를 유용한 representation 학습으로 부르지 않는다. 기존 B2 latent 평균 교환과 이 직접 QKV attention을 구분한다.

## 4. Scalar 전달과 후보별 구조 정보

| 방법 | Joint % | Candidate AUC % | GT mass % | GT rank | 오답 Top1 confidence % |
| --- | --- | --- | --- | --- | --- |
| pure_v | 46.38 | 64.81 | 17.30 | 3.98 | 9.75 |
| primary_E0_real | 48.08 | 67.40 | 36.68 | 3.78 | 33.93 |
| primary_E2_real | 55.97 | 69.01 | 42.63 | 3.30 | 40.82 |
| primary_E4_real | 56.39 | 68.62 | 44.14 | 3.27 | 41.46 |
| primary_global_real | 48.37 | 67.25 | 36.14 | 3.81 | 34.68 |
| scalar_real | 62.14 | 72.32 | 22.97 | 2.75 | 13.16 |

Scalar 대조는 pure visual/acoustic score를 동일 후보 내 표준화하고, outer-fit scene의 localization NLL로 혼합 가중치를 선택했다. 이 대조를 넘지 못하면 256D product/absolute-difference 특징에 새로운 관계 학습의 이득이 있다고 주장할 수 없다. Global 메시지는 query embedding만, candidate 메시지는 query–reference 관계를 전달한다. 비교는 이 정보 계약 안에서 후보별 관계의 가치를 검증하며 모든 global exchange의 불가능성이나 candidate conditioning의 보편적 필수성을 증명하지 않는다.

Query별 repair/harm, donor 변화, acoustic 순위 상관, scalar가 이미 해결한 repair와 scalar를 넘어선 소수 repair를 [증거 분석](artifacts.json) (`evidence_attribution.json`)과 [사례 배열](artifacts.json) (`disambiguation_cases.json`)에 저장했다. 사례는 scene/조건/유형별 query 순서 첫 2개로 고정했다. Feature effective rank는 별도 측정했으며 concat의 낮은 rank도 성능 collapse로 단정하지 않았다.

## 5. 강한 GEOQ와 후속 ranker 검증

| 방법 | Joint % | R@1m % | R@0.1m % | GEOQ 대비 Δpp [95%] | Repair % | Harm % |
| --- | --- | --- | --- | --- | --- | --- |
| GEOQ | 72.30 | 74.29 | 25.78 | +0.00 [+0.00, +0.00] | 0.00 | 0.00 |
| B2support | 70.88 | 73.37 | 25.85 | -1.42 [-2.41, -0.28] | 0.99 | 2.41 |
| base | 69.89 | 72.30 | 24.64 | -2.41 [-3.91, -0.85] | 1.35 | 3.76 |
| E0 | 69.67 | 72.16 | 25.21 | -2.63 [-4.26, -0.99] | 1.42 | 4.05 |
| E1 | 69.60 | 72.09 | 25.36 | -2.70 [-4.33, -0.99] | 1.42 | 4.12 |
| E2 | 70.38 | 72.80 | 24.72 | -1.92 [-3.48, -0.28] | 1.35 | 3.27 |
| E3 | 70.38 | 72.73 | 25.36 | -1.92 [-3.48, -0.21] | 1.85 | 3.76 |
| E4 | 70.03 | 72.44 | 24.72 | -2.27 [-4.19, -0.43] | 1.56 | 3.84 |
| global | 70.03 | 72.30 | 25.28 | -2.27 [-3.84, -0.64] | 1.35 | 3.62 |

각 방법의 score는 같은 8개 추가 feature slot으로 전달했다. 기존 S26·geometry summary52와 candidate0 상대값을 포함해 모든 ranker는 172개 열, HistGB150×15, 동일 utility joint+0.5 fine, 동일 threshold protocol을 사용했다. Base는 추가 slot이 0이며 padding으로 유효 정보량까지 같다고 주장하지 않는다. 기존 full-data GEOQ를 동일 cohort에서 평가한 수치는 72.30%이며 원래 전체 Val11 71.818%와 구분한다.

초기 in-sample adapter score로 ranker를 학습하면 과신과 성능 저하가 컸다. 후속 실험에서는 ranker 학습용 score까지 3/4개 내부 scene fold로 cross-fit했다. 이 변경은 성능을 회복했지만 GEOQ 개선으로 이어지지 않았다. 기존 B2 teacher까지 모든 inner fold로 다시 학습한 완전 nested 실험은 아니다. [GEOQ 대비 repair/harm 사례](artifacts.json) (`crossfit_repair_harm_cases.json`).

## 6. 전체 query 후속 실험

| 방법 | Train12 OOF % | Val8 % | Val11 % |
| --- | --- | --- | --- |
| pure_v | 49.32 | 49.19 | 45.47 |
| E0_real_v | 52.77 | 54.53 | 50.20 |
| E2_real_v | 62.93 | 64.00 | 59.51 |
| E4_real_v | 62.27 | 63.26 | 59.60 |
| global_real_v | 52.44 | 54.12 | 49.60 |
| scalar | 66.89 | 63.90 | 61.53 |
| E5_f | 72.70 | 71.01 | 68.09 |
| GEOQ | 76.94 | 74.80 | 71.82 |

6,387 Train / 4,190 Val8 / 5,775 Val11 query 전체로 E0/A→V/concat/global을 다시 학습했다. 두 seed와 scene holdout을 유지했다. 64-query 실험보다 데이터와 optimizer update 수가 함께 증가했으므로 순수 데이터 규모 효과로 귀속하지 않는다. 위 수치는 query-weighted micro average이며 작은 cohort 표와 직접 차감하지 않는다. 이 표의 새 score는 분기 판별 결과이며, 다음 표에서 전체 query의 새 GEOQ 후속 ranker를 별도로 검증한다. 후보 생성의 개선은 주장하지 않는다.

## 6a. 전체 query · 내부 scene 교차검증 후속 ranker

| 방법 | Joint % | R@1m % | R@0.1m % | 높은 모호성 % | GEOQ 대비 Δpp [95%] | Repair % | Harm % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GEOQ | 71.82 | 73.19 | 23.45 | 60.02 | +0.000 [+0.000, +0.000] | 0.00 | 0.00 |
| base | 71.82 | 73.19 | 23.45 | 60.02 | +0.000 [+0.000, +0.000] | 0.00 | 0.00 |
| E0 | 72.26 | 73.58 | 23.82 | 60.47 | +0.442 [+0.167, +0.678] | 1.50 | 1.06 |
| E2 | 71.91 | 73.28 | 23.68 | 60.30 | +0.095 [-0.262, +0.443] | 1.36 | 1.26 |
| E4 | 71.84 | 73.23 | 23.65 | 59.82 | +0.017 [-0.296, +0.340] | 1.23 | 1.21 |
| scalar | 71.78 | 73.15 | 23.58 | 59.48 | -0.035 [-0.257, +0.166] | 0.83 | 0.87 |

모든 조건은 동일172열·150×15 tree·scene-balanced utility loss·threshold protocol이다. Inner-crossfit model은 외부 held scene을 학습하지 않는다. Base는 기존 GEOQ의 전체 Train/Val8/Val11 수치를 정확히 재현했다. E0의 Val11 seed42/43 개선은 각각 +0.139/+0.745%p이다. E0는 두 branch를 독립적으로 보정한 점수를 GEOQ에 추가하며 상대 modality와 feature를 교환하지 않는다. Raw scalar 추가만으로는 같은 개선을 재현하지 못했다.

E0의 개발셋 개선과 Train OOF의 불확실성을 함께 기록한다. 반복 개발셋·여러 비교 이후의 관찰이므로 독립적인 확증 결과가 아니다. [GT-free experimental selector](artifacts.json) (`inference.py`)와 [동결 후보 설정](artifacts.json) (`candidate_policy.json`)을 보존했으며 원래 기본 모델을 교체하지 않았다. 전체 Val8+Val11의 두 seed 19,930건에서 public inference가 저장된 최종 선택을 정확히 재현했다. 이 검증은 보존된 full-bank B2→Support20에 새 최종 선택만 적용한 것이며, 새 후보 생성이나 coverage 향상을 의미하지 않는다. [전체 query repair/harm 사례](artifacts.json) (`full_repair_harm_cases.json`).

## 6b. Architecture와 objective를 분리한 후속 반증

| Objective family | 구조 | Val11 cohort Joint % | 높은 모호성 % | 모호한 pair % |
| --- | --- | --- | --- | --- |
| primary | E0 | 48.08 | 38.65 | 60.84 |
| primary | E2 | 55.97 | 46.67 | 65.56 |
| primary | E4 | 56.39 | 45.80 | 64.76 |
| primary | global | 48.37 | 37.14 | 60.61 |
| original_objective | E0 | 47.02 | 34.83 | 59.60 |
| original_objective | E2 | 53.34 | 44.56 | 67.04 |
| original_objective | E4 | 57.24 | 46.97 | 66.33 |
| original_objective | global | 45.53 | 36.25 | 60.00 |
| frontend | self | 46.95 | 35.53 | 55.07 |
| frontend | cross | 46.31 | 35.45 | 55.15 |
| frontend | global | 46.73 | 36.35 | 54.73 |
| frontend | concat | 47.09 | 37.09 | 55.76 |
| frontend_original | self | 46.59 | 33.99 | 55.18 |
| frontend_original | cross | 45.17 | 35.30 | 55.01 |
| frontend_original | global | 46.80 | 35.90 | 54.82 |
| frontend_original | concat | 46.52 | 36.73 | 55.20 |

초기 새 모델은 coarse mass와 추가 fused loss를 사용했다. 기존 B2의 정확한 학습 목적과 같다고 주장하지 않는다. 별도80fit에서는 같은 구조·seed·데이터·40epochs를 유지하고 coarse mass + 0.1 Gaussian precision(0.25m/10°) + 0.05 anchor KL, unrecoverable query KL, V/A branch 평균을 복원했다. Gaussian은 고정20의 유효 후보 위에서 정규화했으므로 원래 full-bank 학습과도 구분한다.

이 목적 변경은 명시적 A→V의 concat/scalar 대비 우위나 실제 visual-feature cross-attention의 올바른-audio 특이적 효과를 회복하지 못했다. KL과 precision 각각의 독립 효과까지 분해한 실험은 아니다. [원 목적 감사](artifacts.json) (`objective_audit.json`) · [architecture×objective 비교](artifacts.json) (`objective_analysis.json`).

## 7. Calibration과 비용

Inner scene-crossfit Top1 correctness로 temperature를 선택하고 평가 순위가 정확히 불변임을 검사했다. 평균 Brier 0.2213 → 0.2213. 이는 고정 후보 내 Top1 joint-success confidence 평가이며 calibrated global pose density는 아니다. Calibration과 ranking 개선을 혼동하지 않는다.

| 모듈 | 조건 | 파라미터 | batch64 forward ms |
| --- | --- | --- | --- |
| hypothesis | E0 | 30466 | 0.173 |
| hypothesis | E1 | 30466 | 0.173 |
| hypothesis | E2 | 30466 | 0.180 |
| hypothesis | E3 | 30466 | 0.173 |
| hypothesis | E4 | 30466 | 0.169 |
| hypothesis | global | 30466 | 0.174 |
| hypothesis | donor_train | 30466 | 0.173 |
| hypothesis | noaudio_train | 30466 | 0.171 |
| frontend | self | 20512 | 0.682 |
| frontend | cross | 20512 | 0.203 |
| frontend | global | 20512 | 0.205 |
| frontend | concat | 20491 | 0.149 |

CUDA event 측정은 캐시된 새 모듈 forward만 포함하며 encoder·FP matching·I/O·end-to-end latency는 제외한다. 동시 local 작업의 영향을 받을 수 있다. 각 fit의 실제 학습 시간·step·loss·checkpoint를 저장했다. 전체 wall time에는 data materialization과 검증을 포함한다.

## 8. 정보 계약과 평가 정직성

| 계약 | 정의 |
| --- | --- |
| intermediate | "Frozen40x128 UnLoc features; visual head maps each feature to depth/scale. Original40 predicted depth andscale interpolated to11 angular rays." |
| hypothesis_input | "44 values: FP11rays, observed11rays, scale11, signed normalized residual11; plus mean visual128. FP geometry used only existing hypothesis matching, never supplied to new frontend adapter." |
| intermediate | "FrozenE32x96,C4x64,N16x64 padded96 plus3stream identities+1ordered coordinate =52x100. These are learned joint-channel CNN tokens, not separate physical microphone or calibrated reflection-time tokens. Native265token alternatives already audited historically." |
| hypothesis_input | "Pure observed128, candidateFP128.256D product and absolute-difference evidence. Global message has observed128 only. No learned physicalwall correspondence assumed." |
| prior_probes | "Pairwise acoustic evidence T/N/C analyses and alignment studies reused as evidence; no fresh acoustic encoder or negative sampling training." |

6개 mic는 E/C/N encoder 내에서 이미 혼합된다. 52개의 CNN token을 개별 mic나 특정 벽 반사 token으로 해석하지 않는다. 원본 RIR peak–벽 대응은 가정하지 않았다. 평가 label은 기존 GEOQ와 같은 candidate grid pose 기준이며, 새 선택의 refinement 이득을 따로 주장하지 않는다. 원래 encoder·GEOQ·B2 checkpoint를 수정하지 않았고 새로운 acoustic encoder/negative mining은 수행하지 않았다.

Train12의 새 모듈은 scene-held-out이지만 visual teacher는 더 넓은 Train116에서 학습됐고 historical acoustic encoder도 Train12를 보았다. Acoustic Val3 checkpoint 선택, visual Val19 monitoring 노출이 있다. Val8/Val11은 반복 사용한 development이며 독립 일반화 증거가 아니다. Test는 사용하지 않았다. 현재 알려진 추가 Train scene도 upstream teacher 미노출 scene으로 간주할 수 없어 독립 일반화 주장은 미검증으로 남긴다. 여러 후속 실험의 선택 편향이 있으며 scene bootstrap 구간은 다중검정 보정 없는 기술적 구간이다.

## 9. 결론과 재개 조건

이 설정에서 후보별 acoustic matching evidence는 visual ambiguity를 줄인다. 하지만 그 효용은 복잡한 교환 구조의 고유한 이득으로 이어지지 않았으며, global visual-feature 보정은 올바른 acoustic observation에 특이적인 개선을 보이지 않았다. 개발셋에서 더 유망했던 것은 교환 없는 독립 branch 보정과 강한 후속 선택의 결합이었다. 다만 Train OOF와 독립 평가에서 재검증해야 한다. 새 구조의 핵심 검증 단위는 “특징이 바뀌었는가”보다 “올바른 후보와의 관계를 보존하며 concat/scalar와 강한 최종 기준을 넘는가”여야 한다.

재개한다면 현재 실패한 token 수 확대를 반복하기보다, 조건부 관계 학습이 scalar를 넘는 held-scene 증거 또는 teacher 미노출 새 scene을 먼저 확보한다. 새 데이터는 프로토콜·모델을 동결한 뒤 평가해야 한다. 이번 결과만으로 보편적인 새 학습 원리나 시스템 개선을 확정하지 않는다.

[재현 절차](artifacts.json) (`REPRODUCE.sh`) · [전체 audit](artifacts.json) (`information_audit.json`) · [요구사항 검사](artifacts.json) (`requirement_audit.json`) · [최종 검증](evidence/final_verification.json) · [기존 파일 보존](artifacts.json) (`preservation_verification.json`) · [전체 query 결과](artifacts.json) (`full_summary.json`) · [원 요청](artifacts.json) (`research_request.txt`)

> 역사적 연구 기록. 원본: `residual_spatial_20261108_1144/RESEARCH_REPORT.html`. 원본 SHA256은 [산출물 목록](artifacts.json)에 있습니다.
> 당시의 완료·재개 지시는 현재 실행 요청이 아닙니다. 최신 판정은 [현재 결론](../../../current_findings.md)을 먼저 확인하세요.

# AV-FPLoc: 잔여 cross-modal 정보와 공통 공간 표현

재현 가능한 개선을 찾기 위한 통제 연구 · 2026-11-08 · Test 미사용

## 판정과 연구 범위

**이번 범위에서 GEOQ를 넘어서는 재현 가능한 추가 판별 정보는 확립하지 못했다.** 이는 모든 잔여 정보의 부재를 증명하지 않는다. 작은 개선 신호를 전체 쿼리로 확장하고, 표현 손실·모델 용량·학습량·후보 참조 특징을 순차적으로 검증했다.

감사·실험·후속 검증을 포함한 실제 학습 1032회(내부 교차검증 포함), PCA 분해 20회. 이들은 독립 표본 1032개가 아니다. 통계 단위는 Train 12개 장면과 두 시드다. Val8/Val11은 반복 사용한 개발 데이터다. Test는 사용하지 않았다.

전체 쿼리: Train 6,387 / Val8 4,190 / Val11 5,775. 작은 실험은 장면당 128개, 마지막 FP 중간 특징 pilot은 64개다. 모든 비교는 동일 B2 Support Top-20을 유지했다.

그림: Train OOF changes — 외부 산출물 `research_summary.png` ([목록](artifacts.json)).

[전체 지표·장면 bootstrap 결과](artifacts.json) (`all_results.json`) · [마지막 확장 게이트](artifacts.json) (`reference_gate_outcome.json`)

## 1. V/A 정보 계약

시각 토큰은 40×128, 음향 토큰은 52×100이다. 음향 내용은 E 32×96, C 4×64, N 16×64를 96차원으로 맞춘 뒤 stream/순서 metadata를 붙인다. CNN 토큰의 순서는 방위각이나 벽 반사 위치가 아니다. 시각 ray는 −50°…+50°의 11개 방향이며, 나머지 25개 각도에는 관측값을 만들지 않았다.

GEOQ는 S26 + 후보 ray 요약26 + B2 관측 ray 요약26을 사용한다. R1/R2에는 관측 중간 특징, pure query–FP 임베딩 관계, 후보별 ray 잔차를 추가했다. FP 중간 토큰 비교는 별도 추출한 동결 CNN의 통계/투영 특징을 사용한다.

[information_contract.json](artifacts.json) (`information_contract.json`) · [prior_evidence_audit.json](artifacts.json) (`prior_evidence_audit.json`) · [source_verification.json](artifacts.json) (`source_verification.json`)

## 2. GEOQ 이후 잔여 정보

R0 기본 입력, R1 시각, R2 음향, R3 양쪽, R4 후보 상대 관계, R5 추가 정보 없는 기저 변환을 비교했다. 선형 Ridge와 동일 150×15 tree 예산을 분리했다. 128개 pilot의 R3 tree +0.228pp 신호는 전체 쿼리에서 −0.321pp로 바뀌었다. R3raw는 추가 32차원 투영을 제거한 648차원 후보 블록이며, 원래 VT/AT 요약까지 제거한 raw 전체 토큰은 아니다. 이 대조도 개선하지 못했다. 별도 대조에서는 실제 40×128 / 52×96 전체 토큰을 fit-scene PCA로 44/48차원에 투영했다(표준화 후 분산 약61%/85% 보존). 압축 방식 변경으로도 강한 기준을 넘지 못했지만, 모든 압축 손실 설명을 배제한 것은 아니다. 선형 모델의 후보 공통 query 항은 순위 차이에 직접 기여할 수 없으므로, 선형 음성 결과만으로 정보 부재를 판단하지 않았다.

| 실험 / 모델 | 쿼리/seed | Joint % | GEOQ 대비 Δpp [95% CI] |
| --- | --- | --- | --- |
| phase_a_full/tree_R0 | 6387 | 76.938 | +0.000 [+0.000, +0.000] |
| phase_a_full/tree_R1 | 6387 | 76.711 | -0.227 [-0.445, -0.016] |
| phase_a_full/tree_R2 | 6387 | 76.836 | -0.102 [-0.277, +0.075] |
| phase_a_full/tree_R3 | 6387 | 76.617 | -0.321 [-0.606, -0.041] |
| phase_a_full/tree_R4 | 6387 | 76.757 | -0.180 [-0.409, +0.017] |
| phase_a_full/tree_R5 | 6387 | 76.789 | -0.149 [-0.397, +0.087] |
| phase_a_raw/tree_R3raw | 6387 | 75.889 | -1.049 [-1.562, -0.541] |
| phase_a_raw/tree_R5raw | 6387 | 77.000 | +0.063 [-0.412, +0.527] |

[phase_a_full_analysis.json](evidence/phase_a_full_analysis.json) · [GEOQ 32,704개 query-seed 선택 완전 재현](artifacts.json) (`GEOQ_replay_verification.json`)

## 3. 공간 표현의 물리·수학적 타당성

반지름 5cm의 수평 6-mic 배열과 실제 renderer의 x/z 부호, 후보 yaw/channel roll을 사용했다. 시각 source bearing과 음향 propagation direction의 180° 차이를 포함한 36 yaw ×11 ray 검증 396건이 통과했다(최대 오차 8.82×10⁻¹⁵). 이는 합성 평면파 좌표 검증이며 실제 잔향 DoA 정확도의 증거가 아니다.

3개 주파수 band로 요약한 coherence의 36각도 steering 변환은 rank17이다. 실제 입력 에너지의 평균 5.78%를 버리므로 nullspace 보존 대조군을 추가했다. 합성 응답의 half-height 폭은 90°였다. 10° grid를 10° 방향 분해능으로 해석할 수 없다. 평면 배열의 자유장 elevation 부호 모호성이 있어 SH는 보류했다. 이것이 장면 prior를 통한 모든 높이 정보의 부재를 뜻하지는 않는다.

[common_frame_validation.json](artifacts.json) (`common_frame_validation.json`) · [spatial_validity.json](artifacts.json) (`spatial_validity.json`) · [spatial_bandwidth_audit.json](artifacts.json) (`spatial_bandwidth_audit.json`)

## 4. Angular/Fourier 및 독립 대안

원본 coherence, angular response, angular+nullspace, 완전/절단 sensor Fourier, 후보 상대 관계를 비교했다. 완전 Fourier는 정보 보존적이지만 sensor-index 기저이며 DoA 그 자체는 아니다. 올바른 공통 방향 ray–coherence 곱과 90° 어긋난 대조도 평가했다. 수학적 정렬의 성공은 판별 성능 향상으로 이어지지 않았다.

| 실험 / 모델 | 쿼리/seed | Joint % | GEOQ 대비 Δpp [95% CI] |
| --- | --- | --- | --- |
| spatial_full/tree_S0_raw | 6387 | 76.476 | -0.462 [-0.773, -0.161] |
| spatial_full/tree_S1_beam | 6387 | 76.624 | -0.313 [-0.526, -0.113] |
| spatial_full/tree_S1_preserved | 6387 | 76.476 | -0.462 [-0.742, -0.207] |
| spatial_full/tree_S2_fourier | 6387 | 76.311 | -0.626 [-0.988, -0.267] |
| common_spatial_pilot/tree_aligned | 1536 | 75.911 | -1.562 [-2.344, -0.814] |

그림: Information loss and optimization — 외부 산출물 `mechanism_summary.png` ([목록](artifacts.json)).

[spatial_full_analysis.json](artifacts.json) (`spatial_full_analysis.json`) · [common_spatial_pilot_analysis.json](artifacts.json) (`common_spatial_pilot_analysis.json`)

## 5. Concat / Modulation / Interaction

C1 concat, C2 후보 조건부 product/sum, C3 A→V modulation, C4 lossless angular-visible/nullspace 기저, C5 후보 간 acoustic message, C6 additive 독립 대안을 비교했다. C4 기저 변환과 실제 공통 ray 정렬 실험은 구분한다. 신경망은 모두 33,057 nominal parameter, 같은 cohort 내 동일 optimizer/50epoch를 사용했다. 전체 쿼리 확장은 epoch 수를 유지해 optimizer step도 늘었으므로 순수한 데이터 양만의 인과 비교가 아니다. C0는 V/A 경로를 0으로 하므로 유효 용량이 작다. 이를 보완해 GEOQ만 변환하면서 모든 경로를 사용하는 동일 파라미터 수 대조군을 따로 학습했다. 실제 한 batch에서 C1과 정보 없는 기저 대조군 모두 32,993개 parameter에 nonzero gradient가 있었고 C0는 4,641개였다. 이는 batch 의존 검사다. C6는 head를 세 번 사용해 연산량까지 같지는 않다.

C3는 내부 V를 조절하지만 최종 출력에 직접 A 경로도 있으므로 pure visual head의 인과 효과로 부르지 않는다.

| 실험 / 모델 | 쿼리/seed | Joint % | GEOQ 대비 Δpp [95% CI] |
| --- | --- | --- | --- |
| neural_pilot/C0 | 1536 | 74.219 | -3.255 [-4.884, -1.790] |
| neural_pilot/C1 | 1536 | 68.457 | -9.017 [-12.044, -6.673] |
| neural_pilot/C2 | 1536 | 65.137 | -12.337 [-14.486, -10.645] |
| neural_pilot/C3 | 1536 | 66.406 | -11.068 [-13.965, -8.822] |
| neural_pilot/C4 | 1536 | 67.904 | -9.570 [-11.882, -7.487] |
| neural_pilot/C5 | 1536 | 64.746 | -12.728 [-14.714, -11.263] |
| neural_pilot/C6 | 1536 | 68.457 | -9.017 [-11.426, -6.836] |

| 실험 / 모델 | 쿼리/seed | Joint % | GEOQ 대비 Δpp [95% CI] |
| --- | --- | --- | --- |
| neural_full/C1 | 6387 | 70.424 | -6.513 [-7.868, -5.228] |
| neural_full/C3 | 6387 | 70.158 | -6.779 [-7.828, -5.850] |
| neural_full/C4 | 6387 | 70.456 | -6.482 [-7.808, -5.237] |
| neural_capacity_full/C1 | 6387 | 74.245 | -2.693 [-3.493, -1.996] |
| early_stopping/C1 | 6387 | 72.742 | -4.196 [-5.191, -3.262] |
| token_probe/C1 | 1536 | 67.285 | -10.189 [-12.240, -8.301] |
| token_probe/C3 | 1536 | 65.137 | -12.337 [-14.193, -10.970] |
| token_probe/C4 | 1536 | 68.197 | -9.277 [-11.328, -7.519] |

추가 특징의 학습 MSE는 낮지만 held-scene MSE가 높았다. 내부 장면 검증으로 epoch를 선택해도 GEOQ를 넘지 못했다. 내부 scaler/epoch 선택은 외부 장면 label을 사용하지 않는다. 단 upstream B2를 내부 fold마다 재학습하지 않은 노출 한계는 남는다.

[optimization_generalization_audit.json](artifacts.json) (`optimization_generalization_audit.json`) · [early_stopping_analysis.json](artifacts.json) (`early_stopping_analysis.json`)

## 6. Frozen Visual과 실제 Visual Adapter

이번 신규 실험은 시각 encoder를 동결했다. 이전 연구의 실제 40×128 visual residual adapter 및 동결 depth head 실험을 해시 검증해 재검토했다. self/no-audio, concat, cross-attention, global, donor 및 두 objective 대조에서 audio-specific 우위를 확립하지 못했다. 이 결과는 이번에 새로 학습한 결과로 세지 않았다.

현재 frozen/shared 표현에서 안정적인 추가 정보가 확인되지 않아 D2 전체/부분 unfreeze 및 D3 공동 encoder 학습은 확대하지 않았다. 동결 실패가 fine-tuning 불가능성의 증명은 아니다. 알려진 실패를 더 큰 모델로 반복하는 것보다 정보·일반화 병목을 분리하는 판단이다.

[실제 D1 비교와 원본 검증](artifacts.json) (`prior_evidence_audit.json`)

| 이전 실제 visual 비교 (Val11,64query/scene) | Joint % |
| --- | --- |
| frontend_concat_real | 47.088 |
| frontend_cross_real | 46.307 |
| frontend_global_real | 46.733 |
| frontend_self_real | 46.946 |
| pure_v | 46.378 |

## 7. Acoustic-guided ambiguity와 인과 범위

GT-free ambiguity는 fit-scene pure visual entropy 상위 1/3 및 시각 score gap/후보 거리로 정의했다. GT는 oracle, good/bad pair, repair/harm 등 평가 label에만 사용했다. AUC, ambiguous pair accuracy, 조건부 Top-1, GT rank/mass, margin, high-ambiguity joint를 함께 기록했다. mass는 utility logit의 비보정 softmax 진단이며 calibrated posterior가 아니다.

기존 연구에서는 audio가 pure visual 후보 판별을 도왔고 pure-visual proposal 대조에서도 확인됐다. 그러나 이번 핵심 질문은 그 정보가 이미 강한 GEOQ에 들어간 뒤에도 추가 이득이 남는가이다. 이 둘을 혼동하지 않았다.

[전체 ambiguity/순위/보존 지표](artifacts.json) (`all_results.json`) · [장면·시드별 진단](artifacts.json) (`discrimination/`)

## 8. Matched / Donor / No-audio / Capacity

같은 장면의 다른 query 음향을 recipient FP와 비교했고, 새 음향 블록 제거 및 interaction disable을 평가했다. 동결 개입과 donor/no-audio로 재학습한 모델을 구분한다. 재학습 no-audio 모델의 정상 평가는 noaudio 조건이며 matched 입력은 OOD 개입이다. 기존 GEOQ의 acoustic score, B2-conditioned visual ray 및 proposal은 이 조건들에 남아 있다. 따라서 완전한 audio 제거 효과가 아닌 추가 블록의 조건부 효과다.

| 실험 / 모델 | 쿼리/seed | Joint % | GEOQ 대비 Δpp [95% CI] |
| --- | --- | --- | --- |
| neural_pilot/C1 | 1536 | 68.457 | -9.017 [-12.044, -6.673] |
| neural_retrain_donor/C1 | 1536 | 66.732 | -10.742 [-12.826, -8.952] |
| neural_retrain_noaudio/C1 | 1536 | 71.354 | -6.120 [-7.682, -4.688] |
| neural_queryonly/C1 | 1536 | 72.233 | -5.241 [-7.422, -3.255] |

[GT 독립성·저장 모델 replay·입력 대조 검증](artifacts.json) (`inference_verification.json`)

## 9. 최종 localization과 마지막 정보 공백

GEOQ Val11 joint는 71.818%, R@1m 73.195%, R@0.1m 23.446%, 고정 Top20 oracle 97.342%다. 역사적 B2 MAP 68.078, support/refine 70.078, utility70.675와 구분한다. 이전 E0+GEOQ72.260은 재사용 Val11 개발 결과이며 TrainOOF 개선이 없었다.

마지막으로 FP 최종 임베딩 대신 FP 중간 CNN 특징을 직접 짝지었다. 64개 pilot의 compact 통계 +0.391pp 신호를 전체 Train으로 확장했다. 사전에 저장한 확장 기준은 통과하지 못했다. 따라서 Val/Test 추가 추출과 신규 full-bank 후보 생성은 수행하지 않았다. 기존 full-bank → Support20 경로는 보존했고, 이번 변화는 고정 후보 순위만 평가했다. 선택 pose는 원래 grid 가설이며 신규 refinement 이득은 주장하지 않는다. R@0.1m는 yaw를 포함하지 않는 위치 기준이다.

| 실험 / 모델 | 쿼리/seed | Joint % | GEOQ 대비 Δpp [95% CI] |
| --- | --- | --- | --- |
| reference_probe_full/tree_R0 | 6387 | 76.938 | +0.000 [+0.000, +0.000] |
| reference_probe_full/tree_final | 6387 | 76.288 | -0.650 [-0.900, -0.393] |
| reference_probe_full/tree_compact | 6387 | 76.444 | -0.493 [-0.825, -0.185] |
| reference_probe_full/tree_capacity | 6387 | 77.031 | +0.094 [-0.382, +0.556] |

[reference_full_gate.json](artifacts.json) (`reference_full_gate.json`) · [reference_gate_outcome.json](artifacts.json) (`reference_gate_outcome.json`) · [연산·메모리 비용](artifacts.json) (`benchmark.json`)

## Val11 개발 결과도 함께 보고

추가 정보가 없는 변환 대조군의 작은 양의 점추정도 숨기지 않는다. 가장 높은 신규 전체 Val11 결과는 spatial capacity control71.983%이며, 새로운 cross-modal 정보의 이득이 아니다. 재사용 개발 결과이고 Train 개선을 확립하지 못했다.

| 실험 / 모델 | 쿼리/seed | Joint % | GEOQ 대비 Δpp [95% CI] |
| --- | --- | --- | --- |
| phase_a_full/tree_R0 | 5775 | 71.818 | +0.000 [+0.000, +0.000] |
| phase_a_full/tree_R3 | 5775 | 71.645 | -0.173 [-0.460, +0.078] |
| phase_a_raw/tree_R5raw | 5775 | 71.948 | +0.130 [-0.118, +0.401] |
| spatial_full/tree_capacity | 5775 | 71.983 | +0.165 [-0.252, +0.555] |
| neural_full/C1 | 5775 | 67.195 | -4.623 [-5.572, -3.787] |
| early_stopping/C1 | 5775 | 69.887 | -1.931 [-2.841, -1.095] |

## 10. Repair / Harm 사례와 메커니즘

한두 개선 사례만으로 방법의 효과를 주장하지 않는다. 아래는 각 장면에서 처음 발견한 repair와 harm을 함께 보여준다. 방향 응답은 벽 반사 label이나 feature attribution이 아니다. 사례 JSON에는 donor/noaudio/disabled/misaligned에서 선택이 바뀌는지도 남겼다.

그림: Repair and harm profiles — 외부 산출물 `repair_harm_examples.png` ([목록](artifacts.json)).

[repair_harm_cases.json](artifacts.json) (`repair_harm_cases.json`)

## 11. 지지·반증·미해결 가설

| 주장 | 판정 |
| --- | --- |
| A: audio가 visual 판별을 안내 | 이전 pure-visual 대조에서 지지. 이번 조건부 추가 이득과 별개. |
| B: 공통 공간 표현이 유용 | 좌표/회전 타당성 지지. 동일 정보 대비 안정적 localization 이득은 미확립. |
| C: interaction이 concat/fusion보다 우수 | 시험한 후보 조건부·modulation·relation 모델에서 미확립. |
| D: visual adaptation이 관계 학습을 개선 | 기존 D1 통제에서 미확립. 더 큰 fine-tuning 가능성은 열려 있음. |
| E: 강한 GEOQ 최종 성능 개선 | 이번 연구에서 확립하지 못함. 기본 시스템을 교체하지 않음. |

이 결론은 사용한 표현·목표·데이터·모델 범위에 한정한다. 비선형 잔여 정보, 충분한 신규 장면, 더 충실한 물리 관측 및 다른 학습 목표의 가능성은 남는다.

## 12. Method Contribution 후보와 연구자가 배울 점

**이번 결과만으로 확정할 신규 성능 Method Contribution은 없다.** 가장 실용적인 기여는 좌표 타당성 → 정보 보존 → held-scene 판별 → 강한 downstream 기준을 분리한 검증 절차와 반례다.

- 새 발견: 합법적인 angular 변환도 실제 신호 차원을 버릴 수 있고, 가역적 기저도 성능 개선을 보장하지 않는다.

- 틀린 기대: 큰 중간 특징·공간 정렬·더 많은 학습만으로 GEOQ 이후의 잔여 판별력이 드러난다는 기대는 이번 통제에서 지지되지 않았다.

- 최소 정보: 이번 자료에서는 scalar acoustic evidence와 안정적인 후보 기하 요약을 넘는 필수 추가 정보를 확정하지 못했다.

- 실패 조건: 적은 장면 수, 높은 차원의 query/reference 특징, observed–FP domain 차이, 넓고 혼합된 acoustic 방향 응답에서 일반화가 제한됐다. 정확한 원인별 기여는 모두 식별된 것은 아니다.

- 전이 원리: 동일 후보·정보·용량 통제, 물리 좌표와 token index의 구분, matched retraining, strong baseline 조건부 평가가 필요하다.

후속 후보는 후보별 query–reference 관계를 장면 변화에 강하게 학습하는 접근이지만, 이번 결과는 이를 검증된 방법으로 추천할 근거가 아니다. 다음 성능 주장을 위해서는 upstream teacher까지 노출을 확인한 신규 장면과 동결된 검증 프로토콜이 우선이다.

## 평가 한계·재현·보존

Train12 outer scene split은 새 head와 해당 outer B2 학습을 분리한다. Visual teacher Train116/Val19, acoustic historical Val3 선택 노출이 남아 완전한 pipeline 독립 평가가 아니다. 0/.01 switching threshold는 과거 Train OOF에서 선택된 값을 이번에 고정한 것으로, 새로운 완전 nested threshold 검증이 아니다. 전체 쿼리 확장은 같은 12개 장면 안의 표본 확장이며 독립 장면 확인이 아니다. Val8/Val11 재사용, 다중 탐색 및 12개 장면의 작은 통계 표본을 고려해야 한다. CI는 두 시드를 장면 내 평균한 뒤 scene-cluster bootstrap 10,000회이며 다중 비교 보정은 하지 않았다.

이전 연구 5,627개 파일과 사용자 git patch를 보존 검증했다. 124개 직접 입력과 134개 구조 특징의 원본 receipt도 검증했다. 모든 실패, 초기 오류 수정 기록 및 모델을 남겼다. 저장된 추론 replay는 대표 8개 신경망을 검사했고 전체 신경망의 배포 인증은 아니다. 전체 GEOQ replay는 32,704 query-seed 선택을 확인했다.

[REPRODUCE.sh](artifacts.json) (`REPRODUCE.sh`) · [requirement_audit.json](artifacts.json) (`requirement_audit.json`) · [stopping_decision.json](evidence/stopping_decision.json) · [source_verification.json](artifacts.json) (`source_verification.json`) · [receipt_verification.json](artifacts.json) (`receipt_verification.json`) · [feature_receipt_verification.json](artifacts.json) (`feature_receipt_verification.json`) · [artifact_manifest.json](artifacts.json) (`artifact_manifest.json`)

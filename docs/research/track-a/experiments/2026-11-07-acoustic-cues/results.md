> 역사적 연구 기록. 원본: `acoustic_cues_20261107_1830/RESEARCH_REPORT.md`. 원본 SHA256은 [산출물 목록](artifacts.json)에 있습니다.
> 당시의 완료·재개 지시는 현재 실행 요청이 아닙니다. 최신 판정은 [현재 결론](../../../current_findings.md)을 먼저 확인하세요.

# 무엇이 시각적 모호성을 푸는가: AV-FPLoc 음향 구조 증거

이 연구가 지지하는 설명은 **정규화된 시간·envelope 형태와 초기 구간의 복소 채널 관계가 서로 보완하는 위치 구분 증거를 제공한다**는 것이다. 직접 도달 시간이나 절대 음량으로 위치를 읽는 설명은 현재 입력·센서 구성과 맞지 않는다. 다만 특정 벽의 반사나 실제 DoA를 식별했다는 증거는 없으며, 음향 단서의 존재·encoder 보존·최종 판단의 추가 이득은 서로 다르다.

원래 B2/GEOQ는 보존했다. 다음 수치는 Train12 scene-held-out 또는 재사용 Val8/Val11 개발 결과다. Test와 신규 독립 scene은 사용하지 않았다. 표의 성공률은 %, 차이는 percentage point이다. 세부 수식·한계·실패 시도와 출처는 [METHODS.md](archive/METHODS.md)에 기록했다.

## 1. 실제 이용 가능한 정보

48 kHz의 6채널 RIR, E/C/N 캐시, encoder 중간 표현과 최종128D embedding, 후보별 FP reference를 확인했다. E는 시간-주파수 상대 에너지, C는 **초기와 전체 구간**의 복소 coherency, N은 채널 공통 성분을 제거한 정규화 envelope이다. C의 두 번째 구간을 후기 구간으로 간주하던 해석은 사용할 수 없다.

소스는 반경5 cm 마이크 배열의 중심에 고정되어 있다. 직접 경로는 대부분 sample7로 일정하다. E/C/N은 공통 양의 gain0.1/2/10에 거의 불변이었다. 따라서 검사한 정상 신호·gain 범위에서 절대 음량을 현재 정규화 입력으로 식별한다는 설명은 지지되지 않는다. 전처리는 carrier phase 전체를 보존하지 않지만 C에 요약된 상대 복소 정보는 남긴다.

31개 scene에서 **761,525개 raw record 파일·metadata의 존재**를 감사했다. 실제 raw subset 분석에 사용한 **49,047개 파형**은 파일 SHA256, metadata SHA256, 원래 feature-cache expected SHA256의 일치를 전부 확인했다. 전체 raw를 전수 재인코딩한 것은 아니다. [감사](artifacts.json) (외부 산출물: `${WORKSPACE}/experiments/acoustic_cues_20261107_1830/signal_audit.json`), [subset 원본 연결](artifacts.json) (외부 산출물: `${WORKSPACE}/experiments/acoustic_cues_20261107_1830/raw_subset_verification.json`), [전처리 재현](artifacts.json) (외부 산출물: `${WORKSPACE}/experiments/acoustic_cues_20261107_1830/physics_verification.json`).

## 2. 물리적으로 해석 가능한 단서와 측정 범위

T는 시간별 에너지 분포·중심·초기 비율, N은 envelope 형태·분리된 지배적 peak, C는 채널 쌍의 실수/허수·크기, F는 스펙트럼 형태, M은 마이크별 상대 에너지이다. Raw에서는 도달 시점·에너지 분위수·GCC-PHAT도 측정했다. Peak는 특정 반사로, GCC lag는 직접 DoA로, 에너지 분위수는 RT60으로 명명하지 않았다.

FP와 observation은 각각 단순화된 평면도와 furnished3D에서 얻어져 domain mismatch가 존재한다. 소스/배열 배치, 유한 시간창, 전처리 정규화와 renderer의 합성 응답이 해석을 제한한다. 단서의 세기만 크다고 정답 후보를 지지하는 것은 아니다.

## 3. 동일 선형 학습 조건의 cue 제거·복원

Query→FP descriptor Ridge(alpha10), Train-only scaling, scene OOF, 양쪽 동일 cue 제거, 같은 nominal456차원을 사용했다. 제거 arm은 활성 차원 수가 줄며, 원래 acoustic encoder를 재학습한 실험은 아니다. 기존 visual/acoustic embedding 점수는 입력하지 않았다. 따라서 아래는 **해석 가능한 cue만으로 학습 가능한 구분 능력**이다.

| Cue arm | Train Top20 | Val8 Top20 | Val11 Top20 |
|---|---|---|---|
| all | 61.09 | 66.54 | 69.32 |
| no_C | 40.30 | 46.40 | 47.58 |
| no_T | 56.54 | 59.31 | 65.35 |
| no_N | 53.34 | 59.07 | 64.68 |
| no_TN | 40.16 | 44.87 | 49.85 |
| T | 29.61 | 36.01 | 36.99 |
| N | 36.50 | 40.69 | 42.13 |
| F | 6.59 | 6.25 | 9.07 |
| C | 45.69 | 46.40 | 54.94 |
| no_TNC | 7.03 | 7.42 | 9.68 |
| C_early | 42.20 | 43.75 | 50.98 |
| C_full | 43.35 | 43.87 | 52.03 |
| C_late | 9.03 | 10.10 | 10.51 |
| C_real | 24.57 | 24.87 | 30.06 |
| C_imag | 24.49 | 25.08 | 32.95 |
| C_mag | 22.11 | 23.79 | 27.98 |

| 평가 | All Top1 / Top5 / Top20 | 고정20 Top1 | 고정20 pairwise AUC | Random NMS20 |
|---|---|---|---|---|
| Train12 OOF | 9.03 / 31.38 / 61.09 | 26.05 | 65.58 | 6.60 |
| Val8 개발 | 11.38 / 36.95 / 66.54 | 29.37 | 65.83 | 6.91 |
| Val11 개발 | 11.79 / 38.65 / 69.32 | 28.79 | 66.32 | 8.92 |

AUC는 정답과 오답이 함께 있는 후보 집합에 조건부다. Random은 같은 query·bank의 기존 NMS 대조군이다. 높은 Top20은 Top1 성공이나 최종 fusion 이득과 같지 않다. 세부 R1/joint/fine, GT mass, scene/seed 및 그룹별 값은 [metric_evidence.json](artifacts.json) (외부 산출물: `${WORKSPACE}/experiments/acoustic_cues_20261107_1830/metric_evidence.json`), 원시 scene record는 [cue_metric_analysis.json](artifacts.json) (외부 산출물: `${WORKSPACE}/experiments/acoustic_cues_20261107_1830/cue_metric_analysis.json`)에 있다.

| 대조: Top20 차이 | Train 95% scene CI | Val8 95% scene CI | Val11 95% scene CI |
|---|---|---|---|
| all_minus_no_C | +20.79 [+18.82, +22.98] | +20.14 [+18.56, +22.22] | +21.73 [+19.89, +23.56] |
| all_minus_no_TN | +20.93 [+10.26, +30.15] | +21.67 [+15.20, +27.32] | +19.46 [+14.88, +24.13] |
| joint_removal_loss_minus_sum_single_losses | +8.63 [+7.06, +10.13] | +6.97 [+2.47, +11.70] | +10.86 [+6.77, +15.16] |
| C_early_minus_late | +33.16 [+26.73, +40.12] | +33.65 [+26.49, +40.09] | +40.47 [+33.30, +47.99] |
| C_full_minus_late | +34.32 [+27.90, +41.32] | +33.77 [+27.23, +39.74] | +41.52 [+33.93, +49.41] |
| C_full_minus_early | +1.16 [-1.35, +3.26] | +0.12 [-0.75, +1.22] | +1.06 [-0.02, +2.23] |

구조 probe에서는 허수부가 실수부보다 강했지만, 실제 위치 검색에서는 허수−실수 Top20 차이가 Train−0.08pp, Val8+0.21pp, Val11+2.89pp였고 세 scene-bootstrap 구간 모두0을 포함했다. 따라서 “허수부가 위치 결정에 우월하다”는 초기 설명은 채택하지 않는다. 초기/전체/후기 시간창의 차이와 실수/허수 성분의 우열은 다른 가설이다. 후기 C가 약하다는 결과도 모든 후기 음향 정보의 부재를 뜻하지 않는다.

추가 반증 실험으로 각 Train fold의 첫 scene, 총256 query에서 query/reference 후기 C를 원본 STFT로 직접 계산했다. 같은 고정20·동결된 held-scene cue 학습기에서 캐시 기반 77/512, 원본 기반 75/512 성공이었고 선택 변경은 14/512였다. 이 범위에서 양자화 제거로 후기 성능이 회복되지 않았다. 전체 raw 재학습·full-bank 대조는 아니므로 양자화 영향을 완전히 배제하지는 않는다. [양자화 대조](artifacts.json) (외부 산출물: `${WORKSPACE}/experiments/acoustic_cues_20261107_1830/late_quantization_control.json`).

T/N 동시 제거 손실과 개별 손실의 합이 다르면 단서를 독립적으로 더해 기여도를 배분할 수 없다. C는 초기/전체 시간 요약을 포함하므로 no_TN에도 시간 정보가 일부 남는다. no_TNC는 F/M만 남겨 이 잔존 정보의 역할을 구분한다. 초기/전체/후기 C 비교는 모두48 활성 차원으로 맞췄다. 후기 C는 원래 STFT cross moment를 분해한 값이며 float16 양자화 오차를 별도로 측정했다. [대조](evidence/metric_comparisons.json), [late 검증](artifacts.json) (외부 산출물: `${WORKSPACE}/experiments/acoustic_cues_20261107_1830/late_coherence_verification.json`).

## 4. 현재 encoder·fusion·GEOQ의 실제 민감도

동결 모델의 입력을 변형한 결과는 **민감도**다. Query-only는 reference와의 분포 불일치가 섞인다. early_EN/late_EN은 C와 원래 정규화 척도를 유지하므로 완전한 초기/후기 신호 제거 실험이 아니다. 아래 전체 결과와 재학습 결과를 함께 해석해야 한다.

| Val11 query-only 변형 | Pure A Top1 / Top5 / Top20 | Fusion MAP Top1 | 재추출 support Top1 | 원래20 GEOQ | GEOQ 차이95% CI |
|---|---|---|---|---|---|
| baseline | 20.81 / 54.10 / 83.41 | 68.08 | 70.08 | 71.82 | +0.00 [+0.00, +0.00] |
| no_time | 0.31 / 1.39 / 5.37 | 25.56 | 25.84 | 60.94 | -10.88 [-12.51, -9.46] |
| no_C | 5.78 / 21.49 / 46.77 | 59.71 | 61.25 | 70.10 | -1.72 [-2.20, -1.31] |
| no_freq | 14.51 / 43.15 / 75.48 | 65.53 | 67.64 | 71.13 | -0.68 [-1.10, -0.26] |
| early_EN | 11.17 / 32.00 / 59.64 | 58.95 | 59.89 | 69.31 | -2.51 [-3.23, -1.81] |
| late_EN | 0.24 / 1.26 / 4.38 | 24.41 | 24.58 | 60.81 | -11.01 [-12.89, -9.37] |

| 고정20 acoustic argmax | Train Q / Ref / Both | Val8 Q / Ref / Both | Val11 Q / Ref / Both |
|---|---|---|---|
| baseline | 40.95 / 40.95 / 40.95 | 39.56 / 39.56 / 39.56 | 37.51 / 37.51 / 37.51 |
| no_time | 8.45 / 16.96 / 23.09 | 8.21 / 16.16 / 23.50 | 8.65 / 17.47 / 22.61 |
| no_C | 27.57 / 34.12 / 35.66 | 29.07 / 33.82 / 34.95 | 26.35 / 31.05 / 31.56 |
| no_freq | 33.86 / 35.29 / 34.66 | 35.19 / 36.11 / 36.59 | 32.97 / 34.46 / 34.89 |
| early_EN | 30.48 / 32.78 / 37.62 | 32.76 / 31.38 / 37.32 | 29.45 / 30.96 / 35.11 |
| late_EN | 7.59 / 16.77 / 26.48 | 7.17 / 16.85 / 27.57 | 7.54 / 17.30 / 26.72 |

Query와 reference를 함께 변형할 때 회복되는 부분은 일방적 교란의 큰 손실을 cue의 순수 인과 효과로 볼 수 없음을 보여준다. 양쪽 변형도 동결 모델의 OOD 조건이다. [screen 결과](artifacts.json) (외부 산출물: `${WORKSPACE}/experiments/acoustic_cues_20261107_1830/screen_evidence.json`)는 pure/exchanged, 고정 후보와 재추출 후보를 분리하며, [양쪽 변형](artifacts.json) (외부 산출물: `${WORKSPACE}/experiments/acoustic_cues_20261107_1830/paired_evidence.json`)은 고정20에서 acoustic argmax만 측정한다.

## 5. Visual ambiguity와 rescue 조건

기존 Train entropy threshold를 유지했다. 아래 그룹은 baseline으로 고정했으며, acoustic rescue와 실패 그룹은 GT를 쓰는 사후 분석이다. 모호성이 큰 query를 GT 없이 판단하는 지표와 구별해야 한다. 표는 Val11에서 cue-only 학습기의 Top20과 현재 동결 GEOQ의 민감도를 함께 보여준다.

| Val11 baseline branch | 전체 Top1 / Top20 | High ambiguity Top1 / Top20 |
|---|---|---|
| pure_v | 44.48 / 89.35 | 14.80 / 69.23 |
| pure_a | 20.81 / 83.41 | 21.38 / 84.13 |
| ex_v | 46.81 / 89.80 | 16.91 / 70.59 |
| ex_a | 25.52 / 87.52 | 26.14 / 87.66 |
| fusion | 68.08 / 97.56 | 44.70 / 95.11 |

Pure V는 음향 observation을 사용하지 않는 기준선이다. Exchanged V에는 음향 정보가 이미 유입되므로 no-audio 대조군이 아니다. 두 modality가 공통 구조를 관측하더라도 서로 다른 관측 오차·응답으로 같은 시각적 후보를 구별할 수 있다. 여기서 보인 rescue는 결정에 대한 상보성이며, 시각에 전혀 없는 새로운 물리적 구조 범주를 발견했다는 뜻은 아니다. GT로 정의한 rescue 그룹의 baseline 성공률은 정의상 높으므로 그것 자체를 추가 인과 증거로 쓰지 않는다.

| 고정 baseline 그룹 | 평균 query 수/seed | All cue20 | noC cue20 | noTN cue20 | GEOQ noC 변화 | GEOQ noTime 변화 |
|---|---|---|---|---|---|---|
| low_ambiguity | 2344.0 | 67.85 | 47.01 | 50.28 | -0.47 [-0.75, -0.20] | -4.01 [-4.75, -3.39] |
| medium_ambiguity | 1887.5 | 67.39 | 48.82 | 52.29 | -1.75 [-2.66, -1.08] | -14.12 [-16.04, -11.80] |
| high_ambiguity | 1543.5 | 73.89 | 46.94 | 46.23 | -3.60 [-5.13, -2.31] | -17.36 [-20.19, -14.24] |
| pure_visual20_fail_audio20_success | 467.0 | 71.31 | 41.97 | 48.39 | -3.64 [-6.07, -1.47] | -17.77 [-22.95, -12.39] |
| pure_visual1_fail_audio1_success | 580.0 | 79.14 | 58.28 | 49.31 | -3.97 [-6.58, -1.59] | -28.88 [-34.40, -24.31] |
| both_pure1_fail_fusion1_success | 1151.0 | 70.46 | 49.00 | 50.74 | -5.26 [-6.81, -3.78] | -25.50 [-28.39, -22.63] |
| fusion20_oracle_GEOQ_fail | 1487.5 | 67.63 | 40.97 | 55.16 | +5.85 [+4.70, +7.42] | +8.64 [+7.04, +10.38] |
| strong_wrong_acoustic | 1324.0 | 65.48 | 42.82 | 50.45 | -1.44 [-2.39, -0.54] | -9.63 [-12.13, -7.21] |

Fusion20 oracle/GEOQ 실패 그룹은 baseline GEOQ 성공률이 정의상0이므로, 그 안의 양의 변화는 일부 실패 복구만 뜻한다. 전체 query의 harm와 함께 보아야 하며, 해당 교란이 전반적으로 유리하다는 뜻은 아니다.

Val11에서 후기 coherency 크기는 전체 평균0.914로 초기0.594보다 높았다. 그러나 후기 C의 구조·검색 능력은 별도 실험에서 판단해야 하며, 높은 coherency 크기 자체는 위치 구분 정보량이 아니다. Rescue와 strong-wrong 그룹의 초기 에너지 비율도 약0.880/0.881로 비슷했다. 즉 단순한 신호 세기 기준만으로 음향의 정오를 판별한다는 설명은 지지되지 않는다.

Pure V20/A20 및 V1/A1 rescue와 exchanged 대응 그룹 모두 기계 판독 산출물에 포함했다. Fused 성공은 음향 Top1 성공에만 의존하지 않는다. 반대로 강한 음향 오답 그룹에서도 일부 단서는 유지되므로, 단순한 음향 confidence를 정답 확률로 취급할 수 없다. [그룹 신호 측정값](artifacts.json) (외부 산출물: `${WORKSPACE}/experiments/acoustic_cues_20261107_1830/group_signal_descriptives.json`)은 시간 중심·초기 비율·복소 크기 등을 제공하지만 scene 구성 차이가 섞인 상관관계다. Room type·가구·재료별 인과 효과를 검증한 실험은 아니다.

| Val11 FP grid 면적 구간 | Scene 수 | All cue20 | noC cue20 | noTN cue20 | Random20 |
|---|---|---|---|---|---|
| small | 4 | 75.45 | 51.66 | 64.82 | 13.56 |
| medium | 6 | 67.37 | 46.07 | 43.62 | 6.68 |
| large | 1 | 53.79 | 38.84 | 22.54 | 2.69 |

면적 구간은 Train FP bank의 유효 grid면적 삼분위119.88/164.52m²로 고정했다. Val11 큰 면적 구간은1scene뿐이므로 면적에 따른 일반적 인과 법칙을 주장하지 않는다. Bank가 커지면 random coverage도 낮아지므로 raw Top20만 비교하지 않았다. 확인 가능한 환경 조건은 이 면적/scene별 결과와 기존 visual ambiguity이며, corridor·가구·재료 효과는 별도로 검증해야 한다. [환경별 기록](artifacts.json) (외부 산출물: `${WORKSPACE}/experiments/acoustic_cues_20261107_1830/environment_evidence.json`).

## 6. Encoder 정보 보존과 추가 cue의 실효성

Scene-held-out reconstruction probe에서 final embedding의 Train-mean 대비 상대 MSE는 T0.208, C0.352, N0.445였다. 이는 해당 요약의 선형 복원 가능성을 지지한다. Intermediate576D는 T0.091/C0.258/N0.387이었지만 용량이 더 크다. 128D 고정 projection 결과도 함께 제공한다. F/M의 약한 probe 결과를 정보 부재로 해석하지 않는다. Raw direct timing은 거의 상수여서 relative MSE가 불안정하며 제외하고 해석해야 한다.

FP geometry probe에서 early/full/late C 상대 MSE는 각각0.556/0.562/0.923이었다. 초기 C와 구조의 연관성을 지지하지만, furnished observation의 특정 peak가 특정 벽에 대응한다는 증거는 아니다. [신호 probe](artifacts.json) (외부 산출물: `${WORKSPACE}/experiments/acoustic_cues_20261107_1830/cue_probes.json`), [구조 probe](artifacts.json) (외부 산출물: `${WORKSPACE}/experiments/acoustic_cues_20261107_1830/coherence_geometry_probe.json`), [raw 분산](artifacts.json) (외부 산출물: `${WORKSPACE}/experiments/acoustic_cues_20261107_1830/raw_descriptives.json`).

같은396차원·150×15 tree budget으로 GEOQ에 cue 비교 특징을 추가했으며, 그룹 제거·donor·score-basis 대조군을 사용했다. GEOQ_replay는 기존 모델의 결정을 재현했다.

| 추가 특징 arm | Train GEOQ 차이95% CI | Val8 차이95% CI | Val11 차이95% CI |
|---|---|---|---|
| CUES | -0.27 [-0.45, -0.09] | +0.05 [-0.04, +0.18] | -0.28 [-0.54, -0.01] |
| no_C | -0.23 [-0.45, +0.00] | +0.07 [-0.06, +0.18] | -0.23 [-0.54, +0.05] |
| no_T | -0.13 [-0.34, +0.11] | -0.25 [-0.43, -0.10] | -0.28 [-0.55, -0.05] |
| no_N | -0.27 [-0.55, -0.02] | -0.12 [-0.23, +0.00] | -0.13 [-0.44, +0.13] |
| no_TN | -0.22 [-0.47, +0.03] | +0.02 [-0.20, +0.22] | -0.18 [-0.49, +0.10] |
| basis | +0.06 [-0.41, +0.48] | -0.13 [-0.38, +0.13] | -0.21 [-0.52, +0.04] |
| donor | -0.25 [-0.43, -0.09] | +0.10 [-0.12, +0.29] | -0.07 [-0.20, +0.04] |

| 평가 | 기존 GEOQ | +CUES | Repair / Harm (query-seed) | Visual-success preservation |
|---|---|---|---|---|
| Train12 OOF | 76.94 | 76.67 | 98 / 132 | 93.83 |
| Val8 개발 | 74.80 | 74.84 | 62 / 58 | 93.70 |
| Val11 개발 | 71.82 | 71.54 | 73 / 105 | 93.83 |

이 추가 표현은 검증된 최종 개선이 아니므로 배포하거나 기존 GEOQ를 대체하지 않았다. 이미 보존된 정보와의 중복, 요약 손실, learner의 활용 한계가 경쟁 설명이다. 추가 이득이 없다고 모든 물리 정보가 충분히 활용되었다고 결론 내릴 수 없다. [조건부 추가 실험](artifacts.json) (외부 산출물: `${WORKSPACE}/experiments/acoustic_cues_20261107_1830/ranker_evidence.json`).

## 7. 정답/오답 후보의 신호 사례

성공 사례와 강한 오답 사례를 각각 Train/Val8/Val11에서 하나씩, 미리 정한 raw subset의 첫 해당 query로 선택했다. 정답/오답 후보는 각 GT 그룹에서 B2 acoustic score가 가장 높은 후보를 사용했다. 선택 후 cue-learner margin과 교란 결과를 계산했으며, 반대 방향의 증거도 제외하지 않았다. [수치 사례](artifacts.json) (외부 산출물: `${WORKSPACE}/experiments/acoustic_cues_20261107_1830/case_evidence.json`)는 정확한 query ID·pose·score margin을 담는다.

[Val11 acoustic rescue](artifacts.json) (외부 산출물: `${WORKSPACE}/experiments/acoustic_cues_20261107_1830/case_extra.png`)

[Val11 confident acoustic error](artifacts.json) (외부 산출물: `${WORKSPACE}/experiments/acoustic_cues_20261107_1830/case_failure_extra.png`)

원시 waveform, 여섯 채널 envelope, 초기 coherency를 함께 보면 후보 간 비슷한 응답과 차이가 공존한다. 눈에 띄는 한 peak의 일치만으로 정답을 설명할 수 없다. 그림은 물리적 신호 차이를 보여주며 wall-labelled explanation은 아니다.

## 8. 반증과 남은 경쟁 설명

- **지지:** 정규화된 시간 형태와 복소 채널 구조의 조합은 cue-only held-scene 검색에서 유용하다. 초기/전체 C가 후기 C보다 구조 및 검색 정보를 많이 제공하는지 동일48차원 대조로 검증했다.
- **지지:** 현재 embedding에 T/C/N 요약이 선형적으로 복원 가능하다. 이 사실과 decision-use는 별개다.
- **반증:** query-only 변형의 전체 손실을 cue의 필수성으로 해석하는 주장은 양쪽 변형의 회복 때문에 성립하지 않는다.
- **반증:** 공통 절대 음량 또는 source-to-ring direct time이 현재 모델의 주된 위치 단서라는 설명은 gain 불변성과 고정 배치에 맞지 않는다.
- **지지되지 않음:** 수작업 cue를 추가하면 GEOQ가 안정적으로 개선된다는 가설. 동일 학습 조건과 donor/basis 대조에서 추가 이득이 재현되지 않았다.
- **미해결:** 첫 반사·벽 대응·실제 DoA, real-world phase 일관성, 가구/재료별 인과 효과, 원래 encoder를 완전히 재학습했을 때의 cue 필요성, 독립 scene 일반화.

동일 pose 반복 렌더의 embedding cosine은 평균0.865, 다른 pose는0.067이었다. 초기 C의 반복 차이도0이 아니었다. 후기 waveform cosine은 같은 pose0.911/다른 pose0.910으로, 이 작은 실험에서는 높은 waveform 유사도가 위치 일치의 증거가 아니었다. 순서 반전의 차이는 새 stochastic render 변동과 분리되지 않았다. 2scene×2pose만으로 renderer 전체나 실제 음향의 타당성을 판단하지 않는다. [반복 분석](artifacts.json) (외부 산출물: `${WORKSPACE}/experiments/acoustic_cues_20261107_1830/repeat_analysis.json`), [순서 대조](artifacts.json) (외부 산출물: `${WORKSPACE}/experiments/acoustic_cues_20261107_1830/repeat_order_summary.json`).

## 9. 다음 개선의 우선순위

1. 초기 복소 관계와 정규화 시간/envelope를 함께 유지하되, 반복 렌더·query/reference domain mismatch에 안정적인 표현을 학습한다. 명시적 wall correspondence를 먼저 강제하지 않는다.
2. 현재 점수에 일반 요약을 더하는 것보다 **후보 쌍에서 어떤 신호 차이가 판별 가능한지** 학습하는 방향을 우선한다. 본 연구의 단순 cue 추가는 실패했으므로 동일 방식 확대는 근거가 약하다.
3. Reflection/DoA 기반 구조 제약을 도입하려면 먼저 동시 다채널 phase와 반복성, FP/Observed 모델 차이를 별도 검증한다. 물리적 이름만 붙인 learned cue는 근거가 되지 않는다.
4. Front-end 재학습은 보존할 cue와 제거할 nuisance를 위 결과로 정한 뒤, 동일 예산·여러 seed·독립 scene으로 평가한다. 이번 연구는 새로운 Transformer가 필요하다는 증거를 제공하지 않는다.

## 10. 일반화 가능한 연구 원리와 재현

Raw 정보의 존재, 표현의 decodability, 후보 coverage, 고정 후보의 discrimination, 최종 repair/harm를 각각 측정해야 한다. 한 지표의 개선을 다른 단계의 개선으로 대체할 수 없다. Query-only와 양쪽 변형, 동결 민감도와 matched learner retraining, 단일 제거와 결합 제거를 함께 비교해야 한다. Cue의 중복 때문에 ablation loss를 합산한 인과 배분은 타당하지 않다. Simulator 특성은 진단할 대상이며 곧바로 실제 물리 의미를 보장하지 않는다.

[재현 절차](archive/REPRODUCE.md), [방법과 실행 이력](archive/METHODS.md), [보존 검사](artifacts.json) (외부 산출물: `${WORKSPACE}/experiments/acoustic_cues_20261107_1830/preservation_verification.json`), [완료 검사](evidence/final_verification.json)를 함께 제공한다. 최종 보고서는 새 성능 개선을 주장하지 않으며, 검증된 정보 기여·반증·미해결 해석을 구분한다.

## 부록: 사전 선택한 사례의 수치 확인

| 사례 | Scene / query | B2 acoustic 정답−오답 점수 | All-cue 정답−오답 점수 | All-cue 원래20 Top1 정답 |
|---|---|---|---|---|
| rescue | 102815835 / 80 | +0.0607 | +0.0668 | False |
| rescue | 102344457 / 24 | +0.1527 | -0.2036 | False |
| rescue | 102344115 / 421 | +0.1370 | +0.0139 | False |
| pure acoustic failure | 102815835 / 191 | -0.1423 | -0.0631 | False |
| pure acoustic failure | 102344457 / 65 | +0.0635 | +0.0458 | True |
| pure acoustic failure | 102344115 / 93 | +0.7675 | +0.0607 | True |

한 정답/오답 쌍의 양의 margin이 전체20 후보 중 정답 선택을 보장하지 않는다. 실패 사례는 pure acoustic 기준이므로 B2 exchanged score가 해당 쌍에서 정답을 선호해도 모순이 아니다. 선택된 사례에서 맞는 cue가 존재하더라도 전체 평균의 추가 이득은 별도 검증해야 한다.

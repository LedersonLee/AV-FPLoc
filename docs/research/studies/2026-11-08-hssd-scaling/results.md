> 역사적 연구 기록. 원본: `hssd_scaling_20261108_1332/RESEARCH_REPORT.html`. 원본 SHA256은 [산출물 목록](artifacts.json)에 있습니다.
> 당시의 완료·재개 지시는 현재 실행 요청이 아닙니다. 최신 판정은 [현재 결론](../../current_findings.md)을 먼저 확인하세요.

AV-FPLOC · HSSD SCALING · 2026-11-08 · 검증 완료 · 독립성 제한 명시

# 장면 다양성, 정보 교환과 Geometry 기반 후보 판별

자산 감사 → 실제 입력 → 478 fits → 대조 평가 → 전체query 반증

## 판정과 검증 범위

**이번 실험에서 지지된 원리는 후보의 geometry에 맞게 기존 증거를 해석하는 것이다.** 초기 query-weight 학습에서는 데이터 확대 이득이 나타났지만, 기존의 장면별 동일 가중치까지 복원한 후속 실험에서는 신뢰구간이0을 포함했다. 데이터 확대 효과는 학습 방식에 민감했고, 총 query 고정 시 장면 수만 늘린 개선도 확인되지 않았다. A→V/concat/visual adapter의 추가 정보 전달 효과는 지지되지 않았다.

새 HSSD 데이터셋 전체에 대한 zero-shot이나 완전히 새로운 공간의 일반화 증거가 아니다. **11개 평가는 B2/GEOQ/이번 head 학습에서 제외했으나 visual teacher에는 노출된 장면**이다. 기존 Val19는 재사용 development이며 Val8/Val11을 새 평가로 부르지 않는다. 기존 Test 입력·성능은 사용하지 않았다. Test ID 및 family 제외를 위한 행정 메타데이터만 확인했다.

완료된 supervised fit은 총478회다. 1차250회(70 head +120 inner head +12 B1/B2 +12 visual adapter +36 inner adapter), 전체query 후속108회, OOF 입력 복원108회, 장면 가중치 복원12회를 수행했다. 시드는42/43이며 최종 checkpoint는178개다. 각 단계의 checkpoint를 해당 단계 평가 전에 고정했다. 추가 실험의 평가 장면은 이미 보았으므로 독립성을 주장하지 않는다. 모든 CI는 scene을 단위로 시드를 평균한 10,000회 paired bootstrap이며 다중 비교 보정 전 탐색적 구간이다.

[frozen_models_before_held.json](artifacts.json) (`frozen_models_before_held.json`) · [training_verification.json](artifacts.json) (`training_verification.json`)

## 1–5. 자산 감사, FP 품질, 재사용과 분리

설치된 scene config는 168개다. 승인 non-Test Train116+Val19의 135개에서 실제 RGB·6채널 RIR·FP·DESDF·E/C/N 입력 표본을 검사했다. 13,264개 object configuration/render asset 누락은 0개다. 135 stage mesh(408 MB)의 실제 해시에서 완전 중복은 없었다. 같은 원본 prefix family를 보수적으로 묶고 Test family와 겹치는 Train3개를 제외했다.

전체 dense candidate bank는 Train36+Val19의 55개이며, Test-family dense1개를 제외한 54개가 이번 실험에 사용된다. Train35를 기존12+새학습12+head-held11로 나눴다. Paired positive+7 mined negative 캐시는 135개에 있지만 이를 full-bank localization으로 대체하지 않았다. 48/96장면 학습은 준비된 dense bank 범위를 넘는다.

FP는 기존 구조·연결성 curation 승인과 새 binary/scale/DESDF/indoor-polygon screening을 결합했다. 135개 모두 screening을 통과했으나, 전체 문/벽/누락 방을 새 독립3D GT로 재검증한 것은 아니다. Furniture-free FP는 navigation map이 아니다. 음향 Observed는 furnished native mesh, FP reference는 canonical2Dmap extrusion이다. 실제 stage-only mesh와 구분한다. Source는 RGB 중심,6mic는5cm ring,48kHz이며 materials/transmission은 비활성이다. 135scene의270recordmetadata와71,467개의 yaw/roll 항목을 검증했다. 130scene의 높이는 semantic known-height 가정,5scene은 기존 승인된 실내 높이 평균이다. 따라서 불일치에는 가구와 수직구조 근사가 함께 들어간다. 외부영역 오검사는 기존 outdoor-label 정책으로 수정하고 초기 결과도 남겼다. 미승인16개는 repair/exclude/insufficient로 별도 기록했다.

실제 Habitat furnished/stage-only 렌더와 알려진 승인 query pose replay를 수행했다. 원본 RGB와 평균 절대 픽셀 차이는 0.839/255다. 새23개 scene의 전체 bank encoder 입력을 생성했고 실험54개×128query의 RGB/RIR/structural depth pose 및 sample ID 일치를 검증했다. 장면 규모·실내 region 수·repetition·bank size·entropy·가구밀도를 기록했다. Room/corridor 및 대칭성은 descriptor/semantic proxy이며 완전한 layout taxonomy가 아니다.

그림: 장면 분리와 query-budget 대조 — 외부 산출물 `dataset_scaling_design.png` ([목록](artifacts.json)).

장면 분리와 query-budget 대조 · [PDF](artifacts.json) (`dataset_scaling_design.pdf`)

그림: 동일 pose의 원본 furnished, structural stage와 FP — 외부 산출물 `asset_alignment_example.png` ([목록](artifacts.json)).

동일 pose의 원본 furnished, structural stage와 FP · [PDF](artifacts.json) (`asset_alignment_example.pdf`)

[HSSD_DATASET_AUDIT.md](archive/HSSD_DATASET_AUDIT.md) · [scene_availability.json](artifacts.json) (`scene_availability.json`) · [unready_source_scenes.json](artifacts.json) (`unready_source_scenes.json`) · [split_protocol.json](evidence/split_protocol.json) · [stage_identity_audit.json](artifacts.json) (`stage_identity_audit.json`) · [floorplan_region_checks.json](artifacts.json) (`floorplan_region_checks.json`) · [query_alignment_verification.json](artifacts.json) (`query_alignment_verification.json`)

## 6. 동결 baseline transfer

11 head-held ×128query=1,408개 고유 관측. 표는 두 시드를 scene 내부에서 평균한 scene-macro %. 순서는 R@1m, R@1m & 30°, R@0.1m이며 fine에는 yaw 조건을 넣지 않는다.

| Frozen method | R@1m | Joint | R@0.1m |
| --- | --- | --- | --- |
| pure_visual | 49.929 | 49.077 | 22.443 |
| pure_acoustic | 32.173 | 22.088 | 4.616 |
| fixed_fusion_015 | 68.324 | 66.335 | 26.562 |
| existing_B2 | 71.911 | 70.135 | 24.645 |
| support_nms_local_refinement | 74.467 | 72.408 | 24.929 |
| existing_GEOQ | 76.207 | 74.219 | 25.852 |
| existing_E0_GEOQ | 76.634 | 74.893 | 26.030 |

Pilot에서 fixed fusion은 pure Visual보다17.259%p 높았다. 음향 증거의 실용 가치는 기본 결합에서 이미 나타나며, 이번 interaction 대조는 그 위의 추가 이득을 측정한다. 기존 모델은 같은 HSSD의 과거 학습을 포함한다. 따라서 새 head-held 장면으로의 transfer이며 dataset zero-shot이 아니다. 기존 E0+GEOQ의 탐색적 +0.675%p(CI +0.178~+1.207)는 확정 기여로 간주하지 않고 아래 전체query 후속 검증을 촉발했다.

## 7. Scene/query scaling과 일반화 gap

Small12×128=1,536query, medium24×128=3,072query, 고정 총량24×64=1,536query. 128개는 source 순서의 균등 표본, 64개는 그 절반이다. 24×128−12×128은 scene와 관측 수가 함께 늘어난 비교이며, 24×64−12×128은 총량을 맞춘 비교다. 새로운 scene 구성의 반복 resampling은 수행하지 않았으므로 diversity 효과의 전반적 부재를 증명하지 않는다.

그림: GEOQ, interaction 및 동일 정보 capacity control의 학습/평가 곡선 — 외부 산출물 `scaling_curves.png` ([목록](artifacts.json)).

GEOQ, interaction 및 동일 정보 capacity control의 학습/평가 곡선 · [PDF](artifacts.json) (`scaling_curves.pdf`)

| Pilot paired contrast | Joint Δpp | 95% CI | Seed42 / 43 |
| --- | --- | --- | --- |
| tree_GEOQ_24x128 − tree_GEOQ_12x128 | +1.278 | +0.639 ~ +1.918 | +0.852 / +1.705 |
| tree_GEOQ_24x64 − tree_GEOQ_12x128 | -0.036 | -0.818 ~ +0.746 | -0.284 / +0.213 |
| neural_D1_24x128 − neural_D1_12x128 | +0.923 | -0.071 ~ +1.847 | +0.710 / +1.136 |
| neural_D2_24x128 − neural_D2_12x128 | +0.994 | +0.071 ~ +2.131 | +1.207 / +0.781 |
| tree_GEOQ_24x128 − tree_score_only_24x128 | +0.959 | +0.213 ~ +1.740 | +0.710 / +1.207 |

24×128에서 train/held joint는 GEOQ82.406/74.929%, D1 78.564/70.312%, D2 78.743/69.744%, capacity basis79.297/74.858%다. Neural interaction만 크게 회복되는 양상은 아니다. 추가 입력 없는 geometry basis도 강하므로, 추가 modality 입력이 곧 유용한 정보 획득은 아니다. Head fitting의 train 점수는 in-sample이고 기존12개는 upstream B2에도 in-sample이다. 과거 fullTrain12의 약6,387query보다 이번 pilot 학습 query 수가 적다는 차이를 유지한다.

## 8–9. Visual adaptation과 A→V 대조

D0 additive, D1 concat, D2 audio-gated visual, D3 candidate-relative gate는 동일 입력·정규화·목표·약29.6k parameters를 사용한다. D0는 동일 head를 세 번 호출하므로 계산량까지 같지는 않다. D2는 gate 이후 direct audio도 보존하므로 순수한 visual-only 전달로 해석하지 않는다. D1_noG는 추가 geometry를 S의 고정 basis로 대체하며 S에 남은 geometry/acoustic까지 제거하지 않는다. Capacity control은 모든 branch를 활성화한 채 추가 정보 없는 GEOQ basis를 사용한다.

Visual adapter는 실제40×128 visual feature residual을 frozen depth head로 전달한다. Self-adapter와 audio-adapter의 구조/parameter budget을 맞췄다. 구조 depth Laplace NLL과 residual penalty로 학습하고 3개 inner-scene folds에서 epoch를 골랐다. Full-bank localization objective에 의한 joint adaptation은 별도의 B1/B2 비교다. B1은 private/noexchange, B2는 shared exchange. 고정8epoch로 새로 학습했으며 기존 B2를 warm-start한 실험은 아니다.

| Pilot controlled contrast | Joint Δpp | 95% CI |
| --- | --- | --- |
| neural_D1_24x128 − neural_D0_24x128 | -0.533 | -1.065 ~ -0.036 |
| neural_D2_24x128 − neural_D1_24x128 | -0.568 | -1.349 ~ +0.249 |
| neural_D3_24x128 − neural_D1_24x128 | -0.675 | -1.278 ~ +0.036 |
| b2_B2_24x128_fusion − b2_B1_24x128_fusion | +0.213 | -0.355 ~ +0.817 |
| visual_audio_24x128_fusion − visual_self_24x128_fusion | +0.000 | -0.320 ~ +0.320 |
| visual_audio_24x128_visual − pure_visual | -1.207 | -2.095 ~ -0.249 |
| visual_audio_24x128_fusion − fixed_fusion_015 | -0.284 | -1.456 ~ +0.852 |

Visual depth MAE는 frozen0.182064m에서 self0.180241m, audio0.180224m로 작게 낮아졌지만 pure visual joint는 각각−1.243/−1.207%p였다. 평균 depth 품질 개선이 pose ranking 개선을 보장하지 않는다. Audio와 self의 fusion 차이는0.000%p였다.

D2 matched−donor +0.355%p(CI−1.207~+1.953), matched−disabled −1.101%p(CI−1.918~−0.071). Audio-adapter fusion matched−donor +0.142%p(CI−0.071~+0.355). 올바른 audio에서만 안정적으로 생기는 interaction 이득은 확인되지 않았다. No-audio와 donor는 새 branch만 조작한다. Visual adapter의 disabled 조건은 audio routing을 visual-self routing으로 바꾸며 residual adapter 전체를 제거하지 않는다. D2/D3의 disabled는 modulation gate만 끈다. 기존 B2 proposal/S와 visual-adapter fusion의 recipient pureA 증거는 남아 있다. Donor 학습 및 no-audio 학습 D2도 별도 포함해 test-time shift와 분리했다. Partial backbone fine-tuning/D4는 필요 시 조건이었으며, 현재 신호로 확장하지 않았다.

## 10–12. Geometry, ambiguity와 module-wise 평가

24×128 GEOQ−score-only +0.959%p(CI+0.213~+1.740). 동일 tree budget에서 geometry가 추가 구분력을 제공했다. Candidate-only, observation-only, observation-residual tree도 모두 보존했다. 전체query에서 candidate-geometry tree74.034%, GEOQ73.881%, observed-geometry72.412%, observation-residual72.678%, score-only72.456%였다. Candidate geometry만으로도 효과가 유지되므로 이를 새로운 관측 기반 물리 구조 일치의 발견으로 해석하지 않는다. 그러나 S 자체에도 기존 geometry/acoustic 요약이 남으므로 “geometry 완전 제거” 대조는 아니다. Geometry-conditioned score 해석의 효과를 새 acoustic structural consistency의 증거로 바꾸지 않는다.

그림: Raw field coverage와 Support20 후보 판별을 분리한 평가 — 외부 산출물 `module_evaluation.png` ([목록](artifacts.json)).

Raw field coverage와 Support20 후보 판별을 분리한 평가 · [PDF](artifacts.json) (`module_evaluation.pdf`)

그림: Training threshold로 고정한 high ambiguity 및 전체 query 비교 — 외부 산출물 `subgroup_comparison.png` ([목록](artifacts.json)).

Training threshold로 고정한 high ambiguity 및 전체 query 비교 · [PDF](artifacts.json) (`subgroup_comparison.pdf`)

GT-free thresholds는 train24의 visual entropy tertiles, bank-size/repetition/furniture-density 상위 1/3이다. Visual-failure/acoustic-rescue, both-fail, GEOQ-failure-with-coverage는 GT를 쓰는 진단 그룹이며 inference gate가 아니다. Repetition subgroup은 held2scene뿐이고 large-bank/high-density는4scene뿐이라 일반화 주장을 제한한다. 가구밀도와 furnished-depth warning은 mismatch proxy이며 acoustic mismatch의 직접 GT가 아니다.

Pilot의 GEOQ high-ambiguity joint75.94%, large-bank65.92%, both-modalities-fail50.89%. E0 추가 이득은 high-ambiguity +1.09%p와 both-fail +1.42%p에 나타났지만 acoustic-rescue에서는−0.71%p(CI가0포함)였다. 단순히 acoustic-only rescue를 학습한 결과로 단정할 수 없다. 큰 bank의 낮은 정확도는 난이도와 상관관계이며 bank크기의 인과 효과가 아니다. FP quality screening 실패 scene이 평가에 없으므로 FP 오류의 인과 기여를 이 결과만으로 추정하지 않는다.

Pilot Support20의 joint coverage는98.722%, GEOQ의 coverage-조건부 Top1은75.118%다. 이 범위의 주요 여지는 후보가 없는 문제보다 후보 판별에 있다. Support20 coverage, conditional Top1, GT best rank, pair-AUC, good/bad margin, repair/harm, visual preservation, raw GT mass를 보존했다. GT mass는 별도 calibration 없는 native score softmax다. Evidence mass/lift는 고정 T=.05이며 subgroup entropy의 T=1과 구분했다. Coverage AUC는 K=1/5/10/20 사다리꼴 적분/19, diversity는 후보 XY pairwise 거리다. GEOQ 실패 그룹에서의 repair만 떼어내지 않고 전체 harm과 함께 해석한다.

그림: 각 유형의 첫 두 qualifying scene·첫 query를 고른 설명용 사례. 성능 증거는 전체scene 통계. — 외부 산출물 `repair_harm_cases.png` ([목록](artifacts.json)).

각 유형의 첫 두 qualifying scene·첫 query를 고른 설명용 사례. 성능 증거는 전체scene 통계. · [PDF](artifacts.json) (`repair_harm_cases.pdf`)

[supplementary_results.json](artifacts.json) (`supplementary_results.json`) · [statistical_results.json](artifacts.json) (`statistical_results.json`) · [subgroup_protocol.json](artifacts.json) (`subgroup_protocol.json`)

## 13. 신호에 따른 전체query 후속 실험

동일11scene의 전체 **6,072개 query**로 동결 모델을 재평가했다. Pilot1,408query가 포함되므로 독립 replication이 아니다. 모델·threshold·학습은 변경하지 않았다. 70개 새 head도 전체query에 적용하여 query 표본에 따른 scaling 결론의 민감도를 검사했다.

E0+GEOQ는 사전 설정한 전체query 민감도 검증 기준을 통과하지 못했다. GEOQ 대비 joint +0.345%p, CI[-0.081 ~ +0.821], seed42/43 +0.167/+0.523%p. R@1m Δ+0.200%p, fine Δ+0.110%p. Repair 1.575%, harm 1.231%로 함께 보고한다.

| Frozen method | R@1m | Joint | Fine | Joint ΔGEOQ pp | 95% CI |
| --- | --- | --- | --- | --- | --- |
| pure_visual | 47.929 | 46.881 | 21.573 | -27.087 | -30.445 ~ -23.905 |
| pure_acoustic | 31.621 | 21.707 | 5.008 | -52.261 | -56.520 ~ -47.681 |
| fixed_fusion_015 | 68.227 | 66.250 | 26.883 | -7.719 | -9.181 ~ -6.253 |
| existing_B2 | 71.243 | 69.667 | 24.483 | -4.301 | -5.399 ~ -3.274 |
| support_nms_local_refinement | 73.427 | 71.781 | 24.988 | -2.187 | -3.457 ~ -1.321 |
| existing_GEOQ | 75.500 | 73.968 | 25.713 | +0.000 | +0.000 ~ +0.000 |
| existing_E0_GEOQ | 75.700 | 74.313 | 25.824 | +0.345 | -0.081 ~ +0.821 |
| tree_GEOQ_12x128 | 74.434 | 72.778 | 24.799 | -1.190 | -1.961 ~ -0.591 |
| tree_GEOQ_24x128 | 75.382 | 73.881 | 25.429 | -0.087 | -0.563 ~ +0.392 |
| tree_GEOQ_24x64 | 74.199 | 72.646 | 24.798 | -1.322 | -2.052 ~ -0.786 |
| neural_D1_24x128 | 72.379 | 70.494 | 24.631 | -3.474 | -4.756 ~ -2.061 |
| neural_D2_24x128 | 71.161 | 69.305 | 24.127 | -4.663 | -5.890 ~ -3.346 |
| neural_capacity_24x128 | 75.091 | 73.493 | 24.711 | -0.475 | -1.335 ~ +0.491 |

| Full-query contrast | Joint Δpp | 95% CI | Seed42 / 43 |
| --- | --- | --- | --- |
| existing_E0_GEOQ − existing_GEOQ | +0.345 | -0.081 ~ +0.821 | +0.167 / +0.523 |
| tree_GEOQ_24x128 − existing_GEOQ | -0.087 | -0.563 ~ +0.392 | -0.156 / -0.018 |
| tree_GEOQ_24x128 − existing_E0_GEOQ | -0.432 | -0.865 ~ -0.024 | -0.322 / -0.541 |
| tree_GEOQ_24x128 − tree_GEOQ_12x128 | +1.103 | +0.718 ~ +1.553 | +1.073 / +1.134 |
| tree_GEOQ_24x64 − tree_GEOQ_12x128 | -0.132 | -0.421 ~ +0.154 | -0.367 / +0.103 |
| tree_score_only_24x128 − tree_score_only_12x128 | +0.330 | +0.045 ~ +0.634 | +0.032 / +0.629 |
| tree_score_only_24x64 − tree_score_only_12x128 | -0.122 | -0.479 ~ +0.249 | -0.404 / +0.161 |
| neural_D0_24x128 − neural_D0_12x128 | +0.258 | -0.146 ~ +0.668 | +0.083 / +0.433 |
| neural_D0_24x64 − neural_D0_12x128 | -0.584 | -1.097 ~ -0.009 | -0.718 / -0.449 |
| neural_D1_24x128 − neural_D1_12x128 | +0.219 | -0.183 ~ +0.600 | +0.143 / +0.296 |
| neural_D1_24x64 − neural_D1_12x128 | -0.929 | -1.268 ~ -0.549 | -1.151 / -0.707 |
| neural_D2_24x128 − neural_D2_12x128 | -0.012 | -0.638 ~ +0.649 | +0.343 / -0.368 |
| neural_D2_24x64 − neural_D2_12x128 | -0.309 | -0.856 ~ +0.343 | -0.441 / -0.177 |
| neural_D3_24x128 − neural_D3_12x128 | +0.077 | -0.532 ~ +0.709 | +0.450 / -0.296 |
| neural_D3_24x64 − neural_D3_12x128 | -0.586 | -0.976 ~ -0.037 | -0.537 / -0.635 |
| neural_capacity_24x128 − neural_capacity_12x128 | +0.299 | -0.039 ~ +0.651 | +0.468 / +0.130 |
| neural_capacity_24x64 − neural_capacity_12x128 | -1.154 | -1.490 ~ -0.802 | +0.373 / -2.682 |
| tree_GEOQ_12x128 − tree_score_only_12x128 | +0.653 | +0.384 ~ +0.956 | +0.633 / +0.673 |
| neural_D2_12x128 − neural_D1_12x128 | -0.957 | -1.277 ~ -0.648 | -0.916 / -0.998 |
| neural_D3_12x128 − neural_D1_12x128 | -0.869 | -1.199 ~ -0.582 | -0.886 / -0.852 |
| neural_D1_12x128 − neural_D0_12x128 | +0.135 | +0.006 ~ +0.243 | +0.256 / +0.014 |
| tree_GEOQ_24x128 − tree_score_only_24x128 | +1.426 | +0.873 ~ +2.179 | +1.674 / +1.178 |
| neural_D2_24x128 − neural_D1_24x128 | -1.189 | -1.662 ~ -0.764 | -0.716 / -1.662 |
| neural_D3_24x128 − neural_D1_24x128 | -1.011 | -1.450 ~ -0.610 | -0.579 / -1.444 |
| neural_D1_24x128 − neural_D0_24x128 | +0.096 | -0.156 ~ +0.350 | +0.315 / -0.123 |
| tree_GEOQ_24x64 − tree_score_only_24x64 | +0.642 | +0.122 ~ +1.195 | +0.670 / +0.614 |
| neural_D2_24x64 − neural_D1_24x64 | -0.337 | -0.620 ~ -0.046 | -0.207 / -0.468 |
| neural_D3_24x64 − neural_D1_24x64 | -0.526 | -0.809 ~ -0.259 | -0.273 / -0.780 |
| neural_D1_24x64 − neural_D0_24x64 | -0.211 | -0.418 ~ -0.004 | -0.177 / -0.244 |
| tree_GEOQ_24x128 − tree_candidate_geometry_24x128 | -0.153 | -0.386 ~ +0.087 | -0.238 / -0.068 |
| tree_candidate_geometry_24x128 − tree_score_only_24x128 | +1.579 | +0.909 ~ +2.462 | +1.912 / +1.246 |
| tree_observed_geometry_24x128 − tree_score_only_24x128 | -0.044 | -0.257 ~ +0.191 | +0.146 / -0.234 |
| tree_observation_residual_24x128 − tree_score_only_24x128 | +0.222 | +0.058 ~ +0.408 | +0.724 / -0.280 |

입력 확대 첫 실행에서 acoustic batch-size에 따른 미세 수치 차이를 발견해 완료 처리하지 않았다. 기존 관측128/token64 batch shape를 복원하고 마지막 minibatch의 유효 예제만 반복한 뒤 결과를 버리는 방식으로 수정했다. 원래128개 query의 입력 및 후보/결정을 재검증했다. Raw RIR에 패딩을 넣거나 관측을 합성한 것이 아니다. 초기 실패 로그·산출물은 보존했다.

[full_query_results.json](evidence/full_query_results.json) · [full_evaluation.json](artifacts.json) (`full_evaluation.json`) · [FULL_QUERY_CONFIRMATION_PROTOCOL.json](artifacts.json) (`FULL_QUERY_CONFIRMATION_PROTOCOL.json`)

## 13B. 전체 학습query 확대와 반증

첫 전체query 확인에서 E0 신호가 약해졌으므로, 유지된 학습량 효과를 따라 입력 조건을 바꿨다. 12full6,387query /24full12,627query /24balanced6,387query. Tree score-only/GEOQ와 neural additive/concat/A→V/capacity를 동일 hyperparameter와 3-inner-fold selection으로 비교했다. 추가36refit+72inner=108fits, 전체358supervisedfits이다.

24balanced는 장면별 거의 같은 quota를 부여하되256개밖에 없는 장면의 부족분은 고정 순서로 재분배했다. 총량은 같지만12full의 불균등 장면별 가중치와도 차이가 있어 순수 diversity 인과효과로 단정하지 않는다. 이미 본11scene의6,072query를 다시 쓰므로 **탐색적 재사용 평가**다. 이번36개 모델을 모두 고정한 뒤 평가했고 최종 threshold를 다시 맞추지 않았다. Dev19도 이전 연구에서 사용된 development다.

그림: 전체 학습query 조건의 train/evaluation 곡선. 새 독립평가가 아니다. — 외부 산출물 `expanded_scaling.png` ([목록](artifacts.json)).

전체 학습query 조건의 train/evaluation 곡선. 새 독립평가가 아니다. · [PDF](artifacts.json) (`expanded_scaling.pdf`)

| Expanded head | R@1m | Joint | Fine | ΔGEOQ pp | 95% CI | Repair / harm % |
| --- | --- | --- | --- | --- | --- | --- |
| neural_D0_12full | 73.420 | 71.762 | 24.941 | -2.206 | -3.413 ~ -0.919 | 4.708 / 6.914 |
| neural_D0_24balanced | 73.274 | 71.653 | 24.803 | -2.315 | -3.529 ~ -1.078 | 4.816 / 7.131 |
| neural_D0_24full | 74.491 | 72.875 | 25.194 | -1.094 | -2.265 ~ +0.062 | 4.835 / 5.929 |
| neural_D1_12full | 73.402 | 71.695 | 24.883 | -2.274 | -3.520 ~ -0.924 | 4.695 / 6.969 |
| neural_D1_24balanced | 73.279 | 71.596 | 24.849 | -2.372 | -3.661 ~ -1.042 | 4.735 / 7.107 |
| neural_D1_24full | 74.338 | 72.750 | 25.135 | -1.219 | -2.378 ~ -0.058 | 4.962 / 6.181 |
| neural_D2_12full | 72.710 | 71.073 | 24.762 | -2.895 | -4.182 ~ -1.487 | 4.536 / 7.431 |
| neural_D2_24balanced | 73.062 | 71.370 | 24.793 | -2.598 | -3.914 ~ -1.296 | 4.638 / 7.236 |
| neural_D2_24full | 74.214 | 72.544 | 25.213 | -1.424 | -2.674 ~ -0.119 | 4.913 / 6.337 |
| neural_capacity_12full | 74.865 | 73.349 | 24.394 | -0.619 | -1.529 ~ +0.288 | 3.630 / 4.250 |
| neural_capacity_24balanced | 75.059 | 73.466 | 24.958 | -0.502 | -1.478 ~ +0.365 | 3.494 / 3.996 |
| neural_capacity_24full | 75.324 | 73.822 | 24.203 | -0.146 | -1.141 ~ +0.917 | 3.779 / 3.925 |
| tree_GEOQ_12full | 75.269 | 73.673 | 25.379 | -0.296 | -0.605 ~ -0.022 | 1.268 / 1.564 |
| tree_GEOQ_24balanced | 75.533 | 73.954 | 25.509 | -0.015 | -0.283 ~ +0.291 | 1.548 / 1.563 |
| tree_GEOQ_24full | 75.791 | 74.202 | 25.629 | +0.234 | -0.024 ~ +0.493 | 1.565 / 1.331 |
| tree_score_only_12full | 74.310 | 72.714 | 24.960 | -1.254 | -2.173 ~ -0.540 | 1.718 / 2.972 |
| tree_score_only_24balanced | 74.431 | 72.831 | 25.047 | -1.138 | -1.900 ~ -0.498 | 1.949 / 3.086 |
| tree_score_only_24full | 74.451 | 72.881 | 25.243 | -1.088 | -1.907 ~ -0.476 | 1.839 / 2.927 |

| Expanded contrast | Joint Δpp | 95% CI | Seed42 /43 |
| --- | --- | --- | --- |
| tree_GEOQ_24full − tree_GEOQ_12full | +0.530 | +0.213 ~ +0.891 | +0.354 / +0.705 |
| tree_GEOQ_24balanced − tree_GEOQ_12full | +0.281 | -0.043 ~ +0.616 | +0.331 / +0.231 |
| tree_GEOQ_24full − existing_GEOQ | +0.234 | -0.024 ~ +0.493 | +0.190 / +0.278 |
| tree_GEOQ_24full − existing_E0_GEOQ | -0.111 | -0.498 ~ +0.287 | +0.024 / -0.245 |
| neural_D2_24full − neural_D2_12full | +1.471 | +1.006 ~ +1.886 | +1.286 / +1.655 |
| neural_D2_24full − existing_GEOQ | -1.424 | -2.674 ~ -0.119 | -1.098 / -1.751 |
| neural_D2_24full − existing_E0_GEOQ | -1.769 | -2.828 ~ -0.671 | -1.264 / -2.274 |
| tree_GEOQ_24full − tree_score_only_24full | +1.322 | +0.622 ~ +2.195 | +1.284 / +1.359 |
| neural_D2_24full − neural_D1_24full | -0.206 | -0.465 ~ +0.039 | -0.054 / -0.358 |
| neural_D2_24full − neural_D0_24full | -0.331 | -0.742 ~ +0.048 | -0.187 / -0.475 |
| neural_D2_24full − neural_capacity_24full | -1.278 | -1.879 ~ -0.699 | -0.991 / -1.565 |
| tree_GEOQ_24balanced − tree_score_only_24balanced | +1.123 | +0.450 ~ +1.825 | +1.251 / +0.995 |

별도 단일 ambiguity gate는 기존 train-frozen entropy 상위tertile에서만 E0를 선택하고 나머지는 GEOQ를 유지했다. 11scene joint +0.296%p(CI−0.007~+0.702), Dev19 +0.103%p(CI−0.144~+0.391)여서 확정 개선으로 채택하지 않았다. Threshold를 탐색해 이 결과를 뒤집지 않았다. 이 policy도 전체query subgroup 결과에서 착안한 탐색적 가설이다.

[EXPANDED_TRAINING_PROTOCOL.json](evidence/EXPANDED_TRAINING_PROTOCOL.json) · [expanded_results.json](artifacts.json) (`expanded_results.json`) · [expanded_frozen_models.json](artifacts.json) (`expanded_frozen_models.json`) · [expanded_training_verification.json](artifacts.json) (`expanded_training_verification.json`) · [AMBIGUITY_GATE_PROTOCOL.json](artifacts.json) (`AMBIGUITY_GATE_PROTOCOL.json`) · [ambiguity_gate_results.json](artifacts.json) (`ambiguity_gate_results.json`)

## 13C. 학습 노출·장면 가중치 복원과 결론 수정

기존 GEOQ와 새 full-query head의 학습 방식에 두 차이가 있었다. 기존 모델은 oldTrain12의 B2 out-of-fold 입력과 장면별 동일 총 가중치를 사용했다. 앞선 새 head는 fullB2의 oldTrain12 in-sample 입력과 query별 동일 가중치를 썼다. 두 차이를 분리하기 위해 OOF-only 36refit+72inner와 OOF+scene-weight tree12refit을 추가했다. 이 단계까지 총**478 fits, 최종178개 checkpoint**다.

실제 장면 제외 B2 checkpoint로 old12의 두 시드12,774query-seed를 재실행해 원본 cache와 정확히 일치함을 확인했다. 12full GEOQ에 sorted scene 순서, scene-weight, 기존 train-selected threshold(seed42=0,seed43=.01)를 적용하자 개발19+재사용11의 총17,008query-seed에서 원본 점수 최대오차0.0, 모든 최종 후보 선택이 일치했다. 기존 강한 baseline을 재현한 상태에서 확장 효과를 비교했다. 추가 threshold 탐색은 하지 않았다.

**데이터 확대 효과가 학습 방식에 불변이라는 해석은 반증됐다.** OOF/query-weight에서는12→24full GEOQ +0.506%p(CI+0.271~+0.742)지만, OOF/scene-weight/threshold0에서는+0.158%p(CI−0.235~+0.460), 기존 threshold에서는+0.211%p(CI−0.147~+0.524)였다. 총 query를 고정한24balanced 이득도 확정되지 않았다. 장면별 관측 수 차이가 학습 목표의 가중치와 결합하므로, 초기 scaling 수치를 순수 장면 다양성의 효과로 일반화할 수 없다.

그림: 노출·가중치·operating point를 분리한 대조와 가설 반증 — 외부 산출물 `recipe_controls.png` ([목록](artifacts.json)).

노출·가중치·operating point를 분리한 대조와 가설 반증 · [PDF](artifacts.json) (`recipe_controls.pdf`)

| OOF / scene-weight model | R@1m | Joint | Fine |
| --- | --- | --- | --- |
| existing_E0_GEOQ | 75.700 | 74.313 | 25.824 |
| existing_GEOQ | 75.500 | 73.968 | 25.713 |
| scene_balanced/tree_GEOQ_12full/legacy_threshold | 75.500 | 73.968 | 25.713 |
| scene_balanced/tree_GEOQ_12full/matched | 75.569 | 74.057 | 25.750 |
| scene_balanced/tree_GEOQ_24balanced/legacy_threshold | 75.548 | 74.073 | 25.567 |
| scene_balanced/tree_GEOQ_24balanced/matched | 75.594 | 74.084 | 25.489 |
| scene_balanced/tree_GEOQ_24full/legacy_threshold | 75.660 | 74.180 | 25.757 |
| scene_balanced/tree_GEOQ_24full/matched | 75.678 | 74.215 | 25.765 |

| Scene-weight contrast | Joint Δpp | 95% CI | Seed42 /43 |
| --- | --- | --- | --- |
| scene_balanced/tree_GEOQ_24full/matched − scene_balanced/tree_GEOQ_12full/matched | +0.158 | -0.235 ~ +0.460 | +0.386 / -0.071 |
| scene_balanced/tree_GEOQ_24full/matched − scene_balanced/tree_score_only_24full/matched | +1.218 | +0.510 ~ +2.051 | +1.424 / +1.011 |
| scene_balanced/tree_GEOQ_24full/legacy_threshold − scene_balanced/tree_GEOQ_12full/legacy_threshold | +0.211 | -0.147 ~ +0.524 | +0.386 / +0.036 |
| scene_balanced/tree_GEOQ_24full/legacy_threshold − scene_balanced/tree_score_only_24full/legacy_threshold | +1.120 | +0.476 ~ +1.871 | +1.424 / +0.816 |

그림: B2 scene-exclusion만 복원한 query-weight matched learning curves — 외부 산출물 `oof_scaling.png` ([목록](artifacts.json)).

B2 scene-exclusion만 복원한 query-weight matched learning curves · [PDF](artifacts.json) (`oof_scaling.pdf`)

| OOF / query-weight contrast | Joint Δpp | 95% CI |
| --- | --- | --- |
| tree_GEOQ_24full − tree_GEOQ_12full | +0.506 | +0.271 ~ +0.742 |
| tree_GEOQ_24full − existing_GEOQ | +0.341 | -0.021 ~ +0.628 |
| neural_D2_24full − neural_D2_12full | +1.526 | +1.121 ~ +1.923 |
| tree_GEOQ_24full − tree_score_only_24full | +1.334 | +0.751 ~ +2.003 |
| neural_D2_24full − neural_D1_24full | -0.163 | -0.437 ~ +0.108 |
| neural_D2_24full − neural_capacity_24full | -1.195 | -1.650 ~ -0.767 |
| neural_D2_24full − donor | +1.538 | +0.970 ~ +2.077 |
| neural_D2_24full − disabled | -0.157 | -0.369 ~ +0.073 |

추가 candidate geometry의 이득은 OOF/query-weight +1.334%p, OOF/scene-weight/threshold0 +1.218%p, legacy threshold +1.120%p로 모두 두 시드에서 양수이며 scene CI도0보다 높았다. 이 결과가 이번 조건에서 가장 일관적인 기여다. 새로운 physical observation consistency를 증명한 것은 아니다.

OOF D2의12→24full 개선은+1.526%p이지만,24full에서 concat 대비−0.163%p(CI가0포함), geometry capacity basis 대비−1.195%p(CI−1.650~−0.767)였다. Matched−donor +1.538%p는 올바른 audio에 대한 의존성을 보여주지만, matched−gate-disabled −0.157%p(CI가0포함)여서 interaction gate의 추가 이득을 입증하지 못한다. Audio 정보가 유용하다는 것과 A→V 모듈이 필요하다는 것은 서로 다른 주장이다. D2의 fine은 capacity보다+0.300%p지만 joint 손실과 함께 보고한다.

모든 후속 평가는 이미 본11scene와 재사용Dev19다. OOF 복원도 visual teacher 노출이나 upstream/head fold의 비중첩 문제를 해결하지 않는다. 따라서 이 결과는 완전히 미노출 공간의 독립 일반화 검증이 아니다. Query-weight neural와 scene-weight tree의 수치를 같은 학습 목적이라고 섞지 않으며, geometry 기여는 같은 tree recipe끼리, interaction 기여는 같은 neural recipe끼리 비교한다.

[OOF_ALIGNMENT_PROTOCOL.json](evidence/OOF_ALIGNMENT_PROTOCOL.json) · [SCENE_WEIGHTING_PROTOCOL.json](evidence/SCENE_WEIGHTING_PROTOCOL.json) · [oof_checkpoint_replay.json](artifacts.json) (`oof_checkpoint_replay.json`) · [oof_results.json](evidence/oof_results.json) · [recipe_control_results.json](evidence/recipe_control_results.json) · [scene_balanced_results.json](artifacts.json) (`scene_balanced_results.json`) · [oof_training_verification.json](artifacts.json) (`oof_training_verification.json`) · [scene_balanced_training_verification.json](artifacts.json) (`scene_balanced_training_verification.json`) · [recipe_inference_replay.json](artifacts.json) (`recipe_inference_replay.json`)

## 14. 기여 후보와 반증

**가장 유망한 후보는 geometry-conditioned hypothesis utility ranking과 제한된 추가 evidence의 조합이다.** 핵심은 interaction 복잡도가 아니라 같은 후보 집합에서 repair가 harm을 넘는 조건이다. 기존 E0+GEOQ의 추가 입력 조합은 전체query에서 point gain은 남았으나 CI가0을 포함해 확정 개선으로 채택하지 않았으며, 이번 연구가 새로 발명한 architecture 또는 독립 새 공간 검증으로 표현하지 않는다.

지지: 여러 노출·가중치 조건에서 matched tree의 추가 candidate geometry. 조건부: query-weight 학습에서 관측 수 확대 이득; 기존 scene-weight 복원 후에는 CI가0을 포함하므로 recipe에 불변한 scaling 주장은 미지지. 미지지: 총 query를 고정한 scene 수 증가만의 개선, concat보다 강한 A→V, audio-conditioned visual adaptation의 최종 localization 이득. 미확정: 48/96scene의 효과, complete backbone adaptation, 미노출 pipeline의 일반화, 물리적 room/corridor 유형에 따른 인과 효과.

같은 저차원 interaction 변형을 더 반복하는 대신 유망한 frozen ranking 신호를 전체query로 확대해 반증을 시도했다. 이후 독립성 확보에는 teacher 미노출의 승인 scene 또는 upstream scene-disjoint 재학습과 추가 dense reference가 필요하다. 기존 Test를 탐색 데이터로 전환하지 않았다.

## 연구자가 얻을 수 있는 답

- **데이터 부족인가, 표현 문제인가?** query-weight 학습에서는 관측 수 증가가 일부 모델을 개선했지만 GEOQ scaling은 scene-weight 복원에 민감했다. Richness를 늘린 neural head는 geometry basis보다 약했다. 이 범위에서는 표본 수와 표현/목표의 적합성이 함께 제한한다. 전체 데이터 충분성을 확정할 수 없다.

- **어떤 모듈이 좋아졌는가?** OOF/query-weight의 D2는12→24full에서+1.526%p 개선됐지만 concat을 넘지 못했다. GEOQ의 같은 조건 이득은+0.506%p이며, 기존 scene-weight/threshold 복원 시+0.211%p(CI가0포함)로 약해졌다. 총량 고정의 안정적인 diversity 이득은 확인되지 않았다.

- **Interaction이 필요한 조건은?** 이번 대조에서는 확인되지 않았다. 올바른 audio에서만 나타나는 안정적 이득과 same-feature concat/late-fusion 초과가 함께 필요하다.

- **GEOQ의 장점과 실패 조건은?** compact evidence를 후보 구조와 연결하여 utility를 학습한다. 후보coverage 부족은 ranking으로 복구할 수 없고, 큰 bank/both-modalities-fail에서 정확도가 낮다. 실패 원인의 인과 판정은 별도다.

- **다른 문제로 가져갈 원리는?** GT-free 후보coverage → 조건부 판별 → 최종 pose → repair/harm을 분리하고, 정보량·학습량·parameter budget·upstream exposure를 각각 통제해야 한다.

## 계산 비용, 정합성, 재현

새 RIR 생성0개. 기존24Train scene의8,824 RIR task에서 scene별 median2.24–4.87초/RIR(중앙2.78초)을 확인했다. 병렬 walltime이나 scene 크기의 인과 비용이 아니다. 새23scene frozen 입력 생성은 task합582.9초, scene16.7–52.7초(공유 모델 초기화 제외). Docker 이미지 avfploc:cu124-clean, RTX4090 24GB를 사용했고 작업별 CPU/메모리 제한과 기존 서비스는 유지했다. 상세 비용/저장량은 RESOURCE_SUMMARY.json에 있다. 실측 peak VRAM을 지속 기록하지 않았으므로 peak 값을 꾸며내지 않는다.

기존 checkpoint와 사용자 dirty patch는 변경하지 않았다. 중간 실패(입력 반환 개수, renderer workdir/ABI, outdoor polygon, encoder batch shape)는 CORRECTIONS.json에 원인·수정·재검증을 기록했다. 체크포인트 replay, GT 항목 제거 inference, donor recipient reference 불변성, source query ID와 input receipt hash를 검사했다. 라이브러리/코드/input manifest와 재현 명령을 보존했다. 알려지지 않은 foundation 사전학습 corpus 노출은 미확인이다.

[REPRODUCE.md](archive/REPRODUCE.md) · [RESOURCE_SUMMARY.json](artifacts.json) (`RESOURCE_SUMMARY.json`) · [CORRECTIONS.json](artifacts.json) (`CORRECTIONS.json`) · [inference_verification.json](artifacts.json) (`inference_verification.json`) · [training_verification.json](artifacts.json) (`training_verification.json`) · [MODEL_INPUT_CONTRACT.json](artifacts.json) (`MODEL_INPUT_CONTRACT.json`) · [acoustic_geometry_contract_audit.json](artifacts.json) (`acoustic_geometry_contract_audit.json`) · [source_code_and_baseline_provenance.json](artifacts.json) (`source_code_and_baseline_provenance.json`) · [runtime_versions.json](artifacts.json) (`runtime_versions.json`) · [external_preservation.json](artifacts.json) (`external_preservation.json`) · [DELIVERABLES.json](artifacts.json) (`DELIVERABLES.json`) · [MANIFEST.json](artifacts.json) (`MANIFEST.json`) · [FINAL_VERIFICATION.json](evidence/FINAL_VERIFICATION.json)

> 역사적 연구 기록. 원본: `structural_evidence_20261107_1730/RESEARCH_REPORT.md`. 원본 SHA256은 [산출물 목록](artifacts.json)에 있습니다.
> 당시의 완료·재개 지시는 현재 실행 요청이 아닙니다. 최신 판정은 [현재 결론](../../../current_findings.md)을 먼저 확인하세요.

# AV-FPLoc: 후보 형상에 조건화한 hypothesis discrimination

2026-11-07 · 두 seed(42/43) · Train12 / 재사용 Val8·Val11 · Test 미사용

## 발견과 해석

새 센서 관측이나 cross-attention 없이, **기존 후보 bank에 이미 있는 floorplan 형상으로 기존 multimodal 점수를 조건화**하면 강한52차원 Utility Ranker를 개선할 수 있었다. 모든 주 비교에서 B2의 동일20개 support/refined 후보, 점수, 학습목표와 tree 예산을 유지했다. 따라서 이번 이득은 candidate coverage 증가가 아니라 후보 선택의 개선이다.

초기에는 compact visual residual이 새 matching evidence를 제공한다고 해석했다. 그러나 새 residual block에 다른 query를 넣고 재학습해도 이득이 남았다. 후보 형상만 사용한 GEO가 더 강했고, 같은104개 입력에서 score와 geometry 사이 tree interaction을 금지하자 이득이 사라졌다. 현재 지지되는 설명은 **후보 형상과 기존 점수의 결합을 ranker가 활용하지 못했다**는 것이다. 형상만으로 위치를 알 수 있다는 주장이 아니다. S와 B2 proposals에는 여전히 matched visual/acoustic 정보가 들어 있다.

Train OOF만 사용하는 명시적 선택 코드에서는 두 seed 모두 GEOQ(후보 형상26 + 관측 형상26 + 기존S26, 상대 특징 포함156차원)를 선택했다. seed42 threshold0, seed43 .01로 동결했다. 별도로 고정한 이전 threshold .025/.1에서 GEO(104차원)의 원인 분석 결과를 보고한다. 개발셋에서 가장 높은 GEO를 최종 선택으로 바꾸지 않았다.

## 프로토콜과 평가 무결성

- 기존 checkpoint, scores, support-NMS, local refinement, prior artifacts를 보존했다. B0 기존 선택19,930개 query-seed를 정확히 재현했다. Train6387 / Val8 4190 / Val11 5775 unique queries; 두 seed를 독립 표본으로 세지 않는다.
- HistGradientBoostingRegressor:150 iterations,15 leaves, minleaf100, L2=10, learning_rate=.07, early_stopping=False. Scene-balanced weights. 기존 utility `joint + .5*fine`에서 baseline후보 대비 gain 회귀. objective 변경 실험과 input 변경을 혼합하지 않았다.
- 기존 특징은 S26와 candidate0 대비 차이26. 확장도 같은 상대구조. 후보0 gain=0, 최고 gain이 threshold를 넘을 때만 전환. 특징 생성에는 GT/scene ID가 들어가지 않는다. GT는 supervision 및 사후 labels/사례 분석에만 사용한다.
- primary feature 비교 threshold는 과거에 고정한 .025/.1. 최종 선택은 Train12 outer-adapter OOF의 scene-macro joint 최대화, strongB0 이상의 macro fine 및 이하의 visual harm 조건. OOF는 선택에 사용되어 unbiased 최종 평가가 아니다.
- 기존 upstream OOF가 ranker의 held scene을 간접 노출할 수 있음을 발견했다. 4outer folds×9training scenes×2seeds=72개 feature cache를 해당 held3scene을 제외한 B2 checkpoint로 재생성했다. held3는 해당 fold의 기존 cache. 원래 enrichment와 새 코드가62scene-seed에서 정확히 일치함을 검증했다.
- 이 수정은 B2 adapter 수준이다. pretrained encoder는 다시 nested training하지 않았고, outer-training adapter in-sample 특징과 finalfit upstream-OOF 특징 사이 분포 차이도 남는다. feature 설계는 개발셋 탐색의 영향을 받았다.
- CI는 scene 단위 paired bootstrap10,000회; 두 seed를 scene별로 평균한 query-weighted 효과. 다수 탐색에 대한 보정 CI가 아니며, 독립 일반화 확증으로 해석하지 않는다. Scene별 원자료와 seed별 결과는 final_evaluation.json에 있다.
- 신규 독립 scene 데이터는 확보하지 않았다. Test 관측/결과를 열지 않았다. Val11을 untouched validation이라고 부르지 않는다.

## 접근 가능한 정보와 단계별 기여

상세 경로·receipt·shape·손실은 [INFORMATION_AUDIT.md](archive/INFORMATION_AUDIT.md), input_audit.json 및 signal_verification.json에 기록했다.

| 단계 | 실제 접근 가능한 정보 | 관찰 및 해석 한계 |
|---|---|---|
| Visual 관측/encoder | RGB 경로,40×128token,40depth/scale→11ray | 별도 physical validity/visibility mask 없음; uncertainty는 정답 ray 여부 보증 아님 |
| Candidate geometry | bank pose의11 expected rays, DESDF | 이미 존재했지만 S26에 풍부한 형상 분포가 직접 포함되지 않음; furniture/3D mismatch 가능 |
| Visual scalar | normalized residual 합과 bank z-score | 방향·부호·tail 및 절대 scale 축약. VC 일부 회복은 유효하나 matching 필요성은 donor 재학습에서 반증 |
| Acoustic 원신호 | 실제6channel RIR sample을31scene에서 확인 | 전체 raw FP bank의 가용성은 미확인. 원파형 전체를 학습한 실험 아님 |
| Preencoder | E6×19×95, C2×6×2×19, envelope6×252 | E/C2–128ms500–4000Hz; envelope500–1500Hz, amplitude 정규화. Full complex STFT/정확한 wall reflection label은 cached 입력 아님 |
| Embedding | query/FP128, E32/C4/N16 intermediates | AC/AM/INTER 추가는 해당 learner에서 개선 실패; 정보 부재의 증거가 아님 |
| Score/ranker | S26 absolute+relative52, V.85+A.15 | 강한 B0 정확 재현. 새로운 geometry conditioning으로 추가 개선 |

새로운 preencoder pooling은 E3frequency×8time, C2window×2complex×3frequency, envelope8time을 각 channel에 유지했다. Observed/FP의 yaw→base/roll 인덱스를 원 encoder로 확인했다(8예제; pooling≤1e-7, batch 차이에 따른 embedding error≤1.4e-4). SIG72 및 RAW528는 coarse signal probe이며 전체 waveform 정보의 상한이 아니다. 새로운 RIR 렌더링이나 wall correspondence는 만들지 않았다.

Acoustic FP embedding→11expected ray log-depth Ridge(alpha1) probe는 Train24,576map poses로 학습했다. Held-scene MSE는 train-mean predictor의 약45–46%로 줄었다. 따라서 geometry 정보 일부는 embedding에 선형적으로 읽을 수 있게 남아 있다. 하지만 GDecoded ranker는 direct GEO의 이득을 재현하지 못했다. 불완전한 선형 복원/learner 한계와 실제 정보 손실을 이 결과만으로 구분할 수 없다.

## 결과: 동일 후보의 discrimination

단위 %, metric 순서는 R@1m / R@1m&30° / strict R@0.1m. CI 표는 joint 변화pp.

| 방법 | Val8 R1 / joint / fine | Val11 R1 / joint / fine |
|---|---|---|
| B0 | 75.048 / 73.496 / 22.530 | 72.147 / 70.675 / 23.316 |
| VC | 76.014 / 74.475 / 22.768 | 72.563 / 71.195 / 23.420 |
| VCAC | 76.122 / 74.558 / 22.804 | 72.649 / 71.307 / 23.385 |
| Shape | 76.050 / 74.558 / 22.709 | 72.727 / 71.394 / 23.368 |
| GEO | 76.372 / 74.988 / 22.888 | 73.255 / 71.844 / 23.532 |
| GEOQ | 76.110 / 74.809 / 22.840 | 73.177 / 71.732 / 23.506 |
| GEO_constrained | 74.761 / 73.198 / 22.387 | 72.121 / 70.563 / 23.325 |
| GDecoded | 74.988 / 73.425 / 22.470 | 72.139 / 70.684 / 23.342 |
| SELECTED | 76.098 / 74.797 / 22.864 | 73.195 / 71.818 / 23.446 |

| 방법 vs B0 | Val8 joint Δ [95%CI] pp | Val11 joint Δ [95%CI] pp |
|---|---|---|
| VC | +0.979 [+0.633, +1.376] | +0.519 [+0.229, +0.784] |
| GEO | +1.492 [+0.868, +2.298] | +1.169 [+0.773, +1.638] |
| GEOQ | +1.313 [+0.681, +2.107] | +1.056 [+0.667, +1.515] |
| SELECTED | +1.301 [+0.656, +2.070] | +1.143 [+0.668, +1.724] |

고정 후보 joint oracle은 Val8 95.990%, Val11 97.342%로 동일하다. 선택된 모델도 큰 oracle gap을 남긴다. 선택된 모델의 Val11 fine 변화는+0.130pp, CI[-0.106,+0.382]pp로0을 포함한다. Fine 개선을 확정하지 않는다. .1m경계 query가 많아 `<=.1+1e-9`는 별도 diagnostic으로만 기록했다. Primary strict definition을 바꾸지 않았다.

| 선택 모델 | joint% | repair / harm vs B0 | visual-success 중 실패(B0→새 모델) |
|---|---:|---:|---:|
| Val8 seed42 |75.060|98 /54|110→99|
| Val8 seed43 |74.535|101 /36|131→110|
| Val11 seed42 |71.740|133 /58|148→144|
| Val11 seed43 |71.896|114 /57|178→150|

Visual 보존은 이 네 aggregate에서 개선되지만 모든 query/scene의 무악화를 보장하지 않는다. 선택 모델의 joint 증가는 Val8 seed42 6/8scene, seed43 8/8; Val11 seed42 11/11, seed43 9/11scene에서 양수였다. 두 seed 평균으로는19scene 모두 양수였다. Scene별 효과는 그림과 JSON에 있다.

## 반증과 경쟁 가설

1. **Embedding을 더 주면 해결되는가?** AC/AM 및 고차원V/VA/INTER 대부분 B0보다 악화. 같은 tree budget은 유지했으나 feature search 난도까지 같지는 않다. 용량 부족/과적합/표현 난도 가설은 열어 둔다.
2. **단순 feature 수 또는 score 재표현 때문인가?** NOISE/ScoreBasis는 이득 재현 실패. CAL 네 scalar 요약(26차원으로 확장)은 VC보다 작았다. 동일26새입력 및104총입력 통제를 포함했다.
3. **정확한 angular residual matching이 원인인가?** NoAngular가 이득 대부분 유지. 새 관측 donor로 재학습한 VC_donorV도 이득 유지. Candidate-only GEO/constant-query GEOResidual이 더 강함. Frozen donor에서 성능 저하만 보고 matching necessity를 주장하면 잘못이다.
4. **Acoustic 추가/explicit interaction이 필수인가?** VCAC 대비 donorA/noA 재학습도 비슷한 결과. VCSIG/VCSIGAC의 추가 이득은 일관되지 않음. VCACX/VAI는 engineered product이며 attention/물리적 reflection matching이 아니다.
5. **Absolute 특징만 충분한가?** VC_abs/Shape_abs에서 relative block을0으로 두고 재학습하면 이득 소실. 단 candidate0 relative encoding과 baseline-relative target의 조합을 검증한 것이며 모든 pairwise/listwise 학습의 우열을 뜻하지 않는다.
6. **단순 geometry prior만 충분한가?** GEO_only 및 additive 두75tree 모델은 실패. 더 엄격하게 같은104inputs/150trees의 interaction_cst로 score×geometry split을 금지한 GEO_constrained도 실패. 이 learner에서 이득은 joint conditioning에 의존한다. 최적화와 모델 class의 차이가 완전히 사라지는 인과 실험은 아니다.
7. **Geometry는 encoder에 사라졌는가?** 선형 복원 probe가 일부 보존을 입증. GDecoded의 실패는 완전 정보 부재를 입증하지 않음. Direct bank geometry를 활용하는 경로가 현재는 더 효과적이다.

새 방법은 hypothesis-level cross-attention을 제안하지 않는다. 기존 multimodal score/support/local statistics와 후보/관측 형상 분포를 tree splits에서 함께 평가한다. 물리적 wall identity를 두 modality가 교환하는 구조가 아니다. Expected failure는 map–observation mismatch, novel geometry, 후보 oracle 부재, 양 modality가 함께 틀리는 query, threshold/scene shift이다.

## Matched B1/B2 exchange 평가

기존 B1(private latent, observation-specific route 없음)과 B2(route[1])의8epoch 모델을 재평가했다. 각 fold/full의 train/held scenes, config hash, parameter count, budget을 대조했고 epoch0 module/decoder의 모든 tensor가 동일함을 확인했다. B1도 공유된 학습 가능 초기 latent 및 전역 학습 prior가 있어 완전히 독립 unimodal training이라고 부르지 않는다. 새 candidate ranker와 upstream B1/B2는 다른 수준의 실험이다.

| B2−B1 | Val8 joint Δ [95%CI] pp | Val11 joint Δ [95%CI] pp |
|---|---|---|
| visual_top1 | -0.203 [-0.697, +0.275] | +0.017 [-0.455, +0.520] |
| audio_top1 | +0.394 [+0.000, +0.765] | -0.087 [-0.446, +0.231] |
| fusion_map | +0.346 [+0.000, +0.701] | +0.061 [-0.256, +0.354] |
| support_first | -0.131 [-0.408, +0.156] | -0.182 [-0.327, -0.010] |
| support_oracle20 | +0.024 [-0.024, +0.083] | -0.026 [-0.101, +0.048] |
| fixed_B2_20_score_rank | +0.298 [+0.036, +0.544] | +0.139 [-0.179, +0.442] |

이전 frozen B2 routes-off 효과는 fusion Top1 Val8 +0.191pp / Val11 +0.095pp이며 CI는0을 포함했다.이는 학습된 모델에 대한 개입이며 위 matched-training 차이와 다르다.새 block donor/removed controls는 S/proposal에 남은 modality 정보를 제거하지 않으므로 end-to-end modality removal이 아니다. 전체 modality 제거의 새 upstream 재학습은 수행하지 않았다. 현재 geometry 개선을 설명하는 데 필수는 아니며, exchange가 보편적으로 무효라고 주장하지 않는다.

## 고정 failure subgroup과 query 증거

Train에서 고정한 ambiguity threshold와 이전 coverage로 group을 정의했다. 새 harm은 outcome-defined 기술적 집합이다. N과 repair/harm은 seed 평균이다. Oracle-present/B0-fail은 정의상 harm=0이므로 큰 효과를 모집단 효과로 혼동하지 않는다.

| Group | Val8 N; Δjoint pp; repair/harm | Val11 N; Δjoint pp; repair/harm |
|---|---|---|
| all | 4190; +1.301; 99.5/45 | 5775; +1.143; 123.5/57.5 |
| visual20_fail_audio20_success | 325.5; -0.154; 9.5/10 | 469; +2.239; 22/11.5 |
| audio20_success_audio1_fail | 2514.5; +1.273; 67/35 | 3580.5; +1.271; 84.5/39 |
| both1_fail_fusion20_success | 1486.5; +2.153; 65/33 | 2229; +1.817; 79.5/39 |
| oracle_present_B0_fail | 942.5; +10.557; 99.5/0 | 1540; +8.019; 123.5/0 |
| high_visual_ambiguity | 1486; +1.346; 51/31 | 1543.5; +1.328; 55/34.5 |
| modality_disagreement | 3509.5; +1.496; 96/43.5 | 4926.5; +1.279; 118.5/55.5 |
| new_harm_B0_success | 45; -100.000; 0/45 | 57.5; -100.000; 0/57.5 |

Visual20실패/Audio20성공 subgroup은 Val8에서−0.154pp, Val11에서+2.239pp로 일관되지 않는다. 개선이 모든 acoustic rescue case를 해결한다는 주장은 반증된다. 각 subgroup의 oracle 존재 count, scene/seed CI와 모든 raw score/geometry 사례는 final_evaluation.json 및 case_attribution.json에 있다.

사례는 각 split/group에서 사전 정의한 scene/query 정렬의 첫 repair/harm을 선택했다(최대이득 사례 선별 아님). 사례 분석은 fixed-threshold GEO를 대상으로 한다. 26예제의 old/new S26, expected11rays, observed11rays, 실제 pose와 gain을 보존했다. Frozen 모델에서 모든 후보의 새 geometry를 old-selected geometry로 바꾸면19/26예제에서 old/new gain 순서가 뒤집혔다. 이는 모델 민감도이지 물리적 인과/재학습 제거 실험이 아니다.

- Val8 scene102344457 query127: 새 후보는 visual1.336/audio3.206, 기존선택은1.308/3.113. Predicted gain .249 vs .166으로 정답을 복구; geometry 대체 시 순위 반전. 두 scalar가 모두 조금 더 좋다는 사실만으로 B0가 선택하지 못했던 후보를 geometry-conditioned head가 선택했다.
- Val11 scene102344115 query260: 새 후보의 visual은1.412로 기존1.433보다 낮지만 audio2.877은 기존2.563보다 높다. Gain .061로 전환해 복구; geometry 대체 시 순위 반전. 모달간 score tradeoff를 geometry 조건에서 다르게 평가한 사례다.
- 같은 Val11 scene query800: visual1.794>1.703이나 audio2.198<3.091인 후보로 잘못 전환, gain .044. 같은 tradeoff 방식이 harm도 만든다. 모델 confidence calibration으로 해석하면 안 된다.

실패/특징 중첩은 관측 정보의 본질적 부재를 증명하지 않는다. 모델이 어떤 입력에 민감했는지, 재학습 통제에서 무엇이 필요했는지, 실제 물리적 대응의 진실 여부를 분리한다.

## 실패한 시도 전체와 실행 비용

아래는 고정 prior threshold에서 모든 완료된34arms의 joint%. 세부 seed/scene/R1/fine/repair/harm은 final_evaluation.json. 최종 SELECTED는 별도행의 threshold 선택 결과이며 이 표에서 제외했다.

| Arm | Val8 seed42 /43 | Val11 seed42 /43 |
|---|---|---|
| AC | 72.983 / 72.434 | 69.714 / 70.338 |
| AM | 73.484 / 72.506 | 69.991 / 70.528 |
| B0 | 74.010 / 72.983 | 70.442 / 70.909 |
| CAL | 74.678 / 73.031 | 70.476 / 71.065 |
| GDecoded | 73.962 / 72.888 | 70.528 / 70.840 |
| GEO | 75.513 / 74.463 | 71.654 / 72.035 |
| GEOQ | 75.322 / 74.296 | 71.619 / 71.844 |
| GEOResidual | 75.513 / 74.487 | 71.481 / 71.879 |
| GEO_additive | 73.747 / 72.745 | 70.182 / 70.857 |
| GEO_additive_reverse | 73.031 / 72.840 | 69.610 / 70.580 |
| GEO_constrained | 73.604 / 72.792 | 70.390 / 70.736 |
| GEO_only | 72.936 / 72.315 | 69.749 / 70.424 |
| INTER | 73.174 / 72.315 | 69.593 / 70.459 |
| NOISE | 73.795 / 72.578 | 70.528 / 70.615 |
| NoAngular | 74.845 / 73.771 | 71.013 / 71.411 |
| QGEO | 73.819 / 72.721 | 70.442 / 70.615 |
| RAW | 73.819 / 72.458 | 70.216 / 70.511 |
| SIG | 73.866 / 72.721 | 70.390 / 70.736 |
| ScoreBasis | 73.675 / 72.363 | 70.407 / 70.597 |
| Shape | 75.084 / 74.033 | 71.082 / 71.706 |
| Shape_abs | 72.601 / 72.363 | 69.541 / 70.286 |
| V | 73.628 / 72.411 | 70.130 / 70.632 |
| VA | 73.508 / 72.315 | 69.957 / 70.528 |
| VAI | 73.413 / 72.363 | 70.147 / 70.494 |
| VA_abs | 72.697 / 72.291 | 69.645 / 70.338 |
| VC | 75.084 / 73.866 | 70.840 / 71.550 |
| VCAC | 75.060 / 74.057 | 70.996 / 71.619 |
| VCACX | 74.893 / 73.699 | 70.892 / 71.481 |
| VCAC_donorA | 75.179 / 73.819 | 70.909 / 71.481 |
| VCAC_noA | 74.964 / 73.890 | 70.978 / 71.481 |
| VCSIG | 74.916 / 73.819 | 70.874 / 71.515 |
| VCSIGAC | 74.964 / 73.938 | 71.134 / 71.307 |
| VC_abs | 72.721 / 72.220 | 69.506 / 70.320 |
| VC_donorV | 74.678 / 73.675 | 71.117 / 71.377 |

GEOQ의 GT-free inference API가 모든19,930개 development query-seed의 cached gain과 정확히 일치했다. CPU2threads, scene batch, warm S/rays 기준 head+summary 비용은 seed42 0.121ms/query, seed43 0.132ms/query. Encoder/fullbank scoring/NMS는 제외한 추가 head 비용이며 동시 연구 job이 있었다. End-to-end speedup 주장 아님. 모델당 약475KB. 원래 S와 후보11ray를 재사용하므로 새 sensor preprocessing은 배치에 필요하지 않는다. 상세 timing은 selected_inference_verification.json, 각 학습/추출 wall time은 result.json/receipt에 보존했다. 서로 병렬이므로 단순 합을 GPU-hours로 해석하지 않는다.

## 검증된 사실 / 경쟁 가설 / 미해결

**검증된 사실(현재 data/protocol 범위):** 고정 후보·objective에서 geometry-conditioned discrimination 개선이 두seed와 강화된TrainOOF에서 재현됨. 더 강한direct geometry가 candidate bank에 이미 존재함. Score×geometry interaction을 제한하면 이득 소실. 일부geometry정보는 acoustic embedding에서 선형 복원 가능. 새matching residual이나 새 acoustic block은 이득의 필수조건이 아님.

**가장 유력한 해석:** scalar score/support를 floorplan 형상에 따라 해석하는 기능이 기존ranker에서 부족했다. 기존 representation의 완전한 부재보다 현재 discrimination에서의 활용 부족이 이번에 해결한 병목이다. Candidate geometry가 scene layout prior를 제공하는 효과와 물리적 evidence reliability를 구분하는 추가독립scene 실험은 필요하다.

**미해결:** 원파형의 phase/earlyreflection/absoluteamplitude가 실제로 추가 이득을 줄지, 더 강한embeddingdecoder가direct geometry를 대체할지, frozenencoder 노출이 없는scene에서 유지되는지, simulator/furniture mismatch가 어느failure를 유발하는지. 현 데이터는 simulator/real 또는 furniture조건을 무작위 배정하지 않았으므로 그 원인을 scene bias와 분리해 입증하지 못했다. Fine 경계민감성과 threshold drift도 남는다.

## 다음 투자 순서와 일반 원리

1. **Back-end 우선:** 동결한 GEOQ와 더 간단한 GEO를 새로운 scene에서 검증하고, candidate-conditioned score 해석 및 relative decision을 연구한다. 현재는 새 acoustic encoder나 attention보다 직접적인 근거가 있다.
2. **Front-end는 failure 목적을 좁혀 투자:** geometry로도 남는 oracle-present failures에서 full RIR earlyarrival/phase/amplitude 및3D/furniture mismatch를 별도 matched 조건으로 다룬다. coarse pooling 실패를 waveform 무용성으로 일반화하지 않는다.
3. **Cross-modal interaction은 구체적인 교환 정보가 확인될 때:** matched retraining과 same-feature nointeraction 대조를 유지한다. Cross-attention 추가 자체를 연구 성공 기준으로 삼지 않는다.

다른 multimodal localization에서도 availability, coverage, discrimination을 따로 측정하고, 고정 후보에서 evidence의 효과를 검증해야 한다. Frozen modality 교란은 OOD 민감도일 수 있으므로 재학습 donor/제거 통제를 병행한다. 구조를 복원할 수 있다는 것과 decision에 충분한 fidelity로 활용한다는 것은 다르다. 높은TopK는 실제Top1성능이 아니며, 새 정보 없이 기존 bank context 활용만으로도 ranking이 개선될 수 있다.

## 산출물과 재현

- [results.png](artifacts.json) (외부 산출물: `results.png`), [results.pdf](artifacts.json) (외부 산출물: `results.pdf`): controlled effects, failure groups, scene robustness, nested OOF.
- [REPRODUCE.md](archive/REPRODUCE.md): 실행 순서·환경·freeze/selection·GT-free API.
- final_evaluation.json / matched_B1_analysis.json / case_attribution.json: seed/scene/subgroup 및개별예제.
- final_selection_seed42/43.json: Train-only selection criterion, 전체candidate options 및model SHA256.
- GEOQ_seed42/43/model.joblib 및 inference.py: 동결된방법. 기존B2 checkpoint와후보추출을그대로연결한다.
- final_verification.json / artifact_manifest.json: baseline/input/code 보존, 완료검사, 최종checksum.
- 이전coverage연구: ../modality_coverage_20261107_1700/RESEARCH_REPORT.md 및results.png. Pure/exchanged 구분, K1/2/3/5/10/20, density/random/diverse controls, ambiguity/complementarity 원자료가그곳에보존되어있다.

이번 연구는 첫 batch 이후 residual → donor → candidate geometry → interaction restriction → embedding decoding → outer OOF로 반증과 수정을 반복했다. 재현 가능한 개선과 그 개선에 필요한 조건을 확인해 이 연구 단계를 마친다. 독립 일반화와 원파형 정보의 한계까지 해결했다고 주장하지 않는다.

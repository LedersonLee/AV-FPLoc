# AVLoc reproduction 결과 검토

- 연구 ID: `avloc_reproduction_review_20261117`
- Track / 상태: Track A / 기존 seed42 benchmark의 문서·구현·수치 감사 완료 (새 학습·평가 실행 없음)
- 증거 수준: **Audit only**. 검토 기준은 `f1fce03`의 `2026-11-11-avloc-independent` 결과와 원문 보존본이다.
- 범위: full Val19 (19 scenes, 9,965 queries), Test17 미사용. Val19는 재사용 development이며 seed42 한 개다.

## 핵심 판정

현 저장소에는 이제 AVLoc 원문 구조를 명시적으로 적응·구현한 A0/A1/A2/A3 benchmark가 있다. 이전의 FP-RIR score-bank Top20 coverage와 달리 이 결과는 raw six-channel echo와 RGB에서 ray distance/Laplace scale을 예측하고, direct MAP pose의 R@0.1m/R@1m/Joint(1m,30°)를 낸다. 단, 원문과 동일한 binaural STFT sensor가 아니라 **HSSD six-channel echo adaptation**이고, seed42 한 개·재사용 Val19이므로 논문 수치의 독립 재현 또는 일반화 증거는 아니다.

## 수치 및 modality 격차

동일 20cm XY / 10° yaw 후보 grid의 matched Val19 결과는 다음과 같다. B2/GEOQ는 기존 seed42/43 평균, AVLoc-style arms는 seed42다.

| 방법 | Position≤0.1m | Position≤1m | Joint≤1m/30° |
|---|---:|---:|---:|
| Existing visual |17.43%|46.99%|45.85%|
| FP-RIR cosine acoustic |5.35%|30.38%|21.36%|
| Fixed score fusion |24.28%|66.75%|65.04%|
| B2+GEOQ |23.20%|74.42%|73.07%|
| A0 vision rays |22.03%|50.95%|50.11%|
| A1 acoustic rays |0.15%|9.25%|2.43%|
| A2 joint rays |30.47%|65.72%|64.93%|
| A2 interaction-disabled |12.75%|41.71%|40.34%|
| A3 learned routing |30.92%|67.75%|66.92%|

A1 acoustic-only가 매우 낮다는 것은 구현 오류의 직접 증거는 아니지만, 이 six-channel adaptation에서 원문 AVLoc의 `EA=46.7% R@1m`과 같은 standalone acoustic localization은 재현되지 않았음을 뜻한다. A1 ray MAE는 1.4785m이고, 낮은 ray-error 사분위도 Joint 7.627%에 그친다. 따라서 현재 evidence는 “audio가 원문 수준으로 단독 기하를 푼다”는 해석을 지지하지 않는다.

반면 A2는 disabled control보다 Joint +24.482pp (scene bootstrap 95% CI [+21.565,+27.330])이고, A3 learned routing은 동일 head의 uniform routing보다 Joint +19.182pp [+16.351,+21.977]다. 이는 이 development cohort에서 joint design 및 learned weighting의 효과를 지지한다. 다만 uniform control은 **learned routing으로 학습된 head에 가하는 inference intervention**이므로, 별도 학습한 fixed-router 대비나 독립 calibrated reliability의 증거는 아니다.

근거: [통합 연구 보고서](../2026-11-11-avloc-independent/RESEARCH_REPORT.md), [요청 지표 표](../2026-11-11-avloc-independent/REQUESTED_METRIC_TABLE.md), [uniform routing 대조](../2026-11-11-avloc-independent/A3_UNIFORM_CONTROL.md).

## 10cm grid와 비용

10cm는 floorplan raster 해상도가 아니라 inference position-candidate spacing이다. 동일 frozen model을 20cm에서 10cm grid로 재평가하면 A2는 fine49.33%/Joint74.18%, A3는 fine49.47%/Joint76.19%가 된다. A3−A2 Joint는 +2.007pp [+1.228,+2.787], fine +0.140pp의 CI는 0을 포함한다. 이는 grid intervention이지 학습 개선이 아니다.

native 10cm B2/GEOQ는 FP-RIR reference location을 새로 렌더해야 하므로 실행되지 않았다. interpolation으로 대체하지 않았고, 따라서 10cm A2/A3과 20cm B2/GEOQ의 절대 성능을 우열 비교하면 안 된다. spacing을 절반으로 하면 동일 면적의 position candidate는 약 4배가 되며, FP-RIR bank라면 reference render·storage·matching 부담도 함께 커진다. AVLoc-style arms에는 FP-RIR inference bank가 없다. A0/A1/A2 epoch wall-time 합계는 각각 10.56/3.16/12.40시간이나, shared host에서 측정되어 isolated GPU-hour 또는 end-to-end latency가 아니다.

근거: [grid/calibration 결과](../2026-11-11-avloc-independent/GRID_AND_A2_CALIBRATION.md), [완료 비용 기록](../2026-11-11-avloc-independent/COMPLETED_TRAINING_COST.md).

## 구현 충실도와 한계

구현 감사는 ResNet50/18, 11/36 ray head, 4-layer joint expert, uncertainty-detached per-ray routing, moment matching, Laplace likelihood와 clean-Val early selection을 확인했다. audio sensor/feature protocol은 원문 48kHz binaural 4-channel STFT와 다르며 six-channel HSSD echo를 쓴다. 그러므로 구조적 reproduction은 상당 부분 이뤄졌지만 sensor/data-level exact reproduction은 아니다.

- Test17은 selection 또는 evaluation에 쓰지 않았으나, Val19는 reused development다.
- AVLoc-style 결과는 seed42 하나다. scene bootstrap은 장면 변동성만 측정하며 training-seed 재현성을 대신하지 않는다.
- A3 calibration은 sharpness/NLL과 top-1 success의 일치를 보장하지 않는다. 10cm에서 A2 temperature calibration은 Joint ECE와 Brier를 악화시켰다.
- A2/A3은 broad Joint에서 20cm B2+GEOQ를 넘지 못한다. A3은 fine precision과 FP-RIR-free inference bank 측면의 후보일 뿐, 전면 대체 근거가 아니다.
- 원문 공개 코드/checkpoint의 긍정적 확보는 확인되지 않았다. 따라서 원문 baseline 재학습·split·router 세부의 독립 bitwise 감사는 불가하다.

## 활용 원칙

보고서나 후속 대화에서는 `A3 10cm fine49.47%/Joint76.19%`를 **single-seed, reused-Val19, grid-specific development result**로만 인용한다. B2+GEOQ의 20cm Joint73.07%와 직접 순위를 매기지 말고, native 10cm B2 reference rendering과 seed43 또는 independent held-out 평가 뒤에 비교한다.

이 문서는 새 실험 실행 승인이 아니며, 기존 artifact와 negative result를 재해석해 삭제하지 않는다.

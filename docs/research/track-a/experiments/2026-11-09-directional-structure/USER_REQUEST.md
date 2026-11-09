# AV-FPLoc Track A — Directional Acoustic Structural Evidence

## 1. Research Objective

AV-FPLoc Track A의 다음 연구를 수행하라.

**핵심 목표: 6-channel Room Impulse Response(RIR)에서 방향별 공간 구조를 추정할 수 있는지 검증하고, 그 정보가 Visual Ambiguity를 해소하여 Floorplan Localization을 개선하는지 규명한다.**

기존 Acoustic Encoder는 Floorplan 기반 Reference RIR과의 Contrastive Matching을 통해 유용한 Localization 정보를 제공한다.

그러나 이 Embedding이 구체적으로 어떤 방향별 공간 구조를 반영하는지, 해당 구조를 직접 추정해 Floorplan과 비교할 수 있는지는 충분히 규명되지 않았다.

이번 연구에서는 다음 질문에 답하라.

1. RIR로부터 전방·좌측·후방·우측의 상대적 거리 또는 공간 개방도를 추정할 수 있는가?
2. 그러한 정보는 같은 위치에서 반복 측정해도 안정적인가?
3. Visual이 구별하기 어려운 Pose Hypotheses의 공간 구조 차이를 Acoustic이 구별할 수 있는가?
4. 추정한 Acoustic Geometry를 FP Geometry와 직접 Matching하면 Localization에 유용한가?
5. 기존 Cosine Embedding Matching 및 B2+GEOQ에 추가적인 가치를 제공하는가?

정확한 Acoustic Depth Estimation을 성공의 필수조건으로 삼지 않는다.

상대 거리, 공간 개방도, 방향별 구조적 차이 등 더 단순하고 안정적인 정보도 유효한 연구 성과로 인정한다.

---

## 2. Scope & Existing Research

이번 실험은 **Track A에 속하며, Track B의 새로운 FP-RIR-free Tri-branch Framework와 독립적으로 수행한다.**

기존 B2+GEOQ Framework를 보존하고, Acoustic Structural Evidence의 새로운 기여를 별도로 평가한다.

### Previous Findings

- Acoustic은 Visual Ambiguity가 높은 상황에서도 높은 Top-K Coverage를 제공한다.
- Acoustic의 시간적 에너지 형태, 정규화된 Envelope, 초기 복소 채널 관계에는 Localization 정보가 존재한다.
- Visual이 유사하게 평가하는 Pose Hypothesis Pair에서 Acoustic Score가 높은 판별력을 제공한다.
- 기존 Angular Steering/Fourier 표현은 GEOQ를 안정적으로 능가하지 못했다.
- 6개 수평 마이크의 Angular Response는 넓으며 정밀한 방향 분해능이 입증되지 않았다.
- 이전 Tri-branch Pilot의 Stage A에서 Acoustic-only coarse direction이 관측 가능하다고 보고했으나, 정량적 근거가 요약 보고서에 충분히 제시되지 않았다.

### Required Prior Audit

**새 실험 전에 이전 Stage A의 실제 코드와 산출물을 반드시 확인하라.**

- 어떤 방향별 목표를 예측했는가?
- 어떤 입력 Feature와 모델을 사용했는가?
- 어떤 Ground Truth 및 Loss를 사용했는가?
- Scene-held-out Accuracy와 Baseline은 무엇인가?
- 실제로 학습이 수행됐는가?
- 결과는 Pose Discrimination까지 연결됐는가?
- 이미 완료된 실험과 추가 검증이 필요한 실험은 무엇인가?

기존 실험을 불필요하게 반복하지 말 것.

보고서의 결론과 실제 저장된 결과가 불일치하면 이를 명시하고 근거를 다시 검증한다.

---

## 3. Phase A — Physical Observability & Dataset Construction

### A1. Acoustic Signal Audit

기존 SoundSpaces2.0 및 HSSD 데이터 자산을 재사용한다.

- 6-channel RIR
- 48 kHz sampling
- 반경 약 5cm 수평 원형 마이크 배열
- 배열 중앙의 Acoustic Source
- E/C/N 전처리 특징
- Raw waveform과 초기/후기 응답
- 기존 Acoustic Encoder 중간 및 최종 특징

Source–receiver geometry, 좌표계, yaw, 채널 순서, renderer 설정을 검증한다.

특히 다음 사항에 주의한다.

- 중앙 송신기의 Direct-path Timing은 거의 일정할 수 있다.
- 다중경로 Peak를 특정 벽의 반사라고 단정하지 않는다.
- 채널별 응답의 동시성·위상 일관성을 확인하지 않고 DoA를 물리적 정답으로 해석하지 않는다.
- 기존 CNN Token의 순서를 실제 방위각으로 간주하지 않는다.
- 2D FP의 벽 거리와 실제 3D Acoustic Propagation을 혼동하지 않는다.

### A2. Directional Geometry Targets

우선 4개의 Body-frame 수평 방향을 사용한다.

- Front: 0°
- Left: +90°
- Back: 180°
- Right: −90°

Body 좌표계 및 Yaw Convention을 기존 프로젝트와 일치시킨다.

후보별 2D Floorplan Ray-casting으로 다음 학습 목표를 생성한다.

**Target A — Metric Distance**

각 방향에서 구조적 경계까지의 거리.

**Target B — Ordinal Distance**

Near / Medium / Far 또는 상대적인 거리 순서.

**Target C — Directional Openness**

특정 방향이 상대적으로 열려 있는지, 가까운 구조적 경계가 존재하는지.

**Target D — Directional Uncertainty**

각 방향의 추정 불확실성.

Metric Distance가 불안정하면 Ordinal 또는 Openness 예측을 우선 검증한다.

Open/Closed 및 Near/Far 기준은 Train 분포와 물리적 의미를 이용해 사전에 고정한다.

FP Ray Label은 구조적 감독 신호이며 실제 Acoustic Reflection의 Ground Truth가 아니다.

### A3. Dataset Integrity

- 기존 HSSD 자산과 FP Quality Audit 재사용
- Scene-grouped Train/Evaluation
- Training Scene과 평가 Scene의 중복 방지
- Upstream Encoder의 과거 Scene 노출 명시
- GT Pose는 학습 Label 및 평가에만 사용
- 새로운 독립 Scene이 없으면 일반화 주장을 제한
- 기존 Test는 열지 않음

---

## 4. Phase B — Acoustic Directional Geometry Prediction

**첫 번째 핵심 실험이다.**

RIR에 실제로 방향별 공간 구조를 예측할 정보가 있는지 검증하라.

### B1. Required Baselines

동일한 Scene Split에서 다음을 비교한다.

1. Constant / Train-mean prediction
2. Raw signal 기반의 단순 Acoustic Descriptor
3. E/C/N 기반 Linear Probe
4. 기존 Acoustic Encoder의 Frozen Embedding Probe
5. 작은 Trainable Acoustic Geometry Predictor
6. 독립적으로 발견한 유망한 대안

새로운 대규모 Transformer를 먼저 구현하지 말 것.

가능한 최소한의 모델로 정보의 관측 가능성을 확인한다.

### B2. Feature Candidates

다음은 연구 가설이며 효과가 있다고 전제하지 않는다.

- Early reflection timing
- Relative echo delay
- Time-energy distribution
- Normalized envelope
- Inter-channel delay
- Complex coherence
- Spectral / channel relationships
- Learned Acoustic intermediate representations

실험 결과에서 유망한 Feature를 식별하고 경쟁 가설을 검증한다.

### B3. Required Metrics

**Metric Distance**
- Mean Absolute Error (MAE)
- Median / P90 Error
- Direction-wise Error

**Ordinal / Openness**
- Balanced Accuracy
- AUROC
- Directional Rank Correlation
- Pairwise Near/Far Ordering Accuracy

**Spatial Consistency**
- Yaw rotation consistency
- Direction permutation / channel rotation tests
- Repeated rendering stability
- Scene-held-out performance

**Information Controls**
- Same-scene donor RIR
- Channel permutation
- Single-channel vs multichannel
- Direction-independent prediction
- Capacity-matched non-directional representation

GT-free 입력만으로 물리적 방향 정보를 예측하는지 확인한다.

단순 Room Size나 Scene Identity의 Prior를 학습한 결과와 실제 Acoustic Directional Evidence를 구분하도록 대조한다.

---

## 5. Phase C — Visual Ambiguity Resolution

**이번 연구에서 가장 중요한 Localization 관련 실험이다.**

예측된 Acoustic Geometry가 실제로 Visual의 모호한 Pose Hypotheses를 구별하는지 평가하라.

### C1. Ambiguous Hypothesis Pairs

기존 Visual-only Score를 사용하여 다음 후보 쌍을 추출한다.

- Visual Score가 유사한 두 후보
- 공간적으로 분리된 후보
- 서로 다른 FP Directional Geometry를 가진 후보
- 가능하면 GT 기준 Positive/Negative가 함께 존재하는 후보 쌍

후보 쌍 선정은 GT-free Visual/FP 특성으로 수행하고, GT는 사후 성공 Label과 학습 감독에만 사용한다.

기존 B2 Top-20 후보를 사용할 수도 있지만, 반드시 Pure Visual 기반 후보 집합에서도 별도 평가하라.

그렇지 않으면 Acoustic으로 이미 추출된 후보를 이용해 Acoustic Guidance를 평가하는 문제가 발생한다.

### C2. Acoustic Structural Discrimination

각 후보의 FP Directional Geometry와 예측 Acoustic Geometry를 비교한다.

예를 들어:

- 올바른 후보는 오른쪽으로 공간이 열려 있음
- 잘못된 후보는 오른쪽이 가까운 벽으로 막힘
- Acoustic Predictor가 오른쪽 개방도를 높게 예측

이 차이를 이용해 후보를 구별할 수 있는지 검사한다.

### C3. Required Comparisons

- Visual Score only
- 기존 Acoustic Cosine Matching
- Visual + 기존 Acoustic Scalar Fusion
- Predicted Acoustic Geometry + FP Structural Matching
- Visual + Acoustic Structural Matching
- 기존 B2/GEOQ

### Primary Metrics

- Visually Ambiguous Pair Accuracy
- Candidate Ranking AUROC
- High Visual Ambiguity Success
- GT-success Candidate Rank
- Correct / Incorrect Hypothesis Score Margin
- Repair / Harm

**Acoustic Directional Prediction Accuracy와 실제 Localization Discrimination Gain을 반드시 구분하라.**

방향별 Depth 예측이 정확해도 Pose Discrimination에 추가 이득이 없다면 그 이유를 분석한다.

---

## 6. Phase D — FP-based Acoustic Structural Matching

Phase B/C에서 유용한 정보가 확인되면, 기존 FP-RIR Reference를 사용하지 않는 별도의 Acoustic Structural Matching Score를 구현한다.

이것은 Track B 전체 Framework의 개발이 아니라, Track A의 Acoustic Evidence 연구를 위한 독립적인 Matching Baseline이다.

### D1. Matching Formulation

각 후보 Pose p에서 FP Directional Geometry를 계산한다.

Acoustic Query에서 예측한 구조적 정보와 비교하여 Candidate Score를 생성한다.

검토 가능한 방식:

- Robust Distance Residual
- Near/Far Ordinal Agreement
- Directional Openness Agreement
- Uncertainty-weighted Geometry Consistency
- Joint Directional Structure Matching

한 방식의 성공을 전제하지 말고 서로 다른 비교 목적을 구분하라.

### D2. Important Controls

- Candidate Geometry prior only
- Acoustic observation only
- Matched Acoustic + FP Geometry
- Donor Acoustic + FP Geometry
- Direction-shuffled Acoustic + FP Geometry
- Existing FP-RIR Cosine Matching

단순히 FP Geometry Prior만으로 높은 성능을 얻는 문제를 방지하라.

### D3. Evaluation

가능하다면 Full SE(2) Candidate Bank에서 Structural Acoustic Score Field를 생성하라.

다음 지표를 평가한다.

- Acoustic Top-1 / Top-5 / Top-20
- GT-neighborhood Score Concentration
- Visual Ambiguity 조건별 Coverage
- Candidate-wise Discrimination
- Full-bank Computation Cost
- FP-RIR Reference 생성 및 저장 비용 의존성

이 실험은 기존 B2 Acoustic Matching의 대체 가능성과 보완 가능성을 모두 검토한다.

---

## 7. Phase E — Integration with Existing Track A Framework

유망한 Acoustic Structural Evidence가 확인된 경우에만 기존 B2/GEOQ와의 결합 실험을 수행한다.

### Integration Arms

- E0: Existing B2 + GEOQ
- E1: Acoustic Structural Score + Visual Score
- E2: Existing Acoustic Cosine Score + Structural Score
- E3: Existing B2 + Structural Evidence + Adapted GEOQ
- E4: Independent Alternative

각 Arm에서 입력 정보와 후보 집합의 차이를 명시한다.

### Evaluation Principles

1. 같은 Top-20 후보를 고정하여 Discrimination 효과 측정.
2. 새로운 Acoustic Score로 후보를 추출하는 Full-bank 성능을 별도 측정.
3. GEOQ는 입력 특징 변경 시 재학습.
4. 단순 Score Addition과 학습형 결합을 비교.
5. 기존 Strong GEOQ를 능가하지 못해도 FP-RIR-free Structural Matching의 독립적 가치를 평가.

### Final Metrics

- Joint Success: R@1m & 30°
- R@1m
- R@0.1m
- Top-K Coverage
- Conditional Top-1
- High Visual Ambiguity Performance
- Visual-success Preservation
- Repair / Harm
- Inference and Preprocessing Cost

---

## 8. Experimental Falsification

다음 설명들을 구분하라.

**H1 — Acoustic provides directional geometry.**

실제 RIR이 방향별 FP 구조와 관련된 정보를 제공한다.

**H2 — The information is only a scene prior.**

Predictor가 실제 Acoustic 방향 정보보다 Room/Scene의 전형적인 구조를 학습한다.

**H3 — Directional geometry improves visual disambiguation.**

추정된 구조가 시각적으로 유사한 후보들 사이의 정답 구별력을 높인다.

**H4 — Structural matching offers additional information.**

기존 Cosine Matching 또는 GEOQ가 활용하지 못한 정보를 제공한다.

**H5 — The representation is too uncertain for metric geometry.**

정밀 Depth는 어려우나 상대적 거리나 개방도는 유용하다.

**H6 — Directional cues are not reliably observable.**

현재 센서 구성 및 RIR 조건에서는 방향별 구조 추정의 물리적 근거가 충분하지 않다.

특정 가설을 입증하기 위한 실험만 수행하지 말 것.

반대 결과가 나올 때 경쟁 설명을 독립적으로 탐색하라.

---

## 9. Autonomous Research Protocol

다음 과정을 자율적으로 실행하라.

Prior Artifact Audit → Directional Observability → Acoustic Geometry Prediction → Ambiguous Hypothesis Discrimination → Structural Matching → Controlled Integration → Falsification.

### Execution Requirements

- Phase A/B/C를 이번 연구의 필수 핵심 단계로 취급한다.
- Phase D/E는 확보된 근거에 따라 수행한다.
- 작은 Pilot에서 유망한 결과가 나타나면 Scene-held-out 평가로 확장한다.
- 실패 시 모델 용량만 무작정 확대하지 않고 정보·데이터·물리적 한계를 분석한다.
- 최초 보고서 작성이나 작은 실험 실패만으로 종료하지 않는다.
- 허용된 자원 범위에서 충분한 판단 근거가 확보될 때까지 후속 실험을 수행한다.
- 기존 Checkpoint와 사용자 작업 파일을 보존한다.
- Track B의 독립 Framework 구현을 변경하지 않는다.

---

## 10. Required Deliverables

최종 보고서에는 반드시 다음 내용을 포함하라.

1. 이전 Stage A 실험의 실제 수행 범위와 근거
2. Acoustic Directional Geometry Label 및 물리적 가정
3. Direction별 Distance / Openness 예측 수치
4. Scene-held-out 결과와 정보 없는 Baseline
5. Rotation / Channel / Donor / Repeatability Controls
6. Visually Ambiguous Pair Discrimination
7. Acoustic Structural Score vs 기존 Cosine Matching
8. Acoustic Geometry가 실제 Localization에 기여한 사례
9. 최종 B2/GEOQ 대비 성능과 비용
10. 지지된 가설, 반증된 가설, 미해결 문제

### Mandatory Figures

- Direction별 GT vs Predicted Distance
- Directional Openness Confusion Matrix
- Scene-wise Directional Prediction Results
- Visually Ambiguous Pose Pair 사례
- Acoustic Structural Matching Score Field
- Matching/Localization 비교 및 Ablation

수치와 이미지가 없는 단순한 "Directional cue is observable" 형태의 보고서로 종료하지 말 것.

구체적인 결과가 부정적이라면 부정적 수치를 그대로 보고한다.

### Documentation & Git

현재 프로젝트 및 전역 AGENTS.md의 공통 문서화 규칙을 따른다.

- Track A Docs 디렉터리에 실험 보고서 저장.
- RESEARCH_INDEX.md 갱신.
- 기존 Non-main Docs Branch에 Commit/Push.
- 사용한 Dataset, 코드 Commit, Model Checkpoint 및 Artifact 경로 기록.
- 최종 보고에 Docs Branch, Commit SHA, 동기화 상태 포함.
- 기존 보고서와 실험 산출물 보존.

## Final Research Goal

**6-channel RIR에서 Visual Floorplan Localization에 유용한 방향별 공간 구조 정보를 실제로 추출할 수 있는지 검증하고, 그 정보를 통해 시각적으로 유사한 Pose Hypotheses를 더 정확하게 구별하는 방법을 발견하라.**

특정 Acoustic Depth Model이나 Cross-attention의 성공을 목표로 하지 않는다.

중요한 것은 **Acoustic이 어떤 방향에서 공간이 열려 있거나 막혀 있는지를 얼마나 신뢰성 있게 알려줄 수 있으며, 그 정보가 기존 Matching 및 최종 Pose Decision에 어떤 추가 가치를 제공하는가**이다.
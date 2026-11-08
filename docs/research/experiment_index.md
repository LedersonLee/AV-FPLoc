# 전체 실험 목록

이 채팅의 초기 fusion 감사부터 HSSD 확대까지13개 연속 연구를 정리했다. 2026-11-06 이전 B2/encoder 연구 및 별도 direct-attention 연구는 이 흐름의 배경 의존성이며 이번 신규 연구로 포함하지 않는다. 각 연구가 이를 재사용한 사실은 원본 보고서에 남아 있다.

| 순서 | 연구 | 현재 읽어야 할 판정 |
|---|---|---|
| 1 | [초기 fusion 감사](studies/2026-11-07-fusion-audit/README.md) | 저비용 gate·yaw 개입에서 강한 기준을 넘는 개선을 확인하지 못했다. |
| 2 | [국소 지지도와 정밀화](studies/2026-11-07-support-refinement/README.md) | Val8에서 Support+refinement가 B2보다 joint +1.587%p였다. 학습형 gate 개선이 아닌 후보 결정 규칙의 효과이며 재사용 개발 결과다. |
| 3 | [Fusion과 posterior 결정 분리](studies/2026-11-07-fusion-vs-posterior/README.md) | 고정 support/refinement 효과와 adaptive weighting을 분리했다. Val11은 당시 decoder 조건부 확인이었지만 visual teacher 미노출 평가는 아니다. |
| 4 | [후보 추출과 utility ranking](studies/2026-11-07-hypothesis-discrimination/README.md) | 같은 Top20 후보에서 utility 차이 회귀가 선택을 개선했다. Val8 +0.895pp, 재사용 Val11 +0.597pp이며 coverage 증가와 구분한다. |
| 5 | [Modality별 coverage와 판별](studies/2026-11-07-modality-coverage/README.md) | 관측 기반 음향 coverage는 정보 없는 NMS보다 높지만 Top-K 포함과 실제 Top1 판별 사이 격차가 크다. |
| 6 | [후보 geometry와 GEOQ](studies/2026-11-07-geometry-discrimination/README.md) | 기존 점수와 후보 형상의 결합이 utility ranking을 개선했다. 새 관측의 physical matching 이득이라는 초기 해석은 대조 실험으로 수정했다. |
| 7 | [음향 단서와 물리적 해석](studies/2026-11-07-acoustic-cues/README.md) | 시간·envelope 및 초기 복소 채널 관계의 판별 정보를 확인했다. 고정 직접 경로·정규화 때문에 절대 음량/직접 도달 시간 해석은 지지되지 않았다. |
| 8 | [반복 음향과 후보 쌍](studies/2026-11-07-pairwise-acoustic/README.md) | 모호한 쌍의 acoustic 정보는 있으나 GEOQ 이후 추가 joint 개선은 미확인이다. C3−GEOQ −0.104pp, CI[−0.334,+0.114]. |
| 9 | [Domain alignment와 discrimination](studies/2026-11-07-alignment-discrimination/README.md) | 양성 거리 감소만으로 판별 개선을 설명할 수 없다. Positive-only alignment는 표현을 수축시켰고 matched downstream은 GEOQ를 넘지 못했다. |
| 10 | [경쟁 후보와 negative sampling](studies/2026-11-07-candidate-sampling/README.md) | 학습 negative와 실제 경쟁 후보의 차이를 확인했지만 candidate-aware sampling이 강한 GEOQ를 개선하지 못했다. Multi-positive의 상대 회복을 최종 개선과 구분한다. |
| 11 | [방향성 정보 교환](studies/2026-11-08-directional-interaction/README.md) | A→V는 일부 visual 판별을 돕지만 같은 정보의 concat/scalar를 안정적으로 넘지 못했다. E0+GEOQ 개발셋 신호는 HSSD 연구의 전체query 후속 검증에서 확정되지 않았다. |
| 12 | [잔여 정보와 공통 공간 표현](studies/2026-11-08-residual-spatial/README.md) | 잔여 feature·angular/Fourier·capacity·내부 검증 대조를 거쳤으나 GEOQ 이후 안정적인 추가 이득은 미확인이다. 좌표 정합 성공은 실제 DoA 판별 증거와 다르다. |
| 13 | [HSSD 준비도·확대·학습 조건 복원](studies/2026-11-08-hssd-scaling/README.md) | 135개 non-Test 장면 감사와478 fits. 후보 geometry의 matched 이득은 유지됐으나 기존 장면별 가중치 복원 후12→24 확대 이득은 불확실했다. |

[현재 결론과 반증 관계](current_findings.md) · [공통 평가 원칙](evaluation_protocol.md)

서로 다른 cohort의 절대 성능을 이 목록으로 순위 매기지 않는다. 최신 HSSD478 fits와 residual1032 fits 등은 내부 fold/refit을 포함하는 계산 횟수이며 독립적인 과학적 증거 수가 아니다.

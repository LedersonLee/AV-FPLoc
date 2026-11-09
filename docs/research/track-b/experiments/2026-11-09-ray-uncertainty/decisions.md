# 사전 결정

- U2는 U3 scale의 유효 방향 평균을 broadcast하는 inference-only 대조로 고정한다. 추가 fit/capacity 차이가 없다.
- U4는 V FoV 안에서만 순열하여 학습하지 않은 FoV 밖 scale을 섞는 교란을 피한다. A는16방향 전체 순열이다.
- U1 constant는 Train mean absolute residual로 산정한다. Val 결과로 b/temperature를 조정하지 않는다.
- primary U1–U4는 제공된 Laplace likelihood 식을 그대로 사용한다. U0의 squared error learned scale과 다르다는 교란을 명시한다. clean-only에서 frozen 원본 modality scale을 곱한 사전 지정 secondary 진단을 함께 보고하며 좋은 쪽을 선택하지 않는다.
- log(2b) 항은 후보 weight가 방향별로 달라지면 후보별 weighted mean도 달라진다. 후보 공통항으로 항상 제거하지 않는다. 일정 weight와 가변 weight 모두 분석 테스트한다.
- 원래 F1의 target-hit 처리 및 candidate-hit soft weighting을 유지한다. physical wall confidence를 새로 만들어내지 않는다.
- 다음 연구는 본 연구 결과 후 하나를 권장한다. 현재 router/OOF uncertainty 전체실행은 하지 않는다.


## 최종 결정 (2026-11-10)

U3를 기존 F1의 대체 기본값으로 채택하지 않는다. 방향별 head를 복잡하게 확장하거나 Val 결과로 temperature/b를 재조정하지 않는다. U0를 보존한다. 결과는 pose 개선 실패이며, 작은 ray NLL 개선을 localization 성공으로 승격하지 않는다.

가장 구별력 있는 다음 Study는 **posterior mixture 대 고정 product 비교**다. 정상·visual failure·audio failure에서 단일 모달리티의 과도한 제약이 상대적으로 정상인 모달리티를 끌어내리는지를 분리한다. frozen experts/full bank와 학습 예산을 유지하고, Train-only inner validation에서 미리 정한 temperature/mixture 정책을 결정한다. 단순 fixed mixture, 동일 scale의 product, 가능한 관측 기반 conditional mixture를 같은 후보/동일 조건으로 비교하되, clean 손실과 visual/audio 단독 대비 negative transfer를 함께 본다. U3 ray NLL이 좋아도 product의 pose 실패가 지속되지만 mixture가 완화한다면 의사결정 결합 구조 가설을 지지한다. 반대로 두 결합법 모두 실패하면 관측 표현·FP ray supervision 병목을 우선한다. GT oracle는 진단 상한으로만 쓰며 inference router에 넣지 않는다.

이 후속 Study는 제안만 했으며 이번 범위에서 실행하지 않았다. Train-only OOF uncertainty와 unseen-scene 검증은 현재 방향별 이득의 근거가 약하므로 즉시 대규모 실행하지 않는다. 독립 평가로 일반화 주장을 하려면 향후 선택 절차와 upstream exposure에서 분리된 데이터가 필요하다.

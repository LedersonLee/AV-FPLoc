# 사전 결정

- U2는 U3 scale의 유효 방향 평균을 broadcast하는 inference-only 대조로 고정한다. 추가 fit/capacity 차이가 없다.
- U4는 V FoV 안에서만 순열하여 학습하지 않은 FoV 밖 scale을 섞는 교란을 피한다. A는16방향 전체 순열이다.
- U1 constant는 Train mean absolute residual로 산정한다. Val 결과로 b/temperature를 조정하지 않는다.
- primary U1–U4는 제공된 Laplace likelihood 식을 그대로 사용한다. U0의 squared error learned scale과 다르다는 교란을 명시한다. clean-only에서 frozen 원본 modality scale을 곱한 사전 지정 secondary 진단을 함께 보고하며 좋은 쪽을 선택하지 않는다.
- log(2b) 항은 후보 weight가 방향별로 달라지면 후보별 weighted mean도 달라진다. 후보 공통항으로 항상 제거하지 않는다. 일정 weight와 가변 weight 모두 분석 테스트한다.
- 원래 F1의 target-hit 처리 및 candidate-hit soft weighting을 유지한다. physical wall confidence를 새로 만들어내지 않는다.
- 다음 연구는 본 연구 결과 후 하나를 권장한다. 현재 router/OOF uncertainty 전체실행은 하지 않는다.

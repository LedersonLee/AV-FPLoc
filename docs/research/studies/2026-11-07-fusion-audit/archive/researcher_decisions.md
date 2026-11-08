> 역사적 연구 기록. 원본: `astra_fusion_20261107_0200/researcher_decisions.md`. 원본 SHA256은 [산출물 목록](../artifacts.json)에 있습니다.
> 당시의 완료·재개 지시는 현재 실행 요청이 아닙니다. 최신 판정은 [현재 결론](../../../current_findings.md)을 먼저 확인하세요.

# 연구자 판단 자료

- **현재 고정 w=.1을 교체할 근거가 없다.** 네 개의 사전 고정 pilot을 완료했으며3개 후보 모두3scene joint에서 퇴행했다. 원래 코드/체크포인트는 보존했다. 예산4/4 pilot 사용, confirmation0/1; [선택 결과](../artifacts.json) (외부 산출물: `selection.json`).
- **확립한 관찰:** 동결된 현재 입력에는 query 대응 음향 신호가 있으며, 단독 MAP 성공 여부보다 융합에 유용한 정보가 더 넓다. 셔플 개입에서 큰 성능 하락을 측정했다. [진단](failure_diagnosis.md), [측정](../evidence/analysis.json).
- **확립하지 못한 결론:** 새 adaptive fusion 개선, 독립 scene/test 일반화, 가구 원인, 원시 RIR 표현의 우월성, attention의 인과적 이득. 학습 게이트의 순1개 개선은 충분한 근거가 아니다.
- **우선 다음 연구 판단:** 현재 분포의 Train OOF에서 '음향 단독 정확도' 대신 후보별 visual margin을 음향이 어떻게 수정하는지 학습하는 최소 gate를 설계할지 결정한다. 학습 기준/동일-budget fixed/learned/shuffle 대조를 먼저 고정해야 한다. 이번에는 새 gate를 학습하지 않았다.
- **두 번째 판단:** confidence calibration과 yaw/XY scaling을 분리하는 작은 controlled probe에 추가 예산을 줄지 결정한다. 이번 결과는 broad temperature sweep을 정당화하지 않는다.
- **별도 자원 판단:** furnished/unfurnished 동일-pose counterfactual 생성, raw/early-RIR 표현, 새로운 cross-attention 재학습은 별도 설계/예산 승인 대상이다. 기존 실제 attention과 단순 projection을 혼동하지 않는다.
- **최종 test:** 이번에는 유망 후보가 없고 독립 test를 권고할 단계가 아니다. 앞으로 후보가 생기면 과거에 재사용하지 않은 scene 평가를 설계하고 조건을 freeze한 후 명시적 승인으로1회 실행한다. 이번에 Test arrays 및 추가Val5를 열지 않았다.
- **산출물 상태:** 필수6종 문서/로그, query-level 결과, source/input/checkpoint 해시, 독립 metric 검증, 도표 완료. 상세 표와 재현 안내는 [결과 요약](../results.md).

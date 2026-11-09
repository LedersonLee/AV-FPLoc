# Study 30 — 방향별 관측 불확실성 진단

**상태: preflight 진행 중. 입력 무결성 감사 통과, U0 전체 지표 재현 통과. 불확실성 head의 학습·성능 결과는 아직 없다.** 기존 Study21/24/28의 음성 결과를 보존하고, 강한 Study27 F1을 독립 경로로 유지한다.

질문은 frozen ray mean에서 방향별 scale이 constant-L1/global scale보다 실제 pose matching과 열화 신뢰성에 유용한가이다. L2→L1 변경, 상대 모달리티 scale, FP supervision 근사, in-sample residual optimism을 경쟁 설명으로 구분한다. AVLoc의 구조 복제/새 attention/router/B2 후보 사용은 범위 밖이다.

- [사전 감사](preflight_report.md), [고정 실행 계획](protocol.json), [입력 검증](input_validation.json)
- [결정 기록](decisions.md), [결과 상태](results.md), [재현](reproduction.md)

Train112/reused Val19×64, seeds42/43/44, .25m/36yaw,16rays. Frozen local means/FP maps/scale/upstream. Train에서 작은 scale head만1000updates 학습할 계획이다. 기존 모델/checkpoint를 수정하지 않는다. protocol의 모든 U0–U4×V/A/fusion 및 declared shifts를 보고하며, 결과가 실패해도 조건을 선택적으로 삭제하지 않는다. Test를 사용하지 않는다.

[구현 검증](implementation_validation.md): scale/Laplace 분석 검사5개 및 전체49개 통과, Train target 범위 확인. 실제 head 학습 smoke는 다음 단계다.

[실제 학습 진행](training_progress.md):1seed/4query/2step smoke에서 원본 expert 보존과 head 업데이트 통과;3seed 본 학습 시작.

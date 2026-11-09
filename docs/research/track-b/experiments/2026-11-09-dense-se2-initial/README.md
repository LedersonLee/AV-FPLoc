# Track B — independent dense SE(2) initial investigation

Study ID: `track-b-dense-se2-initial-20261109` · 날짜:2026-11-09 KST · 증거: **Exploratory**.

**초기 데이터 감사와 실제 학습을 수행했다. 전체 연구 프로토콜의 성공/완료를 선언하지 않는다.**
독립 구현은 `track_b/`에 있으며 기존 B2 후보·FP-RIR reference·Track A checkpoint를 입력으로 사용하지 않는다. 사용자 기존 코드 변경은 보존했다. 실행 도중 별도 기존 문서 정리 Commit5449980이 확인되었으며 이번 작업의 변경으로 취급하지 않았다.

- [초기42fits 최종 판정과 다음 반증](final_assessment.md)
- [환경·데이터 감사](audit_report.md)
- [실험 결과·실패·해석](results.md)
- [설계 결정](decisions.md)
- [재현 명령](reproduction.md)
- [실제 코드/입력 해시](provenance.json)
- [실험 레지스트리](experiment_registry.csv)
- [필수 결과 필드](results_master.csv)
- [사용자가 제공한 설계 명세 원문](supplied_protocol.md)
- [공통 평가 원칙](../../../shared/evaluation-protocol/README.md)

원시 manifest, 관측, per-query 예측, checkpoint, 원시 로그는 Git에 넣지 않았다. 실행 산출물은 로컬 `track1/experiments/track_b_20261109/`에 보존한다. Code SHA는 provenance/registry에 기록하고 Docs Commit SHA는 Commit 이후 실행 보고에만 기록한다.

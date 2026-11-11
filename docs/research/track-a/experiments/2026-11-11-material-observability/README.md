# Experiment4A — Material-dependent Acoustic Observability

**Outcome B — Material-dependent but Ambiguous.** 재질 계수의 영향은 관측되지만, 목표 벽의 증거와 다른 반사를 안정적으로 구분하지 못했다.

- [연구 보고서](RESEARCH_REPORT.md)
- [Interactive Acoustic Material Explorer](interactive.html)
- [물리·재질 감사](PHYSICAL_MATERIAL_AUDIT.md)
- [작동 원리](METHOD_AT_A_GLANCE.md)
- [프로토콜](PROTOCOL.md)
- [재현·비용](REPRODUCIBILITY.md)
- [실험 검산](verification.json)
- [화면·원시 데이터 일치 검산](explorer_verification.json)

기본 위치는 `corridor_near`, 예외는 `room_far`와 `single6`다. 실제 저장된 조건만 선택하며 A/B를 공통 시간·진폭 축으로 비교한다. 원시 RIR은 로컬 실험 폴더에 보존했고, Git에는 집계·설정/해시·화면용 요약 데이터만 저장한다.

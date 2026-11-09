# HSSD 준비도·확대·학습 조건 복원

- 연구 ID: `hssd_scaling_20261108_1332`
- 판정: 135개 non-Test 장면 감사와478 fits. 후보 geometry의 matched 이득은 유지됐으나 기존 장면별 가중치 복원 후12→24 확대 이득은 불확실했다.
- 평가 범위: 연구별 query/scene/aggregation을 확인해야 한다. Val8/Val11 재사용 및 visual teacher 노출을 독립 평가로 해석하지 않는다. Test 관측·결과는 연구에 사용하지 않았다.

## 읽기 순서

1. [상세 결과와 해석](results.md)
2. [프로토콜과 변경 이력](protocol.md)
3. [재현과 원본 근거](reproduction.md)
4. [수치 근거](evidence/FINAL_FINDINGS.json) · [원본 위치·해시](artifacts.json)

## 연구 흐름

[이전 연구](../2026-11-08-residual-spatial/README.md) → 이 연구 → 최신 연구

[전체 연구 목록](../../../RESEARCH_INDEX.md) · [현재 결론과 후속 반증](../../../current_findings.md)

이 문서는 기존 결과의 Git 아카이브이며 새 실험이나 독립 재검증을 뜻하지 않는다. 당시 보고서의 해석과 현재 누적 결론을 구분한다.

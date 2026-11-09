# Modality별 coverage와 판별

- 연구 ID: `modality_coverage_20261107_1700`
- 판정: 관측 기반 음향 coverage는 정보 없는 NMS보다 높지만 Top-K 포함과 실제 Top1 판별 사이 격차가 크다.
- 평가 범위: 연구별 query/scene/aggregation을 확인해야 한다. Val8/Val11 재사용 및 visual teacher 노출을 독립 평가로 해석하지 않는다. Test 관측·결과는 연구에 사용하지 않았다.

## 읽기 순서

1. [상세 결과와 해석](results.md)
2. [프로토콜과 변경 이력](protocol.md)
3. [재현과 원본 근거](reproduction.md)
4. [수치 근거](evidence/summary.json) · [원본 위치·해시](artifacts.json)

## 연구 흐름

[이전 연구](../2026-11-07-hypothesis-discrimination/README.md) → 이 연구 → [다음 연구](../2026-11-07-geometry-discrimination/README.md)

[전체 연구 목록](../../../RESEARCH_INDEX.md) · [현재 결론과 후속 반증](../../../current_findings.md)

이 문서는 기존 결과의 Git 아카이브이며 새 실험이나 독립 재검증을 뜻하지 않는다. 당시 보고서의 해석과 현재 누적 결론을 구분한다.

# Directional Evidence / Tri-branch Pilot — Track A

2026-11-09 · Exploratory · 기존 B2 Top20 및 FP-RIR reference를 사용하므로 Track A. 독립 FP-RIR-free Track B 구현이 아니다.

[결과](RESULTS.md) · [Stage A](A_FEASIBILITY.md) · [실험 행렬](B_EXPERIMENT_MATRIX.md) · [한계와 반증](MECHANISM_AND_FALSIFICATION.md) · [재현](REPRODUCE.md) · [산출물 출처](provenance.json) · [통합 Index](../../../RESEARCH_INDEX.md)

음향 coarse near/far 정확도77.67% 대 prior62.00%; GEOQ 이후 추가 localization 이득은 미확인. Attention joint +0.065pp는 불확실하며 fine -1.204pp. F1/F2와 저학습률 후속도 채택 근거가 없어 기존 B2+GEOQ 유지. 기존 문서를 수치 수정 없이 복사했고 새 실험은 수행하지 않았다.

실행 스크립트는 외부 작업공간에 있으며 정확한 실행 Git SHA는 미기록이다. base19c1bd0와 개별 script SHA256은 provenance.json에 있다. Docs Commit을 실험 코드 SHA로 대체하지 않는다. 원본 문서의 실행 명령은 역사적 기록이며 현재 실행 승인이 아니다. 전체 로그/Checkpoint/원본 MANIFEST는 로컬에 보존한다.

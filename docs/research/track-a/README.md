# Track A — FP-RIR-based Acoustic–Visual Floorplan Localization

FP-RIR Reference 기반 기존 framework를 개선하고 Acoustic Structural Evidence 및 Geometry-conditioned Hypothesis Discrimination 원리를 규명한다.

범위: B2/GEOQ, FP-RIR acoustic matching, visual/acoustic score·adaptive fusion, 방향별 거리·개방도·반사 단서, Support-NMS/Top-K 후보 추출, geometry-conditioned 판별, 기존 framework cross-modal interaction/ablation, domain alignment/contrastive learning, HSSD scaling, 일반화·robustness 비교.

[통합 Index](../RESEARCH_INDEX.md)의 연구1–14가 해당된다. 각 보고서는 `experiments/<YYYY-MM-DD-study-id>/`에 있다. 기존 B2 후보 및 FP-RIR reference를 사용하는 Tri-branch Pilot은 이 Track이며 독립 Track B의 증거로 취급하지 않는다. [이관 근거와 전체 목록](../management/migration_report.md)을 참고한다.

새 보고서는 공통 [템플릿](../templates/study.md)과 [평가 원칙](../shared/evaluation-protocol/README.md)을 사용한다. 비교 실험은 Primary Track에만 저장하고 [Track B](../track-b/README.md)에서는 참조한다. 연구별 과거 명령은 현재 실행 승인이 아니다.

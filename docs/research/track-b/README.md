# Track B — Independent FP-RIR-free Tri-branch Localization

RGB / 6-channel RIR / known2D floorplan → 독립 trainable encoders → multimodal interaction → full SE(2) score field → 자체 Top20 hypothesis extraction → pose discrimination → final pose.

FP-RIR Reference Bank 없이 전체 pose field를 직접 생성하는 독립적인 end-to-end framework다. V↔A, V↔FP, A↔FP 정보 교환과 FP-conditioned joint learning, 자체 full-bank scoring/후보 추출/판별을 연구한다. FP-RIR 사전 생성·저장 비용 제거를 accuracy/generalization/compute와 함께 평가한다.

기존 B2 Top20을 입력받는 ranker로 대체하지 않는다. 추론에 FP-RIR bank가 필요하면 Track B 요건을 충족하지 않는다. B2/GEOQ는 비교 baseline으로만 쓰며 핵심 pipeline에 종속시키지 않는다. GEOQ 원리 재사용 시 새 입력/후보로 재학습하고 출처·의존성을 기록한다. Track A 코드를 참고하더라도 독립적인 구현/평가 근거가 필요하다.

**현재 이관된 Track B 실험은 없다.** 기존 Tri-branch Pilot은 [Track A](../track-a/experiments/2026-11-09-directional-tribranch/README.md)다. 새 실험은 `experiments/<YYYY-MM-DD-study-id>/`에 보관하고 [단일 Index](../RESEARCH_INDEX.md)에 등록한다. 비교 연구는 Primary Track 보고서를 참조하여 중복 생성하지 않는다. [공통 평가](../shared/evaluation-protocol/README.md)와 [템플릿](../templates/study.md)을 따른다.

# AV-FPLoc 프로젝트 연구 지침

공통 연구 보고·증거·Git 규칙은 `~/.codex/AGENTS.md`를 따른다. 이 파일은 AV-FPLoc 코드 및 외부 `track1/experiments/`의 관련 연구에 대한 프로젝트 설정이다. 공통 지침 파일이 없는 환경에서는 연구 Git 동기화 전에 해당 설정이 필요함을 보고한다.

## 연구 목표와 두 독립 Track

**Track A — FP-RIR-based Acoustic–Visual Floorplan Localization**
기존 FP-RIR reference 기반 framework를 개선하고 Acoustic Structural Evidence 및 Geometry-conditioned Hypothesis Discrimination 원리를 규명한다. B2/GEOQ, FP-RIR acoustic matching, score/adaptive fusion, 방향별 거리·개방도·반사 단서, Support-NMS/Top-K 추출, 후보 geometry 판별, 기존 framework의 cross-modal interaction/ablation, 일반화·robustness·baseline 비교를 포함한다. 기존 B2 후보 기반 cross-attention/tri-branch는 Track A이며 Stage A/B 명칭과 Track을 혼동하지 않는다.

**Track B — Independent FP-RIR-free Tri-branch Localization**
RGB / 6-channel RIR / known2D floorplan → 독립적인 trainable encoders → multimodal interaction → full SE(2) pose score field → 자체 Top20 추출 → pose discrimination → final pose의 독립적인 end-to-end framework를 개발한다. V↔A/V↔FP/A↔FP 정보 교환, FP-conditioned joint representation, FP-RIR-free full-bank scoring과 자체 후보 추출/판별, FP-RIR 사전 생성·저장 비용 제거의 실용성을 검증한다.

Track B는 **추론에서 FP-RIR Reference Bank에 의존하지 않는다**. 기존 B2 Top20을 입력받는 ranker로 대체하지 않는다. B2/GEOQ는 비교 baseline이지 핵심 pipeline의 의존성이 아니다. GEOQ 원리를 사용하면 새 입력/후보에 맞게 재학습하고 의존성을 명시한다. Track A 코드/결과는 참고할 수 있지만 독립 구현·평가 근거를 유지한다. Accuracy/generalization/compute를 함께 비교한다.

Track 간 비교는 목적과 사용 framework에 따라 Primary Track을 정하고 보고서는 한 곳에만 저장하며 다른 Track에서 참조한다. 별도 Research Review 채팅에서 IJCAI2027 통합 또는 후속 독립 논문 가능성을 평가한다. 이 지침은 해당 채팅에 자동 메시지를 보내는 승인이 아니다.

기존 보고서14개는 구현/후보/reference 의존성을 확인한 뒤 사용자 요청으로 Track A에 보관했다. 새 실험은 위 정의로 분류한다. 특정 architecture 성공을 전제하지 않는다. 단계별 성공·반증·종료 조건은 승인된 protocol에서 정한다. 설정 작업은 실험 실행 승인이 아니다. Checkpoint/원본 결과/사용자 변경을 보존하고 실험별 새 폴더를 사용한다. 승인된 로컬 자원 범위를 따르며 유료 자원·외부 데이터 업로드는 별도 사용자 지시에 따른다.

## Dataset / Evaluation Protocol

- HSSD 기본 자산: `/file2/jeongeon/AV-FPLoc/datasets/HSSD`, `/ssd1/AV-FPLoc/datasets/HSSD`. 다른 데이터셋은 해당 실험에서 출처·사용 권한·split을 별도 명시한다.
- Test 관측/라벨/결과를 방법·모델·epoch·threshold 선택에 사용하지 않는다. Test 식별 메타데이터로 누출/동일 source-family 제외를 확인하는 것과 평가 사용을 구분한다.
- Scene-held-out 및 가능한 scene-grouped inner selection, 여러 seed, paired scene bootstrap을 사용한다. seed/pair/render를 독립 scene으로 세지 않는다.
- 기존 HSSD visual teacher의 Train116/Val19 노출과 acoustic Train12/Val3 선택, 연구 과정의 평가 재사용을 기록한다. 기존 Val 및 head-held 결과를 pristine/full-pipeline independent로 부르지 않는다.
- 동일 후보 비교를 먼저 수행한다. GT는 학습/진단에만 사용하고 inference feature·proposal filtering에는 넣지 않는다. Oracle coverage와 실제 Top1, 표현 probe와 조건부 localization 이득을 분리한다.
- Joint position≤1m & circular yaw≤30°, R@1m, fine position≤0.1m를 함께 보고한다. fine 및 pure-visual 성능 저하를 숨기지 않는다. 다른 metric 정의는 명시한다.
- 기존 상세 원칙: `docs/research/shared/evaluation-protocol/README.md`. 새 실험의 설정/예외/추가 대조는 실험 protocol에 기록하고 공통 원칙을 중복 복사하지 않는다.

## 주요 Baselines

UnLoc visual, 기존 acoustic encoder, standardized fixed fusion(V .85/A .15), 기존 B2, Support-NMS/local refinement, **B2+GEOQ**를 cohort에 맞게 비교한다. GEOQ의156차원 S/G/Q 및 상대 특징 기반 utility gain은 calibrated posterior가 아니다. 예전 수치를 다른 cohort에 그대로 재사용하지 말고 checkpoint·코드·후보 IDs·지표를 확인한다. 구현/출처 진입점은 `docs/research/current_findings.md`와 각 study의 reproduction.md다.

## Docs Git 설정

- Code/Docs 저장소: `/home/jeongeonlee/track1/AV-FPLoc` (다른 checkout에서는 이 AGENTS.md가 있는 Git root).
- 승인 remote: `origin` = `https://github.com/LedersonLee/AV-FPLoc.git`.
- 기존 non-main 작업 브랜치: `refactor/hssd-two-branch-fusion`.
- Docs 루트: `docs/research/`.
- Canonical 통합 Index: `docs/research/RESEARCH_INDEX.md`. 기존 experiment_index.md를 rename했으며 중복 파일을 만들지 않는다.
- Track A: `docs/research/track-a/experiments/<YYYY-MM-DD-study-id>/`.
- Track B: `docs/research/track-b/experiments/<YYYY-MM-DD-study-id>/`.
- 공통 Dataset/Evaluation: `docs/research/shared/evaluation-protocol/README.md`. 공통 내용은 복제하지 않고 참조한다.
- 기존 studies/13개는 Git rename으로 Track A에 이관했고 로컬 B2 후보 기반 Pilot1개를 추가 보관했다. 이동 목록/이전 해시는 `docs/research/management/migration_inventory.json`, 분류·검증 근거는 같은 폴더의 migration_report.md에 있다. 기존 실험 결과를 변경하지 않는다.
- 각 Track 안에서 주제별 하위 폴더가 필요하면 fusion-optimization/hypothesis-discrimination/acoustic-structural-evidence/domain-alignment/cross-modal-interaction/hssd-scaling 등으로 묶을 수 있다. 현재는 기존 고유 study ID/날짜를 유지하여 경로와 출처를 식별한다.
- 연구 양식: `docs/research/templates/study.md`. 성공·실패 arm은 해당 보고서에 누적하고 새로운 후속 연구는 고유 ID와 실제 실행 코드 SHA를 Index에 기록한다. 분류가 불명확하면 임의 Track B 승격 없이 의존성을 확인한다.

## 문서 검증 / 동기화

문서만 수정했다면 실험/학습을 재실행하지 않는다. `python3 docs/research/tools/validate_archive.py`로 기존13개 아카이브의 링크/해시를 검증하고, 새 Track 문서의 링크·Index 필수 필드·코드 provenance·민감/대용량 포함 여부도 확인한다. 기존 exporter는 초기13개 snapshot 전용이며 Track 이관 후 재실행을 차단했다. 일상 갱신에 사용하지 않는다.

이번 작업 파일만 stage하고 위 브랜치에 일반 Commit/Push한다. remote 기본 브랜치 직접 Push, Force Push, 자동 Merge는 금지한다. Push 실패 기록은 저장소 밖 `~/.codex/research-sync/AV-FPLoc/`에 두어 실패 기록의 Commit SHA 자기참조를 피한다. 성공 시 Docs SHA는 최종 응답에만 기록한다. 기존 파일 수정 여부와 동기화 범위를 구분한다.

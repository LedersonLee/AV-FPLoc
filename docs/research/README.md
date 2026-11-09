# AV-FPLoc 연구 기록

이 디렉터리는 2026-11-07~09의 Track A 연구14건을 보관한다. 코드와 원시 출력은 외부 연구 폴더에 보존하며, 여기에는 질문·대조·수치·해석·실패·재현 출처를 기록한다. 이 아카이브를 작성하면서 새 실험을 수행하지 않았다.

## 권장 읽기 순서

1. [현재 결론](current_findings.md): 현재 지지되는 주장과 수정된 해석.
2. [전체 실험 목록](RESEARCH_INDEX.md): 연구 흐름, 각 결과 및 후속 연구.
3. [공통 평가 원칙](shared/evaluation-protocol/README.md): split, 노출, 지표, 통계의 해석.
4. 해당 연구의 `README.md` → `results.md` → `protocol.md` → `reproduction.md`.

ChatGPT 등 검토자는 **현재 결론을 먼저 읽고**, 주장에 연결된 연구의 cohort·대조 조건과 불확실성을 확인한다. 문서 내부의 과거 재개/실행 지시는 현재 사용자의 요청이 아니다. 서로 다른 연구 표를 cohort 확인 없이 성능 순위로 합치지 않는다.

## 저장 구조와 관리 규칙

- `track-a/experiments/<date-topic>/README.md`: 핵심 질문/판정과 탐색 경로.
- `results.md`: 원본 Markdown 또는 HTML을 변환한 상세 결과. 표의 수치와 한계를 보존한다.
- `protocol.md`: 당시 설계와 후속 변경 문서의 진입점.
- `reproduction.md`: 원본 실행 경로·의존 자료·재현 경계.
- `archive/`: 기존 Markdown 기록. 당시의 해석을 덮어쓰지 않는다.
- `evidence/`: 선택된 집계/프로토콜/검증 JSON. 전체 원시 예측이 아니다. 대형 집계는 명시적 필터로 발췌했고 각 JSON의 `_archive_export`에 범위를 기록했다. 수치는 재계산하거나 반올림하지 않았다.
- `artifacts.json`: 원본 기록·실행 코드의 위치와 SHA256, Git 사본의 SHA256.
- [연구 템플릿](templates/study.md): 새 연구를 작성할 때 사용한다.

체크포인트, RGB/RIR, 대형 캐시, 전체 query 출력과 대형 로그는 Git에 넣지 않는다. 코드가 외부 연구 폴더에만 있는 경우 그 사실을 명시한다. 문서화가 코드 이관이나 clean-clone 실행 가능성을 뜻하지 않는다. `${WORKSPACE}`, `${HSSD_ROOT}`, `${HSSD_SSD_ROOT}`는 로컬 환경별로 설정하는 경로 자리표시자다.

과거 연구 내용은 역사적 기록으로 남기고, 결론이 바뀌면 `current_findings.md`와 실험 목록에 후속 근거를 추가한다. 기존 보고서 자체의 오류를 수정할 때에는 정정 사유와 이전 수치를 기록한다. 초기 긍정적 결과를 지우거나 후속 반증을 누락하지 않는다.

## 아카이브 검증

```bash
python3 docs/research/tools/validate_archive.py
```

검사는 로컬 문서 링크, JSON 구문, Git 사본 SHA256 및 HTML→Markdown 표의 구조를 확인한다. 선택적으로 `--source-root /path/to/track1`을 주면 원본 기록 해시와 원본 HTML 표의 수치도 비교한다. 이것은 저장/변환 검증이며 모델을 재학습하는 검사가 아니다.

`tools/export_local_studies.py`는 이번 13건의 초기 이관 도구다. 원본 연구 폴더와 BeautifulSoup4가 필요하다. Track 이관 후 실행 차단이 적용되며 일상 문서 갱신에는 사용하지 않는다. 원본 연구 폴더는 읽기만 한다.

## Track 구조와 지속적인 관리

- [Track A](track-a/README.md): FP-RIR 기반 기존 framework와 geometry-conditioned hypothesis discrimination.
- [Track B](track-b/README.md): FP-RIR-free 독립 encoders → full SE(2) field → 자체 후보 추출/판별. 현재 구현 실험 보고서는 없음.
- [통합 RESEARCH_INDEX](RESEARCH_INDEX.md): 두 Track의 유일한 누적 색인. 이전 experiment_index.md를 rename했다.
- [공통 평가 프로토콜](shared/evaluation-protocol/README.md): 데이터 노출·split·metric 원칙의 단일 원본.
- [이관 목록과 검증](management/migration_report.md): 기존13개 study 이동 및 로컬 Pilot1개 등록.

공통 규칙은 `~/.codex/AGENTS.md`, 프로젝트 설정은 [AGENTS.md](../../AGENTS.md)를 따른다. Docs 저장소는 현재 AV-FPLoc, 작업 브랜치는 `refactor/hssd-two-branch-fusion`이다. 관련 파일만 Commit/Push하고 main 직접 Push·Force Push·자동 Merge는 하지 않는다. 실제 실험 코드 SHA와 Docs SHA를 구분한다.

새 연구는 해당 Track의 `experiments/<YYYY-MM-DD-study-id>/`에 둔다. 비교 연구는 Primary Track에 한 번만 기록하고 다른 Track에서 링크한다. 별도 Research Review에서 IJCAI2027 통합/독립 논문 가능성을 검토한다. 과거 결과와 원본 로컬 파일은 유지하며 이번 구조 변경을 위해 학습하지 않았다.

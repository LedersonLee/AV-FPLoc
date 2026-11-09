# Research instructions / Track migration — 2026-11-09

지침 설정 및 문서 이관 작업이다. 학습·평가 실험을 새로 실행하지 않았다. 최신 사용자 지시가 이전의 “문서를 이동하지 않음”을 대체하여 기존 Track A 보고서를 실제 이관했다. 전역 Git 보고 규칙은 그대로 유지한다.

## 조사와 분류

Git 관리 docs 전체 파일을 이관 전에 조사했다. [전체 파일 목록과 이전/이후 경로·해시](migration_inventory.json)에 모든 조사 항목을 기록했다. 기존 research/studies/의13개 연구는 결과·provenance의 B2/GEOQ/FP reference/기존 후보 의존성을 확인하여 Track A로 분류했다. 각 연구의 실제 근거 문장을 inventory에 저장했다. 연구별 topic 폴더를 새로 합치지 않고 기존 날짜·고유 ID를 유지했다.

기타 docs/architecture, data_contracts, f3loc_baseline/code_inventory, hssd_preprocessing/reproduction 및 협업/계획 문서는 프로젝트 배경 문서로 원래 경로에 유지했다. 독립 Track B 실험 증거로 분류하지 않았다. Track B는 새 독립 FP-RIR-free pipeline 정의와 빈 experiments/만 설정했다.

## 연구별 경로 변경

|연구 ID|이전 경로|새 보고서|
|---|---|---|
|2026-11-07-acoustic-cues|`docs/research/studies/2026-11-07-acoustic-cues`|[보고서](../track-a/experiments/2026-11-07-acoustic-cues/results.md)|
|2026-11-07-alignment-discrimination|`docs/research/studies/2026-11-07-alignment-discrimination`|[보고서](../track-a/experiments/2026-11-07-alignment-discrimination/results.md)|
|2026-11-07-candidate-sampling|`docs/research/studies/2026-11-07-candidate-sampling`|[보고서](../track-a/experiments/2026-11-07-candidate-sampling/results.md)|
|2026-11-07-fusion-audit|`docs/research/studies/2026-11-07-fusion-audit`|[보고서](../track-a/experiments/2026-11-07-fusion-audit/results.md)|
|2026-11-07-fusion-vs-posterior|`docs/research/studies/2026-11-07-fusion-vs-posterior`|[보고서](../track-a/experiments/2026-11-07-fusion-vs-posterior/results.md)|
|2026-11-07-geometry-discrimination|`docs/research/studies/2026-11-07-geometry-discrimination`|[보고서](../track-a/experiments/2026-11-07-geometry-discrimination/results.md)|
|2026-11-07-hypothesis-discrimination|`docs/research/studies/2026-11-07-hypothesis-discrimination`|[보고서](../track-a/experiments/2026-11-07-hypothesis-discrimination/results.md)|
|2026-11-07-modality-coverage|`docs/research/studies/2026-11-07-modality-coverage`|[보고서](../track-a/experiments/2026-11-07-modality-coverage/results.md)|
|2026-11-07-pairwise-acoustic|`docs/research/studies/2026-11-07-pairwise-acoustic`|[보고서](../track-a/experiments/2026-11-07-pairwise-acoustic/results.md)|
|2026-11-07-support-refinement|`docs/research/studies/2026-11-07-support-refinement`|[보고서](../track-a/experiments/2026-11-07-support-refinement/results.md)|
|2026-11-08-directional-interaction|`docs/research/studies/2026-11-08-directional-interaction`|[보고서](../track-a/experiments/2026-11-08-directional-interaction/results.md)|
|2026-11-08-hssd-scaling|`docs/research/studies/2026-11-08-hssd-scaling`|[보고서](../track-a/experiments/2026-11-08-hssd-scaling/results.md)|
|2026-11-08-residual-spatial|`docs/research/studies/2026-11-08-residual-spatial`|[보고서](../track-a/experiments/2026-11-08-residual-spatial/results.md)|
|2026-11-09-directional-tribranch|`${WORKSPACE}/experiments/directional_tribranch_20261109 (로컬 원본 유지)`|[보고서](../track-a/experiments/2026-11-09-directional-tribranch/RESULTS.md)|

기존 Git 관리 study 디렉터리는 git mv로 이동했다. 이전 실험 결과·과거 실패/정정 내용은 보존하고 상대 링크만 새 경로에 맞게 계산했다. Link 변경이 발생한 Git 사본 해시는 artifacts.json에 갱신하되 이전 사본 해시와 원본 소스 해시를 유지했다. Git 이력은 기존 commit에 그대로 존재하며 rename 추적으로 확인 가능하다.

로컬 Tri-branch Pilot은11개 Markdown,4개 Figure(PNG/PDF),3개 작은 설정/검증 JSON을 복사했다. 보고서/그림18개는 원본과 바이트 단위 일치한다. 원본 파일과 실행 코드·가중치·캐시·전체 raw outputs/logs는 로컬에 남겼다. Git 관리 기존13개 아카이브에 Figure 바이너리는 없었고 외부 Figure 출처 안내는 유지했다. 새 Pilot provenance.json은 문서와 외부 script의 해시를 제공한다. 실험 코드는 uncommitted 외부 파일이므로 base SHA를 실행 SHA로 오인하지 않도록 했다.

## 공통 경로와 지침

- `docs/research/experiment_index.md` → `docs/research/RESEARCH_INDEX.md`: 단일 통합 색인,14개 연구의 Track/날짜/핵심 발견/증거 수준/보고서/실험 코드 provenance.
- `docs/research/evaluation_protocol.md` → `docs/research/shared/evaluation-protocol/README.md`: 내용 보존, 중복 없음.
- `~/.codex/AGENTS.md`: 전역 공통 결과 보고·증거 구분·non-main Commit/Push·보존·비밀 제외·실패 복구 규칙. 개인 로컬 설정이며 이 Docs Commit에는 포함하지 않는다.
- `/home/jeongeonlee/track1/AGENTS.md`: 작업공간 루트/외부 experiments에서 AV-FPLoc 프로젝트 지침을 먼저 읽도록 하는 로컬 라우팅. 이 Docs Commit에는 포함하지 않는다.
- 저장소 `AGENTS.md`: Track A/B의 독립 목표·제약, 데이터/평가, baselines, Docs remote/branch, 단일 Index와 Track 저장 구조.
- 기존 template/README/검증 도구도 새 경로에 맞게 갱신. 구형 초기 exporter는 중복 옛 구조/Index 재생성을 방지하도록 차단했다.

전역/프로젝트 기존 AGENTS.md 및 우선하는 override는 조사 시 없었다. 기존 연구 운영 규칙은 README/template/evaluation 문서에서 유지·연결했다. CODEX_HOME은 `/home/jeongeonlee/.codex`, Codex CLI0.163.0. 현재 대화에서 새 전역 AGENTS.md가 실제 전달된 것도 확인했다. 프로젝트 파일의 위치·override 부재·크기·라우팅은 정적으로 검증했다. 새 독립 Codex 세션을 실행해 프로젝트 지침의 자동 주입을 확인한 것은 아니다.

공식 지침에 따르면 instruction chain은 run/session 시작 시 구성된다. 이후 세션은 AV-FPLoc root에서 시작하면 전역+프로젝트 지침을, track1에서 시작하면 전역+workspace 라우팅을 사용한다. 현재 세션이 모든 변경을 자동 재로딩한다고 가정하지 않는다. [OpenAI 공식 AGENTS.md 문서](https://learn.chatgpt.com/docs/agent-configuration/agents-md).

## 검증과 동기화

[검증 기록](configuration_validation.json)에 링크, 기존 source/copy/table, 새 Pilot 사본 해시, 원본 사용자 diff, 경로 중복과 instruction 크기 검사를 기록한다. Git에는 이번 설정/문서만 명시적으로 stage한다. 기존 non-main `refactor/hssd-two-branch-fusion` → 동일 origin branch에 일반 Push한다. Main 직접 Push/Force Push/Merge는 하지 않는다. Docs Commit SHA와 실제 동기화 결과는 Commit 이후 최종 응답과 저장소 밖 동기화 receipt에 기록한다.

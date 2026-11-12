# 전체 실험 목록

이 채팅의 초기 fusion 감사부터 HSSD 확대까지13개 연속 연구를 정리했다. 2026-11-06 이전 B2/encoder 연구 및 별도 direct-attention 연구는 이 흐름의 배경 의존성이며 이번 신규 연구로 포함하지 않는다. 각 연구가 이를 재사용한 사실은 원본 보고서에 남아 있다.

| 순서 | 연구 | 현재 읽어야 할 판정 | 날짜 | Track | 증거 수준 | 실험 코드 Commit SHA |
|---|---|---|---|---|---|---|
| 1 | [초기 fusion 감사](track-a/experiments/2026-11-07-fusion-audit/README.md) | 저비용 gate·yaw 개입에서 강한 기준을 넘는 개선을 확인하지 못했다. | 2026-11-07 | Track A | Exploratory | UNRECORDED — 해당 study의 artifacts.json 소스 해시 참조 |
| 2 | [국소 지지도와 정밀화](track-a/experiments/2026-11-07-support-refinement/README.md) | Val8에서 Support+refinement가 B2보다 joint +1.587%p였다. 학습형 gate 개선이 아닌 후보 결정 규칙의 효과이며 재사용 개발 결과다. | 2026-11-07 | Track A | Exploratory | UNRECORDED — 해당 study의 artifacts.json 소스 해시 참조 |
| 3 | [Fusion과 posterior 결정 분리](track-a/experiments/2026-11-07-fusion-vs-posterior/README.md) | 고정 support/refinement 효과와 adaptive weighting을 분리했다. Val11은 당시 decoder 조건부 확인이었지만 visual teacher 미노출 평가는 아니다. | 2026-11-07 | Track A | Exploratory | UNRECORDED — 해당 study의 artifacts.json 소스 해시 참조 |
| 4 | [후보 추출과 utility ranking](track-a/experiments/2026-11-07-hypothesis-discrimination/README.md) | 같은 Top20 후보에서 utility 차이 회귀가 선택을 개선했다. Val8 +0.895pp, 재사용 Val11 +0.597pp이며 coverage 증가와 구분한다. | 2026-11-07 | Track A | Exploratory | UNRECORDED — 해당 study의 artifacts.json 소스 해시 참조 |
| 5 | [Modality별 coverage와 판별](track-a/experiments/2026-11-07-modality-coverage/README.md) | 관측 기반 음향 coverage는 정보 없는 NMS보다 높지만 Top-K 포함과 실제 Top1 판별 사이 격차가 크다. | 2026-11-07 | Track A | Exploratory | UNRECORDED — 해당 study의 artifacts.json 소스 해시 참조 |
| 6 | [후보 geometry와 GEOQ](track-a/experiments/2026-11-07-geometry-discrimination/README.md) | 기존 점수와 후보 형상의 결합이 utility ranking을 개선했다. 새 관측의 physical matching 이득이라는 초기 해석은 대조 실험으로 수정했다. | 2026-11-07 | Track A | Exploratory | UNRECORDED — 해당 study의 artifacts.json 소스 해시 참조 |
| 7 | [음향 단서와 물리적 해석](track-a/experiments/2026-11-07-acoustic-cues/README.md) | 시간·envelope 및 초기 복소 채널 관계의 판별 정보를 확인했다. 고정 직접 경로·정규화 때문에 절대 음량/직접 도달 시간 해석은 지지되지 않았다. | 2026-11-07 | Track A | Exploratory | UNRECORDED — 해당 study의 artifacts.json 소스 해시 참조 |
| 8 | [반복 음향과 후보 쌍](track-a/experiments/2026-11-07-pairwise-acoustic/README.md) | 모호한 쌍의 acoustic 정보는 있으나 GEOQ 이후 추가 joint 개선은 미확인이다. C3−GEOQ −0.104pp, CI[−0.334,+0.114]. | 2026-11-07 | Track A | Exploratory | UNRECORDED — 해당 study의 artifacts.json 소스 해시 참조 |
| 9 | [Domain alignment와 discrimination](track-a/experiments/2026-11-07-alignment-discrimination/README.md) | 양성 거리 감소만으로 판별 개선을 설명할 수 없다. Positive-only alignment는 표현을 수축시켰고 matched downstream은 GEOQ를 넘지 못했다. | 2026-11-07 | Track A | Exploratory | UNRECORDED — 해당 study의 artifacts.json 소스 해시 참조 |
| 10 | [경쟁 후보와 negative sampling](track-a/experiments/2026-11-07-candidate-sampling/README.md) | 학습 negative와 실제 경쟁 후보의 차이를 확인했지만 candidate-aware sampling이 강한 GEOQ를 개선하지 못했다. Multi-positive의 상대 회복을 최종 개선과 구분한다. | 2026-11-07 | Track A | Exploratory | UNRECORDED — 해당 study의 artifacts.json 소스 해시 참조 |
| 11 | [방향성 정보 교환](track-a/experiments/2026-11-08-directional-interaction/README.md) | A→V는 일부 visual 판별을 돕지만 같은 정보의 concat/scalar를 안정적으로 넘지 못했다. E0+GEOQ 개발셋 신호는 HSSD 연구의 전체query 후속 검증에서 확정되지 않았다. | 2026-11-08 | Track A | Exploratory | UNRECORDED — 해당 study의 artifacts.json 소스 해시 참조 |
| 12 | [잔여 정보와 공통 공간 표현](track-a/experiments/2026-11-08-residual-spatial/README.md) | 잔여 feature·angular/Fourier·capacity·내부 검증 대조를 거쳤으나 GEOQ 이후 안정적인 추가 이득은 미확인이다. 좌표 정합 성공은 실제 DoA 판별 증거와 다르다. | 2026-11-08 | Track A | Exploratory | UNRECORDED — 해당 study의 artifacts.json 소스 해시 참조 |
| 13 | [HSSD 준비도·확대·학습 조건 복원](track-a/experiments/2026-11-08-hssd-scaling/README.md) | 135개 non-Test 장면 감사와478 fits. 후보 geometry의 matched 이득은 유지됐으나 기존 장면별 가중치 복원 후12→24 확대 이득은 불확실했다. | 2026-11-08 | Track A | Exploratory | UNRECORDED — 해당 study의 artifacts.json 소스 해시 참조 |
| 14 | [Directional Evidence / Tri-branch Pilot](track-a/experiments/2026-11-09-directional-tribranch/README.md) | Coarse acoustic structure는 예측 가능하지만 GEOQ 대비 추가 joint/fine 이득은 미확인. 기존 B2 Top20 및 FP-RIR reference 의존이므로 Track A. | 2026-11-09 | Track A | Exploratory | UNRECORDED; base19c1bd02f6edc6758ee9b18675957a2d30b3aeed + provenance.json 외부 script hashes |

| 15 | [독립 Dense SE(2) 초기 구현·실험](track-b/experiments/2026-11-09-dense-se2-initial/README.md) | track-b-dense-se2-initial-20261109: 데이터 감사·전체375,156후보 forward 검증 완료. 초기 모델의 개발셋 localization은 낮음; 42fits·대조·실패 분석 완료(성능 음성결과), 전체 protocol 미완료. | 2026-11-09 | Track B | Exploratory | 55c423833710efe6dd38dbcd58a6e18236ce6f9d (120step); c268a2f3574e7ba82fe566e39c0803e4f6cf737c (2000step); e45b12137c44b05ab9d4cb6b1ea1b54fb812665b geometry follow-up, provenance 참조 |

| 16 | [방향 구조와 Visual ambiguity](track-a/experiments/2026-11-09-directional-structure/README.md) | directional_structure_20261109: 완료·제한적 성공/음성결과 보존. 고정 visual 후보 재정렬 New12 +3.125pp, Transfer11 +3.267pp; cosine/GEOQ 우위는 미확인, fullbank fine 악화. | 2026-11-09 | Track A | Exploratory | d28ad93b3a0d65efd73da05e516b54274d84dc28; 실제 fit별 실행 SHA는 보고서 execution_receipts.json 참조 |

| 17 | [구조 지도학습·동일 초기화 대조](track-b/experiments/2026-11-09-structure-supervision/README.md) | track-b-structure-supervision-20261109: 30fits 완료·혼합/음성결과. 동일 초기화 partial9.028% > joint6.163%, 보조지도 대 dense-only 이득 미확인; 독립 고정 fusion8.941%, NLL 보정의 정확도 이득 없음. 강한 UnLoc·Support 실패 보존. 전체 protocol 진행 중. | 2026-11-09 | Track B | Exploratory | fb6900d (supervised), efcb0cb (controls), 38f76f6 (regimes), c20ce07 (matched); 전체 SHA는 experiment_registry.csv/provenance.json |

| 18 | [후보별 interaction·동용량 대조](track-b/experiments/2026-11-09-matched-interaction/README.md) | track-b-matched-interaction-20261109: 9fits 완료·음성결과. 동일 초기화·용량 attention6.25%, concat/separable7.38%; 우위 미확인. 수치 재생·캐시 비용·참조 bank 범위 감사 완료, 전체 protocol 진행 중. | 2026-11-09 | Track B | Exploratory | aa452ee78a75af31d8b426684a9c81f94ff00039 (수정 실행), 65a8236103d82de5b0c5ac3a68c2ff9dd1828b8c (중단) |

| 19 | [독립 사전학습 시각 특징](track-b/experiments/2026-11-09-foundation-visual/README.md) | track-b-foundation-visual-20261109: 12fits 완료·혼합/음성결과. 동일 초기화 frozen19.27% > joint5.56%, NLL은 개선돼 정확도와 괴리. 손실/FP 대조 진행 중. | 2026-11-09 | Track B | Exploratory | bf032b52959e4eff7351df53a11338ebd40dc580 |
| 20 | [Strong independent fusion](track-b/experiments/2026-11-09-strong-independent-fusion/README.md) | 완료·혼합: 강한 시각 대비 독립 결합 Joint +6.68pp; 밝기0.2에서도24.57%, 시각 자체 저하 없음. raw Top20 감소. | 2026-11-09 | Track B | Exploratory | `5aecaaa240b3a5bf4a0b0f3245e1dfdd19dc7303`; `1b20f6b4daac0d0019300c3380e224db41cdef36` |
| 21 | [Foundation matched interaction](track-b/experiments/2026-11-09-foundation-interaction/README.md) | 9fits 완료·음성결과: attention16.75%, concat18.23%, separable15.54%; 모두 초기 독립결합23.87% 미달. 동일초기화·정상재생 검증;75% 가림attention1.82% < 음향5.12%. | 2026-11-09 | Track B | Exploratory | `6285d4b89e127f5b71479c49071dcc96a2ff908c`; `18c0e3bf4297b9e0176ac3008710de06ec604da9` |
| 22 | [Grid and mask sensitivity](track-b/experiments/2026-11-09-grid-sensitivity/README.md) | 독립F1·joint9모델의6격자 평가 완료. 각3456건 기본재현 불일치0; attention 모든 격자에서 F1 미달. .50m raw20 상승은 MAP 개선 아님. 향후 선택 regime 검증 별도. | 2026-11-09 | Track B | Exploratory | `74e761a2a1584539daab0bb52d8942a7394a2beb`; `6e6f3a5380b6b124ebfdb978b4c1978baad7c275`; `09fddf31d86c02f61551beb68678ed8721f25fbf` |
| 23 | [OOF full-field discrimination](track-b/experiments/2026-11-09-oof-discrimination/README.md) | 30fits 완료·기하 추가 음성결과: diverse20 점수통계29.17% vs 기하24.48%, MAP23.87%. 후보 OOF분리·37검사 통과; head4,608건 지표 재현·정답/ID 입력 없음(전체 추론 아님); 전체 protocol 진행 중. | 2026-11-09 | Track B | Exploratory | `0aedb2ad9cb4690501c7f8ff0e648630241f983b`; `e04336cb2fdbaa8ccde4d5d862840d85298e5490`; `a8cbe16bee26188c1dec6312ca9e780977dcf90e`; audit `30b735c32cacc432e9a96e5489e5c39643f59d95`  추가: [M1 development 후보 연결](track-b/experiments/2026-11-09-oof-discrimination/joint_candidate_validation.md), builder `af902d67d38abe42c3149f9bdc36e0b1eac2d76b`; OOF M1 미완료; 분할 검사·smoke 통과, OOF M1 6fit 진행 중 `884864a35590bb259b34fd8c2e4e785fae9b16ca`. |
| 24 | [Foundation frozen/partial/joint](track-b/experiments/2026-11-09-foundation-regimes/README.md) | 완료27fits: 초기화·고정텐서 보존 검증. frozen/partial도 초기F1보다 Joint6.77–7.73pp 낮음; 인코더 변화만으로 저하 설명 불충분. 구형 출처필드 집계실패 보존 후 해시기반 이력 검증. | 2026-11-09 | Track B | Exploratory | `062860067f82a5f6a0e6ce8411b5c43702ae0ab5`; smoke `b50f10d1afc71396024a30e821f3409f5a41199e` |
| 25 | [Foundation map controls](track-b/experiments/2026-11-09-foundation-map-controls/README.md) | 완료:17,280건·정상재생 불일치0. 기하제거는 거의 균일장, 무작위 yaw attention3.13% vs 정상16.75%; 위치지표 불변인 방향치환 대조, 신규 기작 증거 아님. | 2026-11-09 | Track B | Exploratory | `cd35096d650ee209fd483c835d7c0d193d41b65e` |
| 26 | [Raw observation runtime audit](track-b/experiments/2026-11-09-raw-runtime-audit/README.md) | 완료:6장면 고정질의, 원본 RGB/body-RIR→특징·605,304점수·Top20 비트일치. 41검사 통과; 오프라인 정렬/단일모델 표본·비격리 비용 한계 명시. | 2026-11-09 | Track B | Exploratory | `0c0f0eece4b1bcd6595dc420463684c668875710` |
| 27 | [Expanded development cohort](track-b/experiments/2026-11-09-expanded-cohort/README.md) | 완료36fits/58,368평가. 16→112장면 동일업데이트 F1 Joint+7.24/+8.58pp; 7배업데이트는16장면−2.14pp,112장면 불확실. 초기화·질의·4조건 검증; 재사용Val/독립Test 아님. | 2026-11-09 | Track B | Exploratory | training `5cf803681c2d144f3a620bc00c1ab6b841e48e7c`; compare `518f033995cd40ef353021e893278ab4dea2a9dc` |

| 28 | [Frozen score-scale/interaction controls](track-b/experiments/2026-11-09-frozen-score-controls/README.md) | 완료13,824건·정상/초기 재현 불일치0. 스케일 복원 Joint+2.34–2.78pp, 결합 제거+5.21–5.99pp; 둘 모두 초기F1 복원. NLL 개선과 위치성능 악화 공존; 새 결합 성공 아님. | 2026-11-09 | Track B | Exploratory | `d8ca9d8b836f1c7178ce07056faf3b9f96af9c2f` |

| 29 | [Expanded matched interaction](track-b/experiments/2026-11-09-expanded-interaction/README.md) | 진행 중:112Train/19Val matched9모델 시작. CPU보관·43검사·전체데이터2-step 통과, 초기F1 1,216질의 재현 불일치0; 확대 밝기특징 검증. | 2026-11-09 | Track B | Exploratory | `c083c376631c3f6ed8d89daf9d3a9d93c350fd3b`  집계 출처 검증: [기록](track-b/experiments/2026-11-09-expanded-interaction/aggregation_validation.md), 검증 코드 `cf6db637f4405a2678d9a7c3e35f3b2e3f8206da`. |

[현재 결론과 반증 관계](current_findings.md) · [공통 평가 원칙](shared/evaluation-protocol/README.md)

서로 다른 cohort의 절대 성능을 이 목록으로 순위 매기지 않는다. 최신 HSSD478 fits와 residual1032 fits 등은 내부 fold/refit을 포함하는 계산 횟수이며 독립적인 과학적 증거 수가 아니다.


## 누적 관리 규칙

유일한 통합 RESEARCH_INDEX다. 이전 experiment_index.md를 rename했으며 중복 Index는 없다. 기존13개 결과는 의존성 확인 후 Track A로 이동했고14번째 로컬 Pilot도 Track A로 보관했다. Track B의 독립 Framework 초기 실험은15번에 기록했다. 각 연구의 실패/중단/부정 결과를 보고서에서 보존한다. 신규 행은 고유 ID/실험명, 날짜, Primary Track, 상태, 핵심 발견, 증거 수준, 보고서 경로, 실제 실행 코드 Commit SHA를 포함한다.

기존 실행 코드는 외부 폴더이므로 정확한 실행 Git SHA 미기록은 UNRECORDED로 표시했다. artifacts.json/provenance.json의 소스 해시와 reproduction 문서를 참고한다. Docs SHA를 실행 SHA로 바꾸거나 해당 Docs Commit 자체의 Index에 삽입하지 않는다. Docs SHA는 최종 실행 보고에 기록한다.

기존 연구는 재사용·teacher 노출 등을 고려하여 보수적으로 Exploratory다. Development-validated/Independent-validated 승격에는 프로토콜과 독립성 근거가 필요하다. 성공 여부와 증거 수준은 별개다. Track 간 비교는 Primary Track에 보고서를 하나만 두고 다른 Track에서 링크한다.

Study 19 follow-up: all 21 foundation fits complete; 2×2 FP/auxiliary control gives Joint 5.56/6.25/11.72/18.14%, below frozen 19.27%; mixed/negative, Exploratory reused development. Actual added training SHA `6874941298622eee18819fd48207fea8d5df2812`. Report: [foundation objective controls](track-b/experiments/2026-11-09-foundation-visual/README.md). Overall Track B ongoing.

Study 19 follow-up (2026-11-09): 24 fits completed, mixed/negative, Exploratory. Same-weight FP swap +6.25 pp [5.12,7.38]; trainable FP anchor reaches Joint17.19%, +5.47 pp [3.99,6.86] versus unanchored, without beating frozen19.27%. Actual training SHA 195b6066f4108caeab8a8149c9304439e4ac3d66; inference source fde1de84a95354a2d4ea2638f66c503edb762efa. [Report](track-b/experiments/2026-11-09-foundation-visual/README.md). Full Track B ongoing.

[Track B protocol coverage audit](track-b/protocol_coverage_audit.md): full objective incomplete; M1-specific backend provenance, scaling, matched-model regime/grid controls, physical-shift feasibility and complete costs remain; independent-F1 OOF backend and matched occlusion metrics are complete. Study20 complete occlusion curve:75% occlusion fusion2.26% < acoustic5.12%; normal-calibrated NLL worsens; source `105ae0f95a393a0c9cd22f79784e1bb65759e57e`, aggregate `b5ff944ff758d4552ca347a3f647122c39f64d93`.

[External primary-source reference review](track-b/external_reference_review.md): observation versus sequence and semantic-input differences recorded; full-paper/reproduction and broader audio novelty checks remain. Study22 first fixed-checkpoint grid evaluation preserved (superseded by completed v2 below), actual source `1f79b9d95e699cebd80374e09f69f3874522d532`.

Study22 final F1 evaluation source `6e6f3a5380b6b124ebfdb978b4c1978baad7c275`; numerical first run preserved and final baseline replay has zero mismatches. Full Track B remains ongoing.

## Track A 추가 연구 — ray_joint_20261109

| 연구/ID | 날짜 | Track | 상태·핵심 발견 | 증거 수준 | 보고서 | 실험 코드 Commit SHA |
|---|---|---|---|---|---|---|
| Ray representation & joint learning / ray_joint_20261109 | 2026-11-09 | A | 완료·혼합/음성결과:228checkpoint·1542배열 재생 일치, fullbank2880+conditioned1728기록 검증. Chirp−raw +2.897pp; attention−concat +3.060pp이나 interaction-disabled 우위 미확정. 잔차가중 attention +2.181pp, anchor선택 악화. B2+GEOQ 미개선 및 fine 손해 보존. | Exploratory | [연구 보고서](track-a/experiments/2026-11-09-ray-joint/RESEARCH_REPORT.md) | 18f04ce (입력), 570df35 (결정성 수정), 338a449 (calibration), 6373431 (donor 수정), ff5601d (Transfer), 121fd73 (방향순열), 00502cb (최종 분석); 전체 실제 실행 SHA·소스 해시는 execution_receipts.json/source_inventory.json |

| 30 | [Ray uncertainty diagnostic](track-b/experiments/2026-11-09-ray-uncertainty/README.md) | 완료·주 가설 지지 실패: 7조건/3seed. 정상 U3 Joint29.688% 대 U0 34.402%; U2 대비 추가 pose 이득 미확인, severe occlusion negative transfer 지속. 원본 F1 보존·전체 집계 검증. | 2026-11-09–10 | Track B | Exploratory | uncertainty fit `3caccd76fd46289f9301624255584fe0df490394`; evaluation `bf9a825cf58f584e6803f5026977fc1b32c7aafb`/`ba040da65e3cf3339a716e9dc8a6af07c1a488d7`; aggregation `8f2454a5ce50f0fff059dcc180da2d07057c1920` |

## Track A 진단 도구 — pipeline_viewer_20261110

| 연구/ID | 날짜 | Track | 상태·핵심 발견 | 증거 수준 | 보고서 / HTML | 실제 실행 코드 Commit SHA |
|---|---|---|---|---|---|---|
| Interactive pipeline diagnostic / pipeline_viewer_20261110 | 2026-11-10 | A | 완료: New12 12장면·58질의·2seed의 frozen score/Support20/GEOQ 재생 불일치0. 무작위·실패5유형 포함; 공개 URL 브라우저 검증 통과. 신규 학습·렌더링·Test17 분석 없음. 독립 성능/feature-exchange 우위 주장 아님. | Exploratory (재사용 개발 진단) | [보고서](track-a/experiments/2026-11-10-pipeline-viewer/RESEARCH_REPORT.md) · [Interactive HTML](https://rawcdn.githack.com/LedersonLee/HTML-reports-2/f2cb4aaa2feb431ecacefef4cd018af6eaee577c/echoloc/av-fploc-track-a-pipeline/index.html) | b85258053b3e1d6763fc18c600a62ea4ea3f73ff (최종 exporter); 역사적 학습 SHA UNRECORDED·source/checkpoint hashes는 manifest 참조 |

| 31 | [Learned acoustic structural geometry](track-b/experiments/2026-11-10-acoustic-geometry/README.md) | 완료·방향구조/전이 가설 지지 실패:18표현fit·21전이fit·물리대조·개발holdout19·공식Test17 평가/검산. Test Spatial MAE1.567m, 다른위치RIR교체+.263m이나 방향증거 미확인. Fusion joint legacy34.835%,scratch_total29.075%,finetuned25.766%; pretrain 이득 미확인. | 2026-11-10 | Track B | Exploratory | targets `f9d0f2e92a8d45cb750d65f38f4f3bed14859a42`; features `bd05ef297da92b085975e9c53d4a2c08b73a6f0b`; training `13f18b028b35d694c8e0adee4d138d46fecf9d00`; aggregate `6be2af1a690e039d1453e653d573607aade2cc4c`; diagnostic `774e7908f4f60677f33cb2476a6f75f64feb1831`; transfer `f6e9d922df2b5df31c976a6da363fb1193ee4920`; physical control `8ee29be225f83f9a0786e08201fefa9379eeda98`; holdout eval `6907b72`; verification `1e9b5b9ecf6f185f483486a26f7e339809d5a8bd`; final contrasts `3ee938b` |

## Track C 사후 시각화 — study31_visualization_audit_20261111

| 연구/ID | 날짜 | Track | 상태·핵심 발견 | 증거 수준 | 보고서 | 실제 분석 코드 Commit SHA |
|---|---|---|---|---|---|---|
| Study 31 visualization audit / study31_visualization_audit_20261111 | 2026-11-11 | C (사용자 지정 사후 해석; 원 연구 B 유지) | 완료·음성결과 해석: 저장된 공식 Test 17장면·3seed의 6사례 및 분포 그림. Spatial donor +0.263m, scalar +0.00030m, rotation +0.00224m; angular r 평균0.0159이나 분포는 넓음. 위치별 거리 수준 의존성은 지지, 유용한 방향 복원 미확립. 신규 학습/추론 없음. | Exploratory (공식 Test 사후 재사용·upstream 노출 한계) | [시각화 보고서](track-c/2026-11-11-study31-visualization-audit/README.md) | 55155a7ee03dc21cec0f5b112c4c171ad661db28 (배열 분석/그림); 원 학습13f18b0·평가6907b72, 입력 SHA-256은 audit.json |

## Track A 진행 연구 — AVLoc independent / avloc_20261111

| 연구/ID | 날짜 | Track | 상태·핵심 발견 | 증거 수준 | 보고서 | 실제 실행 코드 Commit SHA |
|---|---|---|---|---|---|---|
| AVLoc independent reimplementation & matched benchmark / avloc_20261111 | 2026-11-11 | A | 진행 중: 원문/리뷰 명세 감사,6ch 전해상도 audio·A0–A3 구조/분리/router 검증 통과.112Train/19Val 68,859질의 전체 감사 통과, 실제 역전파·재개 일치 검사 통과; A0 seed42 전체 학습 시작. 19Val×2격자 matching 검증·19,930행 GEOQ 재생 일치; A0 epoch3 Val ray MAE0.334m(중간); B2seed43 fullVal joint69.24% 재생완료, B2seed43 temperature NLL9.95→5.16/ECE.680→.155이나 pose불변; full baseline완료: joint V45.85/A21.36/fixed65.04/B269.31/GEOQ73.07%, GEOQ fine은fixed보다1.084pp낮음. 전문가 학습 진행. 학습 정량 비교·교정·비용 분석 미완료. SSD 대체 경로0-query 실패 보존. | Exploratory | [연구 진행 보고](track-a/experiments/2026-11-11-avloc-independent/RESEARCH_REPORT.md) | 학습 62c58da20ed26ba95863638bbfd4926829999eec; matching 8eaaf247ae20ac39e132e2192bafee3f8b18ae4c; baseline replay c81f817b2e8a17cd2d12e2b0d845e17337cd5cef; 초기 감사 base e3e0500 + 당시 소스 해시는 source_snapshot.json. |

AVLoc study update (2026-11-11, Exploratory; primary Track A): [seed42 calibration and10cm geometry diagnostics](track-a/experiments/2026-11-11-avloc-independent/RESEARCH_REPORT.md) completed on19Val/9,965queries. All39,860 native checks passed; calibration improved NLL/ECE without Top1 changes. Perfect-profile10cm results are not learned accuracy. Review code `9da96e33607fe0ce8ce9b105e444d2673cfba2f7`; training remains in progress under its recorded source.

AVLoc evaluation implementation update (2026-11-11): ray temperature calibration uses the existing baseline family-fold protocol and native replay checks; [synthetic checks](track-a/experiments/2026-11-11-avloc-independent/ray_calibration_test.json) pass, trained execution pending. Source `a3731d07c0a747307b7d544820f81ffbdfb48725`. Evidence remains Exploratory; no new learned-model conclusion.

AVLoc analysis implementation update (2026-11-11): [ray observability analysis and numerical tests](track-a/experiments/2026-11-11-avloc-independent/RESEARCH_REPORT.md) prepared; source `c44ce51cc7c2fcbaf4a8726f2dd4d8ef496f3a26`. Trained application remains pending; evidence level unchanged (Exploratory).

AVLoc execution update (2026-11-11): [A0 evaluation queue](track-a/experiments/2026-11-11-avloc-independent/a0_evaluation_queue_plan.json) attached to the verified live training container; seven full-Val stages pending normal completion. Source `1a62afe422acf1c605dbfe6ef84f3afbe4346a68`. Not a completed model result.

AVLoc Router analysis implementation update (2026-11-11): [replay-checked weight/error and uncertainty diagnostics](track-a/experiments/2026-11-11-avloc-independent/RESEARCH_REPORT.md) prepared; four analytic tests passed. Source `cdc8b6cd01d6b312f83c01afc30d52f9e14f243e`. Actual A3 application and independent reliability evidence remain pending; Exploratory, no new trained-model result.

## Track A Experiment 1 — reflection_observability_20261111

| 연구/ID | 날짜 | Track | 상태·핵심 발견 | 증거 수준 | 보고서 | 실제 실험 코드 Commit SHA |
|---|---|---|---|---|---|---|
| Geometry-guided Reflection Observability / reflection_observability_20261111 | 2026-11-11 | A | 완료·핵심 물리 시간 정렬 가설 음성/조건부 energy 연관성 양성.96controlled/diagnostic rings; HSSD12장면384질의7680visual후보·6대조. 단일벽2/6m에서약+2.36/+6.15ms지연,좁은expected창귀속energy거의0. HSSD matched-time control 대비power+0.2619[0.1671,0.3681],coherence개선없음. Localization/scorer/Experiment2–4미수행. | Exploratory (시뮬레이션·재사용Train/development·upstream노출) | [RESEARCH_REPORT](track-a/experiments/2026-11-11-reflection-observability/RESEARCH_REPORT.md) | renderer `6a6fc1e83fa8f060d773393988e3da4b535b65a7`; paths `2c33fe3a9abaef3e70f953004d639d3169f7e97c`; final controls `fdbfb08d37b7a6b98deb5ecae58ffaa74438b4cc`; diagnostic sources/pilot dirty provenance는REPRODUCIBILITY.md참조 |


## Track A Experiment 1B — reflection_timing_audit_20261111

| 연구/ID | 날짜 | Track | 상태·핵심 발견 | 증거 수준 | 보고서 | 실제 실험 코드 Commit SHA |
|---|---|---|---|---|---|---|
| Reflection Timing & Renderer Physics Audit / reflection_timing_audit_20261111 | 2026-11-11 | A | 완료·Outcome C(Renderer-dependent Artifact). 기존96RIR 검증/60peak 재생; 신규420mono. Scattering0.5→0만 변경해 peak bias2m +2.347→−0.134ms,6m +6.168→+0.114ms. 잔여 절대오차/내부path GT 미확보로 first-order 정렬은 미검증. HSSD D gate 실패로 생략; Experiment2–4 미수행. 이전 결과 보존 및 원인 해석 확장. | Exploratory (재사용 통제 환경·사후 원인 진단; 독립 Test 아님) | [RESEARCH_REPORT](track-a/experiments/2026-11-11-reflection-timing-audit/RESEARCH_REPORT.md) | C API `4a9070a5faa60bb9f26c60f78f78edac862b3ffb`; runner `f5d16dbbdd6b02826143c6b9136b06588fc7ecbb`; estimators `f86af5d81ef89ee46c56ae9e9ce6529d58a81a17`; scattering `46c1ddfdea29e30f6230126752b27a735f9b84d1`; final audit `65e7eb1588fd7d99a84f6cb5da201dff02269ab1`; pilot dirty/failed provenance는 REPRODUCIBILITY 참조 |


## Track A — semantic_sound_poc_20261111

| 연구/ID | 날짜 | Track | 상태·핵심 발견 | 증거 수준 | 보고서 | 실제 실험 코드 Commit SHA |
|---|---|---|---|---|---|---|
| HSSD Semantic Sound Pose Disambiguation / semantic_sound_poc_20261111 | 2026-11-11 | A | 완료·Oracle 이득 불확실/물리6mic 부분 실패. 1scene124RGB×3seeds: B0 58.06%, B3 63.17%, +5.11pp CI[-2.15,12.63]. B1=B0; 15°noise 이득 없음. 165가구 포함 RIR, 4/6조건TDOA<1sample; 거실 직접음 차단 및10cm receiver sphere 한계. 무효 cast_ray 검출 후16/1108source points 제거·모든 대조 재실행, 초기 결과 보존. 실제 시스템 개선 주장은 No-go. | Exploratory (training scene·synthetic source·GT bearing·upstream teacher/평가 재사용) | [report](track-a/experiments/2026-11-11-semantic-sound-poc/report.md) | primary renderer `ee9e56cd9027011a4c08132928b06d09e352ce4e`; depth-corrected evaluation `77ad36897f62d466b9d5975a64f65800769a962d`; final inference boundary replay `8fe5e91` (full SHA in correction.json); diagnostics dirty-source provenance in manifest/reproduction |

AVLoc Router replay validation correction (2026-11-11): scalar fused-output acceptance reproduced and fixed; six analytic tests passed. Source `151b969075837042309ad4244d0857952b8dc823`. [Correction evidence](track-a/experiments/2026-11-11-avloc-independent/RESEARCH_REPORT.md#router-replay-input-validation-correction). Exploratory engineering validation; trained A3 evidence remains pending.


## Track A Experiment4A — material_observability_20261111

| 연구/ID | 날짜 | Track | 상태·핵심 발견 | 증거 수준 | 보고서 | 실제 실험 코드 Commit SHA |
|---|---|---|---|---|---|---|
| Material-dependent Acoustic Observability / material_observability_20261111 | 2026-11-11 | A | 완료·Outcome B.1980controlled rings+72HSSD native-stage rings. 흡수·profile 효과6poses에서일관,산란효과는기하의존. 결합rule제거벽FPR70.6%,first-order귀속미검증. 대표corridor_near+room_far/single6예외의324조건explorer;349UI검사. 새Network/Experiment2/3미수행. | Exploratory (설계3기하·6poses,재사용Train2scenes;pathGT/실환경일반화미검증) | [RESEARCH_REPORT](track-a/experiments/2026-11-11-material-observability/RESEARCH_REPORT.md) · [Explorer](track-a/experiments/2026-11-11-material-observability/interactive.html) | renderer `aff5991`; profiles/main `47d6740`; analysis `97b5d71013f42f15e372d2f76ebdd51bffae2399`; finalselection `e54e706`; verifier `239de7d`; viewer `a1b0e83`; fullSHA/inputhashes는reproducibility_manifest.json |


| 연구/ID | 날짜 | Track | 상태·핵심 발견 | 증거 수준 | 보고서 | 실제 실험 코드 Commit SHA |
|---|---|---|---|---|---|---|
| AVLoc A0 seed42 initial evaluation | 2026-11-12 | A | A0 100epoch 완료; best99 MAE0.17957m;20cm Joint50.1054%;사후 분석 boolean 오류 수정 및 재개,전체 연구 진행 중 | Exploratory; reused Val19, single seed | [A0 initial results](track-a/experiments/2026-11-11-avloc-independent/A0_INITIAL_RESULTS.md) | training `62c58da20ed26ba95863638bbfd4926829999eec`; analysis/recovery `5f8fdc4e8fdd22a17f1a243646c263fcd23e9ae1` |

AVLoc A0 matched follow-up (2026-11-12, Track A, Exploratory):20cm Joint improves4.255pp over existing visual baseline, with791 harmed queries; below fixed/B2/GEOQ. Separate10cm Joint60.462% is a grid effect. [Evidence](track-a/experiments/2026-11-11-avloc-independent/A0_INITIAL_RESULTS.md#matched-baseline-integration-and-separate10cm-evaluation). Aggregation code `81bcc165bba8bef19b43239d3d6a459f3cb938e6`; training `62c58da20ed26ba95863638bbfd4926829999eec`.

AVLoc A0 calibration replay (2026-11-12, Track A, Exploratory):both grids verified9,965 queries each; temperature lowers NLL but leaves poses unchanged and residual ECE high. [Evidence/figures](track-a/experiments/2026-11-11-avloc-independent/A0_INITIAL_RESULTS.md#temperature-replay-verification-both-grids). Analysis code `1a494e31f89a23fa42ef6f0dd371db707004880c`.

AVLoc A1 seed42 training (2026-11-12, Track A, Exploratory):70 executed epochs, best50 MAE1.478506m; receipt71 is a loop-counter reporting error, not an extra trained epoch. Full evaluation launched. [Report](track-a/experiments/2026-11-11-avloc-independent/A1_TRAINING_COMPLETION.md). Training `62c58da20ed26ba95863638bbfd4926829999eec`; evaluation runner `8e77076c4f9c0f7cb0db32fceb2b230a3f16f31b`.

AVLoc A1 initial negative matching result (2026-11-12, Track A, Exploratory):20cm Joint2.4285%,R@1m9.2524%; well below FP-RIR cosine. Directional variation attenuated, mechanism unproven. [Evidence](track-a/experiments/2026-11-11-avloc-independent/A1_TRAINING_COMPLETION.md#initial20cm-matching-negative-result). Training `62c58da20ed26ba95863638bbfd4926829999eec`; runner `8e77076c4f9c0f7cb0db32fceb2b230a3f16f31b`; aggregation `81bcc165bba8bef19b43239d3d6a459f3cb938e6`.

AVLoc independent equal fusion (2026-11-12, Track A, Exploratory):20cm Joint9.553%,below A0 by40.552pp;418repairs/4459harms.10cm/calibration in progress. [Report](track-a/experiments/2026-11-11-avloc-independent/INDEPENDENT_FUSION_RESULTS.md). Runner `846c49d4b87563e509faceff6c07b69296008d7b`; frozen experts training `62c58da20ed26ba95863638bbfd4926829999eec`.

AVLoc A1/fusion calibration (2026-11-12, Track A, Exploratory):A1 ECE0.00344 coexists with Joint2.4285% and entropy0.95839; low-confidence concentration, not restored localization. Three full-cohort replays passed. [Report](track-a/experiments/2026-11-11-avloc-independent/A1_FUSION_CALIBRATION.md). Code `1a494e31f89a23fa42ef6f0dd371db707004880c`.

AVLoc fusion2x2 decomposition (2026-11-12, Track A, Exploratory):preserve visualFoV+audiooutside Joint52.303% vs A050.105%;macro+2.2835pp CI[1.2879,3.2363],fineCIincludes0. FoVequalmix alone7.396%,locating major harm insideFoV. [Report](track-a/experiments/2026-11-11-avloc-independent/FUSION_DECOMPOSITION.md). Construction `457d3dd`; summary `d9183aea316265f1567a01dc7ce43acfff55b925`.

AVLoc A2 seed42 training (2026-11-12, Track A, Exploratory):99 executed epochs,best79 RayMAE0.560813m; matched evaluation launched,localization/interaction conclusion pending. [Report](track-a/experiments/2026-11-11-avloc-independent/A2_INITIAL_RESULTS.md). Training `62c58da20ed26ba95863638bbfd4926829999eec`; evaluation `8e77076c4f9c0f7cb0db32fceb2b230a3f16f31b`.

AVLoc A2 initial20cm matched evaluation (2026-11-12, Track A, Exploratory):Joint64.927%,fine30.467%;versusB2+GEOQ Joint−8.144pp butfine+7.265pp;improves independent controls,causalinteraction notyetisolated. [Report](track-a/experiments/2026-11-11-avloc-independent/A2_INITIAL_RESULTS.md#completed20cm-evaluation-accuracy-tradeoff-not-uniform-superiority). Training `62c58da20ed26ba95863638bbfd4926829999eec`; pairedanalysis `48882b376ec50fffdaab385ec6e94a0ccbf02e3f`.

AVLoc seed-priority update (2026-11-12, Track A, Exploratory, seed43 deferred/interrupted): A0seed43 saved through epoch8; prioritize seed42 interaction-disabled control and A3, preserve all outputs. No claim of seed stability. [Report](track-a/experiments/2026-11-11-avloc-independent/SEED_PRIORITY_UPDATE.md). Training `62c58da20ed26ba95863638bbfd4926829999eec`.

AVLoc expert gate (2026-11-12, Track A, Exploratory, validation complete): three seed42 experts verified on both grids, 9965 aligned queries each; A3 prerequisite passed without accuracy threshold. [Report](track-a/experiments/2026-11-11-avloc-independent/EXPERT_VALIDATION_GATE.md). Training code `62c58da20ed26ba95863638bbfd4926829999eec`; validator source hash in attached gate JSON.

AVLoc A3 scheduling (2026-11-12, Track A, Exploratory, queued): seed42 full model will follow successful interaction-disabled training; immutable source and expert evidence rechecked. [Protocol](track-a/experiments/2026-11-11-avloc-independent/EXPERT_VALIDATION_GATE.md#a3-scheduling). Scheduler `0b3f8e74e9db3065b14e075d1757c88618b805df`; training `62c58da20ed26ba95863638bbfd4926829999eec`; no new model result yet.

AVLoc frozen-grid and A2 calibration (2026-11-12, Track A, Exploratory, completed20/10cm grid comparison and20cm calibration review): A2 Joint64.927→74.180% with grid refinement; no retraining. Temperature reduces NLL154.848→5.460, Top1 unchanged. [Report](track-a/experiments/2026-11-11-avloc-independent/GRID_AND_A2_CALIBRATION.md). Training `62c58da20ed26ba95863638bbfd4926829999eec`; grid analysis `598696f`.

AVLoc matched score-field figures (2026-11-12, Track A, Exploratory, completed initial7-method replay): 4queries/3scenes including2uniformrandom and repair/harm extremes;28fields match saved evaluations. [Report](track-a/experiments/2026-11-11-avloc-independent/MATCHED_SCORE_FIELDS.md). Figure/replay code `4117246`; expert training `62c58da20ed26ba95863638bbfd4926829999eec`. A3/disabled pending.

AVLoc A1 success conditions (2026-11-12, Track A, Exploratory, completed diagnostic):20cm fine15cases/3scenes,7wrongyaw; lowest ray-error quartile Joint7.627%; ideal geometry alone does not explain acoustic failures. [Report](track-a/experiments/2026-11-11-avloc-independent/ACOUSTIC_SUCCESS_CONDITIONS.md). Analysis `50dd3b1`; training `62c58da20ed26ba95863638bbfd4926829999eec`.

AVLoc requested metric table (2026-11-12, Track A, Exploratory, completed reaggregation): position0.1/0.5/1m, Joint1m30°, yaw10/30/50° from saved finalposes;20cm primary and10cm separate. [Table and grid definition](track-a/experiments/2026-11-11-avloc-independent/REQUESTED_METRIC_TABLE.md). Training `62c58da20ed26ba95863638bbfd4926829999eec`; aggregation source hash in requested_metrics.json.

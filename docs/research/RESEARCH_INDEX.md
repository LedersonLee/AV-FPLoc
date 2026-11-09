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
| 23 | [OOF full-field discrimination](track-b/experiments/2026-11-09-oof-discrimination/README.md) | 30fits 완료·기하 추가 음성결과: diverse20 점수통계29.17% vs 기하24.48%, MAP23.87%. 후보 OOF분리·37검사 통과; head4,608건 지표 재현·정답/ID 입력 없음(전체 추론 아님); 전체 protocol 진행 중. | 2026-11-09 | Track B | Exploratory | `0aedb2ad9cb4690501c7f8ff0e648630241f983b`; `e04336cb2fdbaa8ccde4d5d862840d85298e5490`; `a8cbe16bee26188c1dec6312ca9e780977dcf90e`; audit `30b735c32cacc432e9a96e5489e5c39643f59d95` |
| 24 | [Foundation frozen/partial/joint](track-b/experiments/2026-11-09-foundation-regimes/README.md) | 진행 중: 동일초기화·고정가중치 보존38검사, 실데이터2-step 점검 통과. frozen/partial×3방식×3seed 신규18fits 시작; 기존joint9fits와 비교 예정. | 2026-11-09 | Track B | Exploratory | `062860067f82a5f6a0e6ce8411b5c43702ae0ab5`; smoke `b50f10d1afc71396024a30e821f3409f5a41199e` |
| 25 | [Foundation map controls](track-b/experiments/2026-11-09-foundation-map-controls/README.md) | 완료:17,280건·정상재생 불일치0. 기하제거는 거의 균일장, 무작위 yaw attention3.13% vs 정상16.75%; 위치지표 불변인 방향치환 대조, 신규 기작 증거 아님. | 2026-11-09 | Track B | Exploratory | `cd35096d650ee209fd483c835d7c0d193d41b65e` |
| 26 | [Raw observation runtime audit](track-b/experiments/2026-11-09-raw-runtime-audit/README.md) | 완료:6장면 고정질의, 원본 RGB/body-RIR→특징·605,304점수·Top20 비트일치. 41검사 통과; 오프라인 정렬/단일모델 표본·비격리 비용 한계 명시. | 2026-11-09 | Track B | Exploratory | `0c0f0eece4b1bcd6595dc420463684c668875710` |
| 27 | [Expanded development cohort](track-b/experiments/2026-11-09-expanded-cohort/README.md) | 112Train·19Val·8,384질의 구조캐시 생성·검증 완료, 기존22장면 텐서 동일. 시각특징131장면 검증·기존22장면 정확 일치. 동일Val19의 장면수×학습량2×2·36단계 중16장면 두조건18단계 완료·검증;112장면 학습 시작, Val19 F1 Joint 기본27.17%/7배25.03%(조건간 추론 보류), 각조건 raw20/ECE 악화; uniform Joint0.43%·지도131개 정확중복0; 고정가중치 평가4,608기록 정확 재현;43검사/부분집합 이어학습 점검 통과. CPU장면전송 짧은 대조 비트불일치(최대8.6e−8), GPU반복도 미세차이; 핵심 위치지표 차이0. 독립Test 아님. | 2026-11-09 | Track B | Exploratory | preflight `f70fe38bf674327afbc2764dfc93dc818d3cf962`; cache/stream `461cb6ea52dd76bec09acf3a2bb5be08f50139c6`; scaling `5cf803681c2d144f3a620bc00c1ab6b841e48e7c`; uniform `99b8cddfd70e279f6e1a011e62ba98bae2c4f633`; map audit `d268cab16c74beed65f2f53bea0913d93ab5405b` |

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

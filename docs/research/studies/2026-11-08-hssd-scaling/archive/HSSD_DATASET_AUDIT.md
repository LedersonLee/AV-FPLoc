> 역사적 연구 기록. 원본: `hssd_scaling_20261108_1332/HSSD_DATASET_AUDIT.md`. 원본 SHA256은 [산출물 목록](../artifacts.json)에 있습니다.
> 당시의 완료·재개 지시는 현재 실행 요청이 아닙니다. 최신 판정은 [현재 결론](../../../current_findings.md)을 먼저 확인하세요.

# HSSD DATASET AUDIT

완료 상태: readiness audit, 실제 동결 encoder 입력, 원본 renderer replay, 비용 조사와 358회 완료된 supervised fit을 수행했다. 상세 실험 결과는 RESEARCH_REPORT.html 및 기계 판독 JSON에 별도로 보존한다.

## 확인 범위

원본 scene configuration **168개**를 발견했다. HSSD-200 전체 확보로 해석하지 않는다. 기존 승인 split은 Train116/Val19/Test17이며, **Train/Val135개**에서 scene별 실제 RGB decode, 6채널 RIR 유한값·비무음, stage GLB header, FP binary/scale, DESDF, E/C/N 특징 샘플을 확인했다. 135개 모두 이 검사에 통과했다. 전체 raw 파일의 완전 검증은 아니다. 대표 renderer 실행은 아래 별도 검증했다. Test는 split 식별 메타데이터만 확인했으며 입력/성능을 열지 않았다.

## 완료율과 실제 사용 가능 범위

|자산|확인된 Train/Val 범위|현재 상태|
|---|---:|---|
|Furnished scene 구성 + structural stage|135/135|원본 stage와 object instance 구성 확인; 대표 scene 실제 renderer 재실행 통과|
|2D FP + 구조 DESDF|135/135|기존 curation 승인 + 현재 scale/finite 검사 통과|
|RGB·pose manifest|135/135|각 scene 샘플 decode; 학습용61,502 / 개발용9,965 query 기록|
|Observed RIR 및 FP paired references|135/135|6채널 raw 샘플 및 E/C/N 캐시 검사 통과|
|전체 dense candidate bank|55/135|Train36 + Val19. 나머지 paired8 후보를 full-bank localization으로 취급하지 않음|
|기존 encoder 호환 특징|135/135|E/C/N 형태 확인; 새23 scene 전체bank frozen encoder forward 통과|

완료율의 분모는 원본168이 아니라 기존 승인된 non-Test135다. 미승인16개 source scene은 사용 가능하다고 가정하지 않는다. 장면별 목록·파일·이유는 `scene_availability.json`, 전체 후보 bank 출처는 `dense_bank_catalog.json`에 있다.

## FP 품질

135개는 기존 automatic/manual structural curation을 계승해 accept로 분류하되, 이는 새로운 전 장면 수작업 mesh 검토가 아니다. 각 scene 기록에 해당 승인 이유를 보존했다. 원본 curation의 excluded/repair-required 장면은 새 학습에 넣지 않는다. 지도는 stage slice, semantic indoor support와 connectivity 검사를 사용하고, 해상도0.01m/구조 ray0.1m이다. Door/window/glass 비차폐 정책, 높은 창문·개구부와 2D 표현 차이, furnished depth 불일치가 남는다. Furniture-free FP를 collision-free navigation map으로 해석하지 않는다. Missing rooms/false connection은 기존 curation 증거이며 현재 샘플 검사만으로 전부 재검증되었다고 주장하지 않는다.

## 노출과 분리

기존 visual teacher는 Train116/Val19 노출이 있다. 따라서 이번 head-held 장면도 전체 pipeline의 독립 장면은 아니다. 기존 acoustic checkpoint는 Train12/Val3 선택 이력을 유지한다. 과거 B2/GEOQ의 Train12를 small로 유지하고, 새 dense23개를 면적 순으로 번갈아12학습/11평가로 배정했다. 최종24학습/11head-held, Val19개발을 사용한다. 신규 평가 결과 확인 전 분리를 고정했다.

Test와 같은 원본 prefix family의 Train3개는 보수적으로 제외한다. 그중 dense1개가 있어 Train36→35다. prefix만으로 furnishing 동일성을 확정하지 않으며 가능한 누출을 차단하는 보수적 그룹이다. Train/Val의103997478 family도 같은 그룹 취급하며 해당 Train variant는 선택된 denseTrain에 없다.

## 기존 자산과 provenance

기준 root: `${HSSD_ROOT}` 및 `${HSSD_SSD_ROOT}`.

- source/hssd-hab: scene/stage/object 구성
- 2DFP, UnLoc/structural_targets, frame_adapter_v4_furnished_qa: 지도/시각 입력
- AcousticContrastive/full_scene_v1: 전체 paired manifest/cache 기록
- AcousticContrastive/features/5b52966f431ea529/native_yup/4c28daac3d24c038: 수정 Y-up E/C/N
- ReliabilityFusion/expansions/train36_val8_test3_v1: denseTrain36/Val8 기존 bank 및 특징 (Test 하위 입력 미사용)
- RIR/manifests/dense_banks_v1: Val19 densebank

상단 README의 scene5-only RIR 설명은 현재 실제 자산보다 오래되었다. 초기 node2 실패 기록도 이후 full_scene 완료와 구분했다. source receipt 해시는 장면 JSON에 보존한다.

## 누락 작업과 비용

전체 원본 장면의 dense RIR 재생성은 시작하지 않는다. 우선 기존 dense35Train의 새23개에서 frozen encoder만 계산한다. 현재 샘플 점검 비용은 105.7초였다. 대표 장면 및 새23개 장면에서 실제 forward·저장 비용을 측정했다. 48/96 dense scene 실험은 현재 ready 자산 범위를 넘으며, paired8 후보 수를 늘리는 것으로 대체하지 않는다. 신규 raw rendering 비용은 아래 기존 8,824개 task receipt를 사용해 추정하며 신규 전체 렌더링은 하지 않았다.

## 실행 이력

`prepare_inputs.py --smoke`: 새Train scene의8query/64cells에서 동결 encoder forward를 검증한 후 전체bank·128query로 확대했다. full-bank baseline transfer와12/24 scene scaling, 후속 전체query 평가를 수행했다.

## 추가 실제 검증

- 새Train 102815859에서 furnished / HSSD uncluttered / true stage-only를 동일 좌표로 실제 Habitat 렌더링했다(`render_smoke/`). 원본 uncluttered와 stage-only를 구분했다. 기존 이미지의 ABI 설정과 작업 폴더를 복원해 실행했고 초기 실패 로그도 보존했다.
- 새 scene 동결 encoder pilot:8query/64cells(2,304 yaw 포함 후보), 입력 추출·추론·저장1.36초(모델 초기화 제외). 전체 densebank 추출 비용은 아래와 같다.
- 실내 semantic polygon과 FP의10cm screening:135scene 모두 큰 실내 영역 저피복/과도한 외부free 징후 없음. 초기 outdoor 포함 진단은 별도 보존하고 기존 전처리의 exterior-label 정책으로 수정했다. 이는 문/벽의 독립GT 검증이 아니다.
- 기존Val 장면128query, 두 시드에서 새 scorer의 후보 indices/labels 및 S/V/AC/AM/VT/AT 특징을 원본과 비트 단위로 재현했다(`baseline_smoke_replay.json`).
- 과거 dense rendering pilot72RIR의142.11초(1worker,2acoustic threads; task단위) 기록이 있다. 약1.97초/RIR이나 scene-dependent이며 새 장면 전체비용을 확정하는 수치가 아니다. 6개 heading×cell 수를 적용한 대략 비용보다 캐시 재사용 비용이 훨씬 낮아 현재 신규RIR 생성은 필요하지 않다.

## 완료된 추가 확인

- Non-Test135개 장면이 참조하는 object asset13,264개에서 configuration과 render mesh 누락0개. 텍스처 전체의 decode 검사는 아니며 대표 렌더링을 별도 수행했다.
- 새23개 장면의 전체 densebank와128query에 대한 frozen encoder 추출 완료. 신규RIR 생성0개. 원본 기록을 덮어쓰지 않았다.
- 실험54개 장면(Train35+Dev19)의 각128query에서 RGB metadata pose, RIR query pose, 구조 depth target의sampleID 정렬 확인 완료.
- 의미 영역 중심의 초기 렌더는 벽을 향하는 단조로운 뷰였으므로, 실제 승인query8의pose/480×640/HFoV106.2602/logicalyaw0→Habitat−90°로 추가 렌더했다. 원본RGB와 평균 절대 픽셀 차이는 0.839/255이며, 실내 구조·가구가 보이는 것을 확인했다. 완전 픽셀 동일성으로 주장하지 않는다.

## 미승인 source와 scene별 비용

기존 split 밖16개 source의 accept/repair/exclude/insufficient 판단은 `unready_source_scenes.json`에 분리했다. 이들 입력을 실험에 사용하지 않았다.

과거Train24장면의 렌더링 task8,824개에서 실제 기록을 수집했다. Scene별 중앙값 기준 RIR당 task시간 범위 2.24–4.87초, scene중앙값 2.78초다. 병렬도·재시도·초기화·리소스 경쟁이 섞여 scene크기의 인과 효과로 해석하지 않는다. `generation_cost_audit.json`에 scene별 cell/시간 대조를 위한 원본 경로와 해시를 보존했다. 현재23장면 frozen입력 추출은 scene당 16.7–52.7초, task합 582.9초였다.

## 원본 동일성 검사

Non-Test135 stage mesh의 실제 SHA-256을 계산했다(408,320,056bytes). 완전 동일한 mesh 중복은 없었다. 이는 기하학적 유사성을 배제하지 않으며, source-prefix family 제외 규칙을 함께 유지한다. `stage_identity_audit.json`에 해시와 그룹을 보존한다.

## 48/96 학습 장면 확장의 참고 비용

Teacher 노출은 유지되며 Test/Val source-family를 제외한 추가 non-dense Train77개가 있다. 면적에 따라 고르게 뽑는 예시 계획과 기존 dense cell/area 비율로, 학습24→48에는 추가24개 densebank, 24→96에는 추가72개가 필요하다. 관측 query RIR은 이미 있어 새 FP reference만 거칠게 추정했다. RIR-task 시간 합계 범위는 각각365–819시간(중앙 추정457시간), 1,074–2,408시간(중앙1,345시간)이다. 실제 병렬 walltime/GPU시간이 아니다. Raw float32 저장은 약357–368GiB / 1,049–1,083GiB이며 관측 RIR의 중앙 크기를 대용한 추정이다. 기존 paired-reference와의 overlap 재사용/valid-pose rejection/음향 tail/I/O를 반영한 순증분 견적이 아니다. `future_scaling_cost.json`에 예시장면·가정·식을 보존했다. 이 데이터 생성은 실행하지 않았다.

## 음향 source/receiver 계약의 실제 확인

135개 non-Test scene에서 Observed/FP 각1개 raw record metadata, 총270개를 검사했다. Source는 RGB의 정확한 world XYZ 중심이며6개 microphone은 반경0.05m의 같은 높이 ring이다. 48kHz, diffraction 사용, materials/transmission 비활성 계약을 확인했다. 전체71,467개의 승인 non-Test positive-pair schedule 항목에서 logical yaw → native heading/channel roll 변환도 검사했다. `acoustic_geometry_contract_audit.json`에 record해시와 복원한 microphone XYZ를 보존했다.

Observed는 furnished native geometry, FP acoustic reference는 **canonical2Dmap extrusion**이다. 실제 stage-only mesh와 같은 것으로 취급하지 않는다. 130scene은 semantic region floor/extrusion height를 알고 있다는 가정, 5scene은 기존 사용자가 승인한 실내 semantic 높이 평균을 사용한다. 그러므로 reference mismatch에는 가구뿐 아니라 높이/3D구조 근사도 들어간다. 이 감사는 metadata/계약 검증이며 새로270개 RIR을 렌더링한 것이 아니다.

## 최종 후속 검증 업데이트

전체 연구 완료 fit478회(178 refit+300 inner). 358회 시점의 확대 결과 뒤 OOF 입력 복원108회와 장면 가중치 복원12회를 추가했다. 실제 OOF 출력12,774query-seed 재실행과 원본 GEOQ 점수/최종 선택의 정확한 복원을 확인했다. Query-weight scaling 이득은 기존 scene-weight 복원 후 CI가0을 포함했으므로 데이터 다양성만의 확정 이득으로 해석하지 않는다. `recipe_control_results.json`과 최종 보고서13C가 이 결론을 대체한다. 추가 candidate geometry의 matched 이득은 유지됐다. 원본 Test 관측/결과 및 기존 연구 파일은 변경하지 않았다.

# 완료 감사 — 2026-11-10

원 요청의 연구 종료 조건은 모든 U0–U4 정상/declared shifts와 대조·검증·보고 완료이며, 긍정적인 결과가 필수 조건은 아니다. 본 연구는 완료됐으나 주 가설 지지에는 실패했다.

|요구 사항|확인한 증거 및 범위|
|---|---|
|강한 Study27 기준선과 입력 계약|input_validation.json:6 checkpoint/131 structural·feature 파일 해시,112/19 split,원시 기준선 요약 일치|
|원본 F1 보존|u0_replay_validation.json:14592행/14metrics 불일치0; completed_fit_summary.json:3seed state/file 보존,expert gradient 없음|
|full field 원본 비교의 한계|역사적 전체 score 배열/선택 index는 미저장. 원본 per-query metrics 재현은 전체 cohort, 실제 full-field cache/rebuild/chunk 비교는 runtime_validation.json의1query/15fields로 한정. 두 범위를 혼동하지 않음|
|head-only Train 학습|main_training_protocol.json와3fit:1000steps,7168queries,Train112,4225params/modality,고정 final step|
|U1/U2/U3/U4 대조와수식|ray_uncertainty.py 및5개 uncertainty tests:constant-L1,global 집계,유효 방향 순열,log항의가변weight효과,FoV 배제,양수scale|
|GT/ID 없는 추론과reference 독립|runtime_audit.md:GT/target/ID 제거 후 추론,Python open0건. offline schedule alignment/native I/O 제외 한계 명시|
|원본 입력 변형|3occlusion validation의19scene/1216query/6pilot exact; audio_noise_validation의1216clean preprocessing bitwise replay와SNR/hash 검증|
|전체 7조건·3seed·19scene|final_aggregation_validation.json:동일 cohort와426816rows/117groups 검증. 모든평가 process exit0|
|MAP/strict/raw/diverse/pose reliability|results.md 전체105primary행과 final_summary.json의117행. raw5/10/20,diverse20,nearest-gridNLL,entropy,ECE,selective risk|
|장면 CI 및 negative transfer|final_paired.json994개; final_normal_shift_transfer.json990개. 19scene bootstrap,query/seed 독립표본 취급 안함|
|ray calibration·FoV·거리·방향|final_ray_calibration.json56개 condition/arm/modality 집계,20strata씩. 각16방향,visible/outside,short/long 및50/80/90/95% coverage|
|scale 정책과cost|final_secondary_scale_policy.json132개 보조 비교, final_field_scales.json, runtime_validation.json. timing은1query cached-map 제한측정이며 end-to-end와조건별대표latency는NA|
|yaw/GT outside/boundary/ties|test_extended_metrics.py:wrap,GT99m,confidence독립,tiedrank; test_rays.py:mapexitcensoring; test_structure.py:rasterboundary허용오차; uncertainty constant-L1 exacttie 및CPU/GPU tolerance|
|최종 회귀 검증|GPU 가시 컨테이너에서 track_b/tests 전체51passed,기존autocast경고2. CPU-only 확인은50passed/1CUDA skipped. skipped를통과로세지않음|
|실패·부정·정정 보존|results.md:주 가설 음성결과,초기constant-scale 테스트실패,미완료집계거부,aggregate_v1 codeSHA오입력과v2수치동일성 정정|
|Test/Val 선택 금지·일반화 한계|protocol과평가 code에Test미사용,고정head/b/temperature/conditions. reusedVal/upstream노출,in-sample residual,physicalreflection불일치명시|
|재현·registry·hash|reproduction.md,experiment_registry.csv,provenance.json,final_source_hashes.json. 대형데이터/weights/rawpredictions는외부root|
|최종 결정과후속 제안|decisions.md:U0유지,U3기본채택안함,posterior mixture vs product 한가지 제안만하고실행안함|

배포 성능·독립 Test·실환경 음향 잡음·AVLoc 원문 비교·material/furniture shift는 이 증거로 검증되지 않았다. 이 NA/범위 한계는 성공으로 바꾸지 않는다. Docs Commit SHA는 이 문서에 삽입하지 않고 커밋 후 최종 응답에서 보고한다.

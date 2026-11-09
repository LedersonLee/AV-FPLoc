# 사전 감사

관찰: 체크포인트6개,131개 structural cache와131개 upstream feature 파일, 대응 입력 manifest hash가 기존 평가 protocol과 모두 일치했다. 모든 평가14,592개 method/seed/query key는 유일하며, 원시 per-query 지표 평균이 저장 summary와 일치한다. 실제 F1 Joint34.402412%, Visual21.847588%, Acoustic7.044956%, F1 Diverse20 89.665570%. 이는 기존 결과의 파일 일관성 확인이며 새로운 재현 결과가 아니다.

기존9개 확장 interaction 및6개 OOF fit 프로세스는 exit0/summary9·3·3으로 종료했음을 확인했다. 새 목표 이전 결과는 덮어쓰지 않았다. 기존 Track A dirty 파일2개를 보존했다. 작업 branch/remote는 지정값이며 시작 시 원격과 차이가 없었다.

데이터 계약은 입력 검증 JSON에 고정했다. ray target은 log1p(range)/log(21),20m cap; 후보 FP descriptor에는 학습된 residual이 있어 단순 원시 ray와 동일하지 않다. 이를 모든 U1–U4에서 고정해 공유한다. 후보 weight는 .25+.75*hit이며 outside-FoV는0이다. GT target_hit는 기존 보조 loss에서 제외 mask로 쓰지 않았으므로 이번에도 별도 hard mask로 변경하지 않는다. FP black은 wall/exterior/unknown 혼합이며, acoustic reflection GT가 아니다. 실제 sensor/body alignment는 기존 offline schedule 사용이고 추론 head에는 pose/world yaw/ID/GT를 주지 않는다.

해석: 입력 경로를 임의 대체할 이유가 없다. 불확실성 유용성은 아직 미검증이며 정상 Train residual은 base expert의 in-sample 잔차라 낙관 편향 가능성이 있다. severe OOD scale 실패도 가능한 결과다.

제한: 이번 CPU 감사는 cache/feature/checkpoint hash와 원시 평가 지표 일관성이다. 모든 원본 RGB/RIR 바이트를 새로 재해시한 감사나 모델 재추론을 대신하지 않는다. 새 U0 평가가 진행 중이며, 역사적 full score tensor/선택 index가 보존되지 않은 범위에서 bitwise 원본 field equality를 주장하지 않는다. 재현 지표와 새 cached/uncached score 동치 검사를 구분해야 한다.

참조 확인: 사용자 지정 OpenReview forum/PDF를 모두 요청했으나 접근 오류, 해당 ID 검색도 결과가 없었다. AVLoc 세부 설명은 사용자 제공 설계 배경으로만 취급하며 전문 검증/우월성 주장을 하지 않는다. 이는 제공된 수식의 로컬 실험을 막는 데이터 장애는 아니다.

검증 코드 SHA:96db20f075ded8f5c66234496c7b2dfbfb3741cf. 같은 checkout에서 U0 evaluator 실행 시작(handle93759). 다음 gate는19장면×3seed 전부 원래 MAP/raw/diverse20/NLL 등 지표와 대조하고, 별도 score 동치 및 head tests를 통과하는 것이다.

후속 확인: U0 실행 exit0. 14,592행/15지표의 원본 대응 재현에서 불일치0. 학습 시작 전 남은 것은 신규 scorer/head 구현 테스트와 Train-only smoke다.

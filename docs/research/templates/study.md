# 연구 제목

- Study ID / 날짜 / Track / 상태(성공·실패·부정·중단):
- 증거 수준(Exploratory / Development-validated / Independent-validated) 및 근거:
- 선행 연구와 변경한 가설:
- 실제 실험 코드 Commit SHA / dirty 상태·패치 또는 외부 source SHA256 / 실행 환경:
- Docs Commit SHA는 이 템플릿/Index에 자체 삽입하지 않고 최종 실행 보고에 기록:

## 질문과 경쟁 설명

어떤 관측이 각 가설을 구분하는가? 성공/기각 조건과 아직 모르는 것은 무엇인가?

## 프로토콜

Scene IDs, query 수, upstream 노출, split, seed, 입력/후보/목표/parameter budget, selection/threshold, 평가 지표와 통계를 기록한다. 최초 사전 설정과 결과를 본 뒤 추가한 후속 실험을 구분한다.

## 결과

| 대조 | Cohort와 평균 방식 | 기준 | 개입 | 차이와 CI | 시드별 결과 | 판정 |
|---|---|---|---|---|---|---|

모듈 지표와 최종 pose, repair와 harm, 실패/부정 결과를 함께 보존한다.

## 해석과 한계

관측 사실 / 가능한 설명 / 개입으로 지지된 원인 / 일반화 주장을 구분한다. 누출·평가 재사용·다중 탐색·유효 capacity 차이를 기록한다.

## 후속 판단과 연구자가 얻는 원리

무엇을 채택·기각·보류했는가? 어떤 조건에서 실패하는가? 다른 설정에서 확인할 수 있는 예측은 무엇인가? 이 문서 작성 자체는 후속 실행 승인이 아니다.

## 재현과 증거

설정·명령·checkpoint/입력 hash·외부 artifact 위치·실패/정정 로그를 연결한다. clean-clone만으로 재현 불가능하면 필요한 외부 의존성을 명시한다.

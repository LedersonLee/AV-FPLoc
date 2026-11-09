# Decisions / preregistered follow-up logic

1. **독립 Track B**: 새 track_b폴더와 기존 승인 non-main브랜치 사용. B2/FP-RIR reference입력 없음. 원본 Track A/사용자 변경 보존.
2. **단계0 audit 우선**: Test는 administrative metadata만. Official split 유지, 동일 source-family 제외. Near-duplicate 미완료를 명시하고 Exploratory로 제한.
3. **작은 feasibility 먼저**: 0.25m/36 audit와 별도로 0.5m/12 pilot, RGB64×48, FP0.1m, d32, raw6ch8192samples. 계산량과 기본학습 실패를 빠르게 확인하기 위한 축소이며 원 프로토콜 full-resolution 결과를 대체하지 않는다. Chunk512, float32, full-bank CE.
4. **120→2000steps**: 초기15fits에서 Train도 거의 random 수준. 동일 초기값·sample order·모델·데이터를 유지하고 학습 길이만 늘려 underfit 설명을 검사. Test를 보거나 bestepoch를 선택하지 않음.
5. **Local patch→long-range geometry**: FP query의 대규모 근사중복이 관찰돼 long-range boundary ray의 구별력을 진단. 7pilot scenes의 ray signatures는 모두 서로 달랐다. 이에 acoustic/joint각3seed·2000steps, rays 대 동일parameter zero-input 비교를 추가. 기하 입력은 RIR합성이나 echo-time 변환을 하지 않음.
6. **용량 통제 한계**: local5arm은 전체 allocated52,324parameters이나 active path가 다름. 엄밀 F3가 아님. Ray-vs-zero는 같은 layer수/초기화/학습순서로 input information을 통제하지만 useful activecapacity까지 같다는 주장은 하지 않음.
7. **수치 정정**: audit_v1의 ≤0.1 strictfloat경계 문제를 발견해1e-6m허용오차 upperbound를 별도 저장. 최초 기록 삭제 없음.
8. **후단 보류는 과학적 gate**: retrieveroracle가 낮으면 discriminator보다 dense representation을 먼저 수정한다. 이는 전체 연구 완료/종료가 아니며 사용자의 재승인을 요구하지 않는다.

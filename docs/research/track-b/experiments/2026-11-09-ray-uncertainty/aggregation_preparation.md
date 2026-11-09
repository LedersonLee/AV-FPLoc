# 전체 조건 집계 준비와 검증

**완료: 집계 구현·51개 테스트·미완료 결과 거부 검사. 미완료: 전체7조건 결과의 실제 최종 집계.** 집계 source `8cef748f29a4b424119cdbf6056270096b713764`(초기 구현93b2c147aaade0a56179b256f81201d90cfc8309).

모든7조건의 protocol/predictions/ray records/summary/validation/timing/field spreads가 있어야 실행한다.19개장면·3개seed·64query, 동일한모든query keys와head/선언protocol hash, primary/secondary 정책 집합, 고유행수를 검사한다. 정상98,496행과각열화54,720행을 요구한다. 현재의 불완전한 실제 출력을 입력했을 때 `Await complete declared evaluations`로 거부하고 집계 폴더를 만들지 않음을 확인했다([검증](aggregation_gate_validation.json)). 예상 실패 traceback은 외부 `aggregation_gate_validation_v1/expected_failure.txt`에 보존했다.

주요 비교는 조건별V/A/fusion의U3−U1/U2/U4/U0, 보조 비교는U3 fusion−U0 acoustic 및U3 fusion−U3 acoustic이다.19개장면 안의query/seed 평균을 만든 후10,000번 scene bootstrap한다. 동일query 효과를2개에서20개로 복제해도 신뢰구간이 달라지지 않는 합성검사를 통과했다. 중복/누락query는 거부한다. CI는다중비교 보정하지 않았으며 exploratory로 보고한다.

Ray 진단은각방향,visible/outside-FoV,5m이하/초과 범위의MAE/Laplace NLL/mean scale/scale-error Pearson/50·80·90·95% coverage/동일표본수scale bin을 산출한다.5m는 고정된해석용거리구간이며성과로선택하지않는다. 시각FoV밖과해당방향의수치는 likelihood에사용되지않는 unsupported 진단임을 표시한다. U1의상수scale은상관계수 NA로둔다. PoseECE/selective risk는기존검증된함수를쓴다.

첫새단위검사에서 상수float32 scale의std에반올림잔차가남아 Pearson을계산해NaN이발생했다(1passed/1failed). 상수판별을표준편차대신max−min 범위로수정했다. 재검증은전체51passed,기존autocast경고2개,1.96초. 실패를성공으로소급표시하지않으며변경은집계에만적용했다. 실행중모델/평가코드는수정하지않았다.

재현: avfploc:cu124-clean, `python -m pytest -q -p no:cacheprovider track_b/tests`; 모든조건완료후 `python -m track_b.scripts.summarize_ray_uncertainty --root <study30-root> --out <new-output> --code-sha 8cef748f29a4b424119cdbf6056270096b713764`. 현재어떤효과/contrast도최종확정하지않았다.


정상 대비 열화 변화(각 U0–U4 및 V/A/fixed)와 정상 조건 original-scales 대비 primary 차이를 기존 장면 paired bootstrap으로 집계한다. 결과에 따라 보정값을 선택하지 않는다. 관련 분석 단위/constant-scale 검사 2개 통과. 집계 source: `652dbbcbd12f10e69af6ac5e92feeec667f2ca34`.

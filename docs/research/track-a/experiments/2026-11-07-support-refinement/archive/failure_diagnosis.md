> 역사적 연구 기록. 원본: `astra_fusion_v2_20261107_0225/failure_diagnosis.md`. 원본 SHA256은 [산출물 목록](../artifacts.json)에 있습니다.
> 당시의 완료·재개 지시는 현재 실행 요청이 아닙니다. 최신 판정은 [현재 결론](../../../../current_findings.md)을 먼저 확인하세요.

# 실패 진단과 가설 업데이트

- 재개출발점: [이전실패진단](../../2026-11-07-fusion-audit/archive/failure_diagnosis.md), [이전실험로그](../artifacts.json) (외부 산출물: `../astra_fusion_20261107_0200/experiment_log.csv`). 이전단순가중치/entropy/yaw실패를반복하지않고더강한B2와Train OOF를사용했다.
- 원인 G(비교confound): 이전원래branch와최근적응branch를구분. B2 checkpoint/index재현으로공정비교를확보했다.
- 원인 C(gate): 절대risk선형/비선형,기준대비gain회귀모두Train개선이Val로안전이됐다. 이미있는`reliability_fusion/case_crossfit_policy.py`/generalization_audit의비슷한위험도검토했으며새B2기준에서도비슷한실패를측정했다. 깊은gate를더학습하는것을자동해결책으로취급하지않았다.
- 원인 F(posterior geometry): 고립된point MAP 대신주변pose의확률질량으로영역을선택하면실제개선. XY-only/yaw-only/degree-normalized대조가이해석을지지한다. 지지도최대점을그대로출력하는것과영역내원래peak로정밀화하는것의차이를별도로검증해R.1손실을완화했다.
- 원인 B(calibration): T는Train정확도로선택됐고확률calibration으로fit하지않았다. support 성공영역NLL은좋아져도pointNLL/ECE는악화. score기하개선과calibration개선을구분한다.
- 원인 A/D/E: audio셔플의큰퇴행은기존음향의대응정보기여를보이지만표현손실/interaction최적구조/가구반사원인을완전히판정하지않는다. 실제Q/K/V direct attention과B2 latent교환을구분해감사했다.

## 최종 outcome subgroup (2seed 평균)

| 구간 | query/seed | 기준 joint | 최종 joint | fixed대비복구/악화 |
|---|---:|---:|---:|---:|
| both | 641.5 | 98.207% | 98.051% | 1.5/2.5 |
| vision_only | 1445.5 | 90.453% | 90.661% | 26.5/23.5 |
| audio_only | 458.0 | 75.873% | 79.039% | 30.5/16.0 |
| neither | 1645.0 | 41.976% | 45.015% | 108.0/58.0 |

- 이구간은GT를사용한사후분석이며실제조명/소음degradation label이아니다. 학습/추론입력에그구간값을사용하지않는다.
- 원래시각성공의조건부성공차이+.096%p,95%[-.595,+.629]. 일부scene에서신뢰시각훼손이증가했다. 모든scene/모든query에대한보존보장을주장하지않는다.
- 사례는각유형의첫해당query를같은규칙으로선택: seed42/scene108736737_177263406에서복구sample320,337,357; 악화11,62,89; 두모달리티단독+기존융합모두실패에서복구337,358,655. [case file](../artifacts.json) (외부 산출물: `final_cases.json`)에GT와모든pose/index/error를저장했다.
- 각반복의Observation/Hypothesis/Intervention/Outcome/Remaining uncertainty는[iterations.md](iterations.md)에있다. 상세per-query NLL/rank/entropy는`controls_seed42/samples.json`, `controls_seed43/samples.json`이고최종예측은[final_samples.json](../artifacts.json) (외부 산출물: `final_samples.json`)이다.

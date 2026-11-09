# Scale head 및 likelihood 구현 검증

**완료: 분석 테스트5개, 전체49개 테스트 통과. 미완료: 실제 checkpoint 관측의 학습 smoke와 모든 U1–U4 성능 평가.** 소스 `5f0fb177cce1b2726f67f9e6f3ad65a6e733ae73`.

사전 protocol대로 관측 hidden128 + 방향 sin/cos를 받아 shared130→32→1 head가16개 scale을 만든다. modality당4,225개 파라미터이며 floor.001/ceiling1/초기.1을 사용한다. 기존 predictor를 eval/frozen 상태로 요구하고 hidden/mean 추출은 no_grad다. GT/scene/query ID/candidate pose는 inference 인자에 없다. 원래 F1 model은 변경하지 않았다.

검증 관찰: frozen mean이 원래 forward와 bitwise 같고, loss 역전파는 scale head에만 도달했다. constant-b에서 L1 analytical field/순위가 맞고, equidistant tie는 유지됐다. 후보 chunk1/full 계산은 동일했다. FoV 밖 mean/FP/scale을 바꿔도 visual field가 동일했다. U4가 FoV 안 scale의 multiset을 보존하면서 field를 바꾸고, U2는 유효 방향 평균을 broadcast했다. 양수가 아닌 scale/모든 weight0 입력은 거부한다. CPU/GPU 비교는 rtol1e-5,atol2e-5로 통과했다.

해석: 후보 weight가 일정하면 log(2b)의 방향 평균은 후보 공통항이다. weight가 후보마다 달라지면 이 항만으로도 차이가 생기는 분석 예제가 확인됐다. 따라서 구현은 해당 항을 제거하지 않는다. 방향 permutation이 field를 바꾸는 것은 정보 유용성의 증거가 아니다.

Train112 전부의7,168query/114,688개 normalized target을 CPU에서 읽어 최소0.0598851,최대1.0,유한값을 확인했다. target_hit true는114,633개이고, 나머지는 기존 loss 정책대로 별도 hard exclusion하지 않는다. [범위 기록](train_range_validation.json). 이 검사는 Train target 범위이며 residual calibration을 뜻하지 않는다. 사전 scale 범위를 Val로 조정하지 않았다.

재현 환경 avfploc:cu124-clean,OMP_NUM_THREADS2,repo read-only mount,CUDA 테스트 가능. `python -m pytest -q -p no:cacheprovider track_b/tests/test_ray_uncertainty.py`:5passed0.76s. 전체 `track_b/tests`:49passed1.92s, 기존 autocast deprecation warning2개. GPU fit은 미실행이다. 기존 전체U0 metric replay와 합쳐도 실제 scorer runtime I/O trace,실제 cached/uncached full field,shift별 효과까지 입증한 것은 아니다.

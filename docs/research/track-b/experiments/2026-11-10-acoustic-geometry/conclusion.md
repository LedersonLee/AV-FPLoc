# 최종 연구 결론 — Learned Acoustic Structural Geometry

2026-11-10, Study31, Track B. **실험·검증·산출물 작성 완료. 주 가설의 강한 형태는 지지되지 않았다.** 학습된 RIR 표현에는 관측 위치에 따른 거리 규모 정보가 있지만, 검증한 모델에서 방향별 구조 복원과 geometry 사전학습의 localization 개선은 입증되지 않았다. 모든 실험은 결과의 성공 여부와 별개로 기록했다.

## 관찰 사실

95개 train 장면/94개 source family,17개 inner-val 장면/16개 family를 사용해 Echo-style/Spatial × 회귀/확률 head ×3seed의12개 모델과 phase-zero6개 모델을 학습했다. 별도의21개 전이 fit에는 공유 visual3개와 acoustic6조건×3seed가 포함된다.39개 체크포인트를 고정한 뒤 과거 개발 holdout19개와 공식 Test17개 장면을 평가했다. 공식 Test는 각 장면64개 pose의 furnished/stage-only 관측 및 첫 pose 반복 관측, 총2,193개 RIR을 새로 생성했다.

[공식 Test 표](official_results.md)의 furnished 조건에서 Echo/Spatial 회귀 MAE는1.535/1.567m이다. Spatial 회귀는 train-only median보다.164m 개선됐고, 동일 장면의 다른 위치 RIR로 교체하면 MAE가.263m 악화됐다(95% family CI [.223,.307]). 반면 예측의 방향을 평균으로 대체한 MAE 변화는+.00030m [−.00083,.00156],180° 회전은+.00224m [−.00002,.00475]에 그쳤다. 관측 내 방향 상관은.0159이다. Phase 제거는+.05295m [.04097,.06659] 악화를 보였다. 위치별 거리 규모에 대한 감도와 방향별 형상 재구성을 구분해야 한다.

확률 head는80개 거리 bin의 여러 가설을 표현하도록 구현됐지만, 이것이 물리적 반사 경로나 방향 구조를 식별했다는 뜻은 아니다. Furnished Test의90% 구간 coverage는 Echo.925/Spatial.945이며 폭은6.965/7.449m이다. 넓은 구간으로 얻은 coverage를 정밀한 구조 예측으로 해석하지 않는다. NLL/CRPS와 실패 heatmap을 함께 제공한다.

| Furnished Test 전이 조건 | Fixed-fusion joint1m30° | Fine0.1m |
|---|---:|---:|
| 기존 acoustic branch |34.835%|3.064%|
| 동일 용량 scratch,local1000 |25.613%|1.471%|
| 동일 용량 scratch,total2500 |29.075%|2.022%|
| Geometry frozen |23.468%|1.379%|
| Geometry fine-tuned |25.766%|1.532%|
| Geometry joint auxiliary |25.153%|1.501%|

공유 visual-only joint는23.070%다. Fine-tuned−scratch 차이는+.153pp [−.797,1.134]로 이득이 확인되지 않았다. Frozen−scratch는−2.145pp [−3.615,−.674]다. 총 update를 맞춘 scratch 대비 frozen/fine-tuned/joint는 각각−5.607/−3.309/−3.922pp이며 세 구간 모두 음수다. Legacy는 더 큰 acoustic branch이므로 pretraining 자체의 해석에는 동일 용량 scratch 대조가 더 직접적이다. Encoder만 교체했고 기존 map/candidate/matching 및.85/.15 fusion은 유지했다. 업데이트 수와 FLOPs의 동일성을 혼동하지 않는다.

## 물리 대조와 경쟁 설명

전체 scene object를 제거한 stage-only 입력은 문·창문까지 제거한다. 공식 Test Spatial 회귀에서 stage-only−furnished MAE는+.0899m [.0582,.1256]이지만 이를 순수 가구 효과로 부를 수 없다. 별도의16개 개발 family/128pose에서 문·창문 template의 원래 transform을 보존한 대조를 추가했다. 이때 Spatial 회귀 furnished−opening-preserved 차이는+.0119m [−.0115,.0353]이다. 제거 대상에 고정 fixture가 포함될 수 있고 반복 렌더도 비결정적이므로 furniture invariance나 동등성을 주장하지 않는다. [대응쌍 결과](pair_results.md).

**해석:** 현재 결과는 방향 구조 표현보다 room/range-scale prior와 관측별 규모 변화에 대한 감도에 더 부합한다. **미검증 인과 가설:** source-centred 작은 array의 식별 한계, 마스킹된 GT/fixture 불일치, frontend의 방향 정보 활용 부족, 모델·학습 예산 등이 원인일 수 있다. 이번 대조만으로 특정 원인을 확정하지 않는다. 음향으로 유일하게 식별되지 않는 geometry의 가능성을 인정하며, 모든 encoder가 실패한다는 일반화도 하지 않는다.

## 후속 Geometry-conditioned Cross-modal Matching 판단

현재 encoder를 고정한 채 matching 구조만 복잡하게 만드는 것이 필요하거나 충분하다는 근거는 없다. 이번 결과가 보여준 직접적인 문제는 공간 방향 구조가 거의 재구성되지 않았고, 전이에도 명확한 이득이 없다는 점이다. 후속 연구를 시작한다면 (1) 실제 방향 cue를 사용하는지 회전·scalarization·다른 위치 donor 대조로 먼저 입증하고, (2) 방향 불확실성을 후보 geometry별 점수로 연결할 때 동일 용량·예산 대조와 fine/pure-visual 저하를 함께 평가해야 한다. 현재 Test 결과를 이용한 추가 구조 선택에는 새 개발/평가 분리가 필요하다. 이 후속 모델 탐색은 완료된 초기 실험의 성공으로 소급하지 않는다.

## 검증 범위와 일반화 한계

[공식 입력 검증](official_input_verification.json):17개 장면,2,193개 원시 파일의 hash와 조건별 개수를 검증했다. [공식 결과 검산](official_verification.json):39개 frozen checkpoint,17개 cache,1,224개 표현 scene/condition 기록,195,840개 localization 질의 기록을 확인했다. 모든 acoustic 조건과 arm에서 공유 visual 예측은 같다. 개발 holdout19개도 별도로 검산했고, target/model/transfer/opening 단위검사7개가 통과했다.

Structural GT는 receiver 높이에서 native stage의 수직면에 교차하는 거리이며 visible depth·음향 전파 거리가 아니다. No-hit/>20m/외곽 경계 불일치를 mask한다. 따라서 결과는 남은 유효 방향에 조건부이며 완전한360° 구조 정확도가 아니다. 기본 재질/전달 비활성화 시뮬레이션, furnished navigability로 선택된 pose,5cm6-mic array와 centre source에 한정된다. 과거 개발셋 재사용, 일부 공식 Test 관측/visual 평가 이력, upstream Hypersim asset-family 노출을 완전히 배제하지 못하므로 증거 수준은 **Exploratory**로 유지한다. 새 RIR 생성은 과거 장면 노출을 지우지 않는다.

실측 추론은2개 사전 지정 개발 장면,seed42의6개 arm에서 decoded RGB+body RIR→full field+Top20을 검증했다. 전체 feature/field/Top20 재생은 bitwise 일치한다. Median72.4–75.5ms에는 backbone/STFT가 포함되지만 I/O·모델 로드·map 준비는 제외되며 공유 GPU 측정이다. 별도 CPU/GPU paired 평가의 최대 예측 차이1.58mm도 기록했다. 전체 공식 score field의 독립 재생을 주장하지 않는다.

## 재현·기록

[재현 명령](reproduction.md), [요구사항 감사](completion_audit.md), [방법 고정 및 checkpoint hashes](method_freeze.json), [전체 공식 집계](official_summary.json), [표현 대조 CI](official_contrasts.json), [RGB/GT/예측·heatmap](results.md). 실제 학습·렌더링·평가 code SHA는 각 registry/protocol 및 단일 RESEARCH_INDEX.md에 기록했다. 실패한 target split/GT 준비, 실행 환경 smoke, 엄격 CPU/GPU parity gate는 보존했다. 원시 데이터·예측·로그·가중치는 외부 실험 폴더에 남기고 작은 집계와 config/hash만 Git에 저장했다. 기존 사용자 수정2개는 포함하지 않았다. Docs commit SHA와 push 상태는 commit 이후 최종 실행 응답에 기록한다.

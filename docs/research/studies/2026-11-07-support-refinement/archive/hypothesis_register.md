> 역사적 연구 기록. 원본: `astra_fusion_v2_20261107_0225/hypothesis_register.md`. 원본 SHA256은 [산출물 목록](../artifacts.json)에 있습니다.
> 당시의 완료·재개 지시는 현재 실행 요청이 아닙니다. 최신 판정은 [현재 결론](../../../current_findings.md)을 먼저 확인하세요.

# 가설 등록 v2 — 결과 이전

- 이전 실패: global audio suppression, naive entropy gating, yaw marginalization. 정보 있는 음향을 더 적극적으로 이용하되 reliable visual localization을 보존해야 한다.
- 이전 audit를 재사용: `../astra_fusion_20261107_0200/fusion_system_audit.md`. 이번에는 역사적 Val8 strongest fixed baseline PRIOR_B2 epoch8, acoustic w=.15를 실제 checkpoint 재실행한다. Train12에서는 기존4fold held-scene checkpoint, Val3에서는 fullTrain checkpoint. seed42 먼저, 유망하면43과 추가Val5를 확인한다. 모든 inference features에 GT/sceneID 없음.

## H5: 후보별 성공위험은 단순한 선형 점수관계로 판별할 수 있다

- 근거: 기존 oracle-menu 여유9.10%p, 과거 B2 grid oracle와fixed의 격차~11.9%p. 단독 branch 성공여부만 선택하면 둘 다 실패하지만 융합되는 경우를 놓친다.
- 개입: 사전8개 weight [0,.05,.1,.15,.2,.3,.5,1]의 full-bank MAP 후보마다22개 score/peak separation/scale/depth 특성을 계산. Train OOF 특징·GT success label로 logistic regression C1 학습. 후보 성공 확률로 선택, 변화 확률차 threshold는Train scene4fold만으로 결정.
- 반증: Train holdout/Val macro joint,R1 개선 없거나 reliable-vision harm 증가. Oracle 여유가 입력으로 식별 가능하다는 가정을 반증할 수 있다.
- 대조: 동일 B2 fixed.15, fixed.1, vision, acoustic. 원래 adaptive58/최근 CNNcross는 참고 baseline으로 별도 비교.
- 비용: 캐시 준비 GPU 수분/추가 VRAM<11GB cap, 작은 gate CPU 수십초. seed42 exploratory,43 confirmation.

## H6: 후보별 비선형 score geometry가 선형 게이트의 병목이다

- 개입: H5의 모든 input/proposal/label/Train selection 동일, classifier만 HistGradientBoosting(max_iter100, leaves7,min_leaf100,L2=10)로 교체. parameter/compute가 동일하다는 주장 없음; 분류기 비교.
- 예측: held-scene에서 선형 모델보다 개선하고 visual harm 유지. 반증: in-sample만 좋아지거나 validation 성능/보존 기준 미달.
- 과적합 위험:12Train scenes, pretrained encoder 자체는nested OOF가 아님. Gate cross-validation도 upstream 모델까지 nested하지 않으므로 TrainCV를 독립 일반화 성능으로 보지 않는다. Val은 전체Train과 scene-disjoint.

## H7: 점별 MAP가 성공영역의 posterior geometry와 맞지 않는다

- H5/H6가 실패하면 후속: full-bank에서 지역 후보 지지도를 사용하는 deterministic fusion 또는 후보의 geometric support 특징으로 한 번에 한 요소 추가.
- 반증: 높은 local support가 잘못된 평면도 대칭을 강화하거나 reliable-vision harm가 늘면 기각.
- GT-free geometry와 Train-only 온도/threshold 선택 유지. 기존 후보/metric 정의 변경 없음. label에 맞춘 후보 생성/GT injection 금지.

## 진행/확인 기준

- 고정 실험 횟수 중단 없음. 실패 원인에 따라 다음 가설로 진행. Train-only model/hyperparameter 선택을 우선하며 Val3 search는 기록한다. 추가Val5는 candidate freeze 뒤확인하고 결과 본 뒤재선택하면 이를개발사용으로 명시.
- 유망: matched baseline보다 micro/macro R1 및joint 양호하고 reliable-vision harm이 악화되지 않음. 확인: seed42/43 방향 일치, scene별/cluster uncertainty, 음향 정보 활용 대조, degradation subgroup 개선, 기존 strongest와 비교.
- 로컬 자원 자유 사용 승인받음(사용자 답변: “you can use local resource freely.”). 기존 GPU 프로세스 유지, 유료/외부업로드/새 dataset generation 없음. 각 실행 시간·메모리 기록.

## H8: 절대위험보다 기준 대비 기대 개선을 직접 학습해야 한다

- 관찰: H5/H6 Train OOF joint macro +.62/+.85%p, Val3 -5/-2 query. tree가 시각 harm41→22로 줄였으나 rescue도 감소.
- 개입: 동일8후보, tree 용량/epoch/Trainfold/threshold grid 유지. 후보 features + baseline과의feature차이44D→success(candidate)-success(fixed15) 회귀. 같은 pose 후보는gain0 강제. GT는Train target뿐.
- 예측: 불확실한 절대성공 예측보다기존성공훼손을직접벌점으로다뤄순개선. 반증: Val macro/전체joint 악화 또는신뢰시각악화. 비용CPU수초. 성공시absolute risk에같은paircontext를주는대조로target와features효과분리.

## H7 실행 사양: 후보 공간의 국소 지지도

- H5/H6/H8 결과후pivot: query-level가중치선택이Train에선좋지만Val에서개선하지못함. 같은scalar 후보메뉴의선택만으로는국소posterior geometry를놓칠수있다.
- 새개입: fixed.15 전체pose score를 exp((score-max)/T)로변환하고, XY반경.4m 안의기존후보와yaw±10°를합산. 이local mass가최대인기존후보를출력. 후보추가/GT주입/평가지표변경없음.
- T=[.05,.1,.25] 3개는Train scene macrojoint/R1로선택, Val로선택하지않음. 커널범위고정. 같은지원계산을vision에도적용해일반postprocessing이득과fusion효과구분.
- 예상CPU sparse-neighbor sum수분, GPU추가학습없음. 실패조건: selectedT가Val주지표퇴행/reliable-vision harm악화. 양성이면다른seed·추가Val·support-normalization/vision-only대조확인. 학습된reliability 증거로포장하지않음.

## H9: 지지도 최대점을 그대로 출력하면 basin 내부 정밀도가 손실된다

- 관찰: support 방식은seed평균8scene joint+1.766%p, 모든scene 개선, reliable-vision harm149.5→144.5지만 R.1−2.446%p.
- 개입: T=.05/r=.4/yaw10 고정. support 최고후보 주변 동일.4m/10deg 영역 안에서원래fixed15 score 최대후보로정밀화. 새후보/재학습/GT 없음.
- 예측: basin선택이득을유지하면서 R.1복구. 반증: joint 이득이소멸하거나harm증가/정밀도미복구.
- 먼저Train OOF seed42 평가. Val8은지금부터postprocessing확인에재사용됨을명시. 새로운독립test로주장하지않음. 같은규칙을seed43에서확인. 추가튜닝없음.
- 비용: cached adapted output 재사용, 수분이하. 기존support와originalMAP대조.

## 최종 상태

| 가설 | 관찰된 결과 | 판단 |
|---|---|---|
| H5 선형 후보 위험 | Val3 joint -0.293%p | 채택하지 않음 |
| H6 비선형 후보 위험 | Val3 joint -0.117%p | 채택하지 않음 |
| H8 기준 대비 이득 회귀 | Val3 joint -0.763%p | 채택하지 않음 |
| H7 국소 posterior 지지도 | Val8·2seed joint +1.766%p, R.1 -2.446%p | 주요 개선 재현, 정밀도 한계 발견 |
| H9 영역 내부 정밀화 | Val8·2seed joint +1.587%p, R.1 +0.143%p | 최종 권고 후보; 통계와 제한은 결과 요약 참고 |

- 실패한 선택기에서 멈추지 않고 다른 가설로 전환했으며, 발견한 부작용도 후속 개입으로 검증했다.
- 최종 음향 셔플 대조는 joint30.418%, 최종 시각 단독 결정 방식은49.905%였다. 최종 융합72.601%는 음향 전역 억제로 얻은 결과가 아니다.
- Test 일반화, 엄격한 시각 비열등성, 학습형 adaptive gate는 여전히 미검증이다. 이번 종료 근거는 재현된 validation 개선이며, 이 열린 문제들이 해결됐다는 뜻은 아니다.

> 역사적 연구 기록. 원본: `hypothesis_discrimination_20261107_1538/RESEARCH_REPORT.md`. 원본 SHA256은 [산출물 목록](artifacts.json)에 있습니다.
> 당시의 완료·재개 지시는 현재 실행 요청이 아닙니다. 최신 판정은 [현재 결론](../../../current_findings.md)을 먼저 확인하세요.

# Hypothesis Extraction & Discrimination — 결과

## 확인한 개선과 범위

같은20개 support/refinement 후보를 유지하면서 후보의 **첫 후보 대비 utility 이득**을 학습해 Top-1을 개선했다. Utility는 joint_success + 0.5×fine_success이며, 두 seed 모두 이 가중치를 Train 장면 OOF에서 선택했다. 후보 coverage 증가를 학습형 ranking 개선으로 혼동하지 않았다.

| 재사용 개발 집합 | 관측 수 | 기존 support/refine joint | 학습 ranking joint | 변화와 장면95% CI |
|---|---:|---:|---:|---|
| Val8 | 4190 | 72.601% | 73.496% | +0.895pp [+0.682pp, +1.134pp] |
| Val11 | 5775 | 70.078% | 70.675% | +0.597pp [+0.325pp, +0.867pp] |

**일반화 성능 주장이 아니다.** Val8·Val11은 이전 연구에서 이미 사용된 자료다. 새 validation 데이터를 확보하지 않았으며 이번 결과는 개발 자료 내 재현과 통제 실험이다. 모델 학습·임계값·utility 가중치 선택은 Train12 장면4fold OOF에서 수행했지만, 연구 방향 자체는 반복된 개발 결과의 영향을 받았다. 따라서 bootstrap CI도 선택 편향을 제거한 확증적 구간이 아니다. Test 결과는 열지 않았다.

## 후보 추출과 coverage

원래 B2 epoch8 점수와 full-bank 후보를 고정했다. 세 추출기는 score NMS, Euclidean support NMS, connectivity-aware support NMS다. Support는 T=.05 점수 질량의0.4m/±10° 집계이며 NMS는1m/30° 이웃을 억제한다. 각 중심은 해당0.4m/10° 이웃의 원래 fusion score 최대점으로 정밀화한다. 정답은 후보 추출에 사용하지 않는다. 모든 후보는 기존의 알려진 scene 내 candidate bank에서 선택한다.
후보가 서로 다른지 보장하면서 기존 bank와 비교할 수 있어 NMS를 첫 통제 실험으로 택했다. KDE/mean-shift/clustering으로 연속 pose를 새로 만드는 실험은 수행하지 않았다. 이번 자료에서는 약96% coverage에 비해 Top-1이 낮으므로 우선 같은 후보의 discrimination을 검증했다. NMS 밖의 추출기가 불필요하다고 결론내리지는 않는다.

| 집합/추출 | Top1 포함률 | Top5 | Top10 | Top20 |
|---|---:|---:|---:|---:|
| val/score_nms | 71.014% | 90.919% | 94.189% | 95.871% |
| val/support_nms | 72.601% | 91.301% | 94.272% | 95.990% |
| val/connected_nms | 72.601% | 91.301% | 94.272% | 95.990% |
| extra/score_nms | 68.078% | 91.091% | 95.351% | 97.506% |
| extra/support_nms | 70.078% | 91.922% | 95.307% | 97.342% |
| extra/connected_nms | 70.078% | 91.913% | 95.307% | 97.351% |

Oracle은 GT로 성공 후보의 존재를 사후 판정한 값이다. 실제 모델의 Top-1 정확도와 다르며96%를 달성 가능한 성능으로 주장하지 않는다. 기존 score/support 첫 후보와 old baseline/refinement를 정확히 replay했고, Euclidean NMS 중심의 SE(2) 분리와 refined 후보의 중복 여부를 검사했다.

## 연결성 비교와 비용

0.2m candidate 격자의8-neighbor edge를1cm floorplan raster에서61개 점으로 검사해 blocked pixel을 지나는 edge를 제거했다. 그 graph의 bounded shortest-path neighborhood를 사용한다. 모든 candidate의 world→pixel 변환과 free-space 여부를 검증했다. Habitat navmesh나 정확한 연속 geodesic이 아니며, candidate-grid 누락과 octile 거리 근사가 Euclidean 차이에 섞일 수 있다.
31개 Train/Val 장면의 평균 connected/Euclidean0.4m 이웃 수 비율은 0.998391이다. 이 반경에서는 대부분의 이웃이 그대로 남았다.
학습된 Euclidean-support ranker와 임계값을 고정해 extractor만 바꾼 결과:

| Seed/집합 | score NMS | Euclidean support | connected support |
|---|---:|---:|---:|
| 42/val | 72.625% | 74.010% | 74.010% |
| 42/extra | 68.883% | 70.442% | 70.459% |
| 43/val | 71.575% | 72.983% | 72.983% |
| 43/extra | 69.056% | 70.909% | 70.909% |

연결성을 추가한 것이 개선의 주원인이라는 증거는 없다. 같은 ranker에서 Euclidean과 connected 결과는 거의 같다. Score NMS로 교체하면 성능이 낮아져 support 기반 mode 추출의 역할은 남는다. 이는 이 자료·반경에 한정된 결론이다.

- val: 세 추출기와26개 특징 생성의 합산 CPU median 16.54ms/query, p95 23.91ms. Encoder/score matching 및 graph 최초 생성 제외.
- extra: 세 추출기와26개 특징 생성의 합산 CPU median 14.91ms/query, p95 26.88ms. Encoder/score matching 및 graph 최초 생성 제외.
- 준비된 특징에 대한 단일 query utility ranking median 0.549ms, p95 0.648ms. 단일 extractor의 배포 latency로 합산 추출 시간을 해석하면 안 된다.
- Graph 최초 생성/입출력 시간은 scene별 geometry receipt에 기록했다(합계 7.0s). 캐시를 재사용했으며 전체 온라인 pipeline latency나 GPU peak 측정값은 아니다.

## Ranking 실험과 실패

26개 GT-free 특징: V/A/fused score와 peak gap, 국소 modality 분포·질량·분산·상관, modality peak와의 상대 위치·방향, 이웃 수/연결 비율, 예측 depth uncertainty. Scene ID·절대 좌표·GT pose를 feature로 넣지 않았다. 최종 model은 candidate 특징26개와 첫 후보 대비 차이26개를 사용한다.
초기150-iteration/15-leaf gradient-boosted success classifier는 score-NMS Top1을71.384→71.981%로 높였으나 fixed support/refinement를 넘지 못했다. Support 후보에서 Train fine-preservation 제약을 없애고 joint만으로 임계값을 선택해도 Val8 seed42는72.673%로 낮았다.
같은 support 후보에서 success(candidate)−success(first) 회귀는73.842%를 얻었지만 R@0.1m가18.807%로 떨어졌다. 정답 후보 하나를 찾는 것만 보상하면 이미 성공하는 baseline을 다른 coarse-success pose로 바꾸면서 fine precision을 잃을 수 있다.
후속 utility difference 회귀는 joint+.25×fine 또는 joint+.5×fine을 Train OOF로 비교했다. Train macro joint를 최대화하되 Train macro fine과 visual preservation을 악화시키지 않는 threshold를 선택했다. 두 seed 모두 fine weight.5를 선택했고 threshold는42:.025,43:.1이었다. Val 결과를 보고 threshold를 검색하지 않았다. Model은 expected utility gain을 예측하며 calibrated probability가 아니다.
Candidate0은 검증된 support/refinement baseline이다. 모든 candidate의 utility gain이 임계값 이하이면 baseline을 유지한다. 추론 API는 truth를 입력받지 않는다.

## 무엇이 discrimination에 기여했는가

동일 후보·utility·capacity·Train-selected threshold를 고정하고 feature group만 제거해 재학습했다.

| 특징 구성 | seed42 joint | seed43 joint |
|---|---:|---:|
| full | 74.010% | 72.983% |
| absolute_only | 72.673% | 72.291% |
| scores_only | 72.912% | 72.339% |
| without_direct_audio | 73.532% | 72.673% |
| without_spatial_agreement | 73.437% | 72.745% |

가장 큰 감소는 baseline-relative context 제거에서 나타났다. 점수/margin만으로도 충분하지 않았으며 국소 분포와 상대적 spatial/modal agreement 특징이 추가 정보를 제공했다. Feature 수 변화와 model approximation의 영향까지 배제한 유일한 물리적 원인 증명은 아니지만, 같은 추출기를 고정한 feature-refit 대조로 기여를 확인했다. without_direct_audio에서도 fused score/support와 acoustic-conditioned 후보는 남으므로 음향 모달리티 제거 실험으로 부르지 않는다.
성공 사례의 특징 median은 단순히 audio score나 local correlation이 높아지는 규칙이 아니었다. 두 seed에서 repair의 median fused score 변화는 음수였다. 따라서 더 높은 원점수만 다시 고른 것이 아니다. Repair/harm 집단의 일부 특징 분포는 겹쳐 단일 cue로 정답을 판별할 수 없었다. `case_analysis.json`에 모든 특징 median, 선택 rank histogram, deterministic repair/harm 예시를 보존했다.
Seed42 joint-only gain 모델은107건을 복구하고67건을 해치며1071번 바꿨다. Precision-aware 모델은85건 복구/38건 손실로273번만 바꿨다. 무조건 많은 후보로 전환하기보다 baseline 대비 필요한 변경을 예측하는 목표가 coarse/fine tradeoff를 개선한 것으로 해석된다.

## 부작용과 재현 범위

| 집합 | R@1m 변화95% CI | joint 변화95% CI | R@0.1m 변화95% CI |
|---|---|---|---|
| val | +0.823pp [+0.548pp, +1.115pp] | +0.895pp [+0.682pp, +1.134pp] | +0.251pp [+0.136pp, +0.363pp] |
| extra | +0.442pp [+0.195pp, +0.665pp] | +0.597pp [+0.325pp, +0.867pp] | +0.069pp [-0.081pp, +0.249pp] |

Val11 fine 변화의 CI는0을 포함한다. 두 seed 평균이 양수라는 사실을 formal noninferiority나 일반화된 정밀도 개선으로 표현하지 않는다. Bootstrap은 query별 seed 평균 후 scene을10,000회 paired resampling했으며 seed 반복으로 관측 수를 두 배로 세지 않았다.

| 집합/seed | baseline joint | ranked joint | visual harm baseline→ranked | repair / harm |
|---|---:|---:|---:|---:|
| val/42 | 72.888% | 74.010% | 146→110 | 85/38 |
| val/43 | 72.315% | 72.983% | 149→131 | 36/8 |
| extra/42 | 69.766% | 70.442% | 184→148 | 98/59 |
| extra/43 | 70.390% | 70.909% | 196→178 | 45/15 |

모델 학습은 Train-only지만, B2 OOF와 ranking CV는 upstream까지 완전히 nested되지는 않았다. 기존 시각 encoder도 Val19를 모니터링했다. 알려진 scene 내부 탐색, simulated acoustic, 후보 격자·yaw bin·floorplan 품질에 대한 의존성도 남는다. 이미 본 데이터를 다시 미사용 validation이라고 부르지 않는다.

## 일반 원리와 다음 반증 실험

1. 높은 oracle coverage가 후보 선택의 학습 가능성을 보장하지는 않는다. Coverage와 실제 ranking을 각각 고정하고 측정한다.
2. 강한 baseline이 있을 때는 후보의 절대적 성공보다 baseline을 바꾸는 이득을 비교하는 특징과 학습 목표가 유용할 수 있다.
3. Coarse success만 보상하면 fine pose를 불필요하게 바꿀 수 있다. 목적에 필요한 정밀도를 utility에 포함하고 변경 수와 repair/harm도 검사한다.
4. 다음 반증 실험은 신규 데이터의 frozen ranking 확인, scene-nested upstream 검증, 벽이 조밀한 데이터의 geodesic/Euclidean 비교, 같은 capacity에서 상대 context와 추가 무관 특징의 비교, structured ray residual/hypothesis attention 대조다. 이는 향후 연구이며 이번에 검증한 개선 이상의 주장을 하지 않는다.

## 재현 자료

`PROTOCOL.md`, 각 model/result.json, `summary.json`, `cross_extraction_seed*.json`, `case_analysis.json`, `inference_verification.json`을 참조한다. Feature extraction은 `modes.py`, GT-free ranking API는 `inference.py`에 있다. 이 API는 이미 생성된 특징과 후보 indices를 받는다. Dataset extraction helper는 기존 workspace와 Docker에 의존하므로 standalone deployment package라고 주장하지 않는다.

[Results](artifacts.json) (외부 산출물: `${WORKSPACE}/experiments/hypothesis_discrimination_20261107_1538/results.png`)

> 역사적 연구 기록. 원본: `astra_fusion_20261107_0200/results_summary.md`. 원본 SHA256은 [산출물 목록](artifacts.json)에 있습니다.
> 당시의 완료·재개 지시는 현재 실행 요청이 아닙니다. 최신 판정은 [현재 결론](../../../current_findings.md)을 먼저 확인하세요.

# 결과 요약

- **새 개입의 개선은 확인하지 못했다.** 기준 재현 후4개의 저비용 pilot을 수행했고 사전 조건을 통과한 후보가 없어 확인 실행0/1에서 중단했다. 테스트 데이터 접근·새 학습·가중치 수정 없음.
- 동일한 Val3/1704 query, 동일 FP 후보/GT/시각·음향 특징, 동일 측정 함수. 이번 baseline은 동결 특징에서 점수와 융합을 재계산한 결과이며 RGB/RIR부터 전체 encoder를 다시 실행한 결과는 아니다.
- 아래 Joint는 translation≤1m AND circular yaw error≤30°. Macro는 scene 동등 가중, 나머지는 query 동등 가중. 연구 목표에 따라 성능·퇴행·불확실성을 함께 보고한다.

| 방법 | R@1m (%) | Joint (%) | Macro R@1m (%) | Macro Joint (%) | 시각 대비 복구/악화 (joint) |
|---|---:|---:|---:|---:|---:|
| 시각 단독 | 55.16 | 54.46 | 53.91 | 53.21 | 0/0 |
| 음향 단독 | 35.80 | 26.47 | 34.78 | 25.73 | 171/648 |
| 고정 w=.1 | 75.65 | 74.24 | 74.50 | 73.13 | 386/49 |
| 고정 w=.5 | 65.55 | 63.03 | 64.31 | 61.81 | 347/201 |
| 기존 학습 게이트 | 75.88 | 74.30 | 74.82 | 73.27 | 396/58 |
| P1 고정 w=.05 | 74.30 | 72.89 | 73.21 | 71.87 | 344/30 |
| P2 엔트로피 게이트 | 75.23 | 73.83 | 74.07 | 72.70 | 379/49 |
| P3 음향 yaw 주변화 | 74.65 | 72.83 | 73.37 | 71.65 | 370/57 |
| P4 음향 셔플 | 33.86 | 33.22 | 33.06 | 32.37 | 65/427 |

## 판단

- 기존 adaptive58는 fixed01보다 joint **+0.0587%p(순1 query)**. 51 repair/50 harm. 장면별 차이는 -0.7353/+1.4583/-0.2941%p이며3scene cluster-bootstrap95% 기술 구간 [-0.7353,1.4583]%p. 신뢰할 일반화 개선으로 판단하지 않는다.
- P1은 joint -1.3498%p. 시각 성공 harm은49→30으로 줄지만 rescue386→344로 줄어 퇴행한다. 전역 음향 억제는 채택하지 않는다.
- P2는 joint -0.4108%p. 1615/1704 query에서 음향을 유지하고89개에서 차단; aggregate 뿐 아니라3개 장면 모두 퇴행. 낮은 entropy를 높은 localization 신뢰도로 해석할 근거가 없다.
- P3는 joint -1.4085%p, harm49→57. 이 구체적 yaw 제거+재정규화는 기각한다. 음향 yaw 자체의 필요성을 확정하는 순수 ablation은 아니다.
- P4는 joint -41.0211%p, 시각 단독보다도 낮다. 동일 scene에서 query 대응을 깨는 개입은 실제 음향 대응의 기여를 지지한다. scene-prior만으로 성능을 설명하기 어렵다. 한 derangement이므로 효과 크기의 permutation 분산은 미측정.
- 각 pilot의 joint 장면별 차이와 기술적 구간은 [analysis.json](evidence/analysis.json). 3scene bootstrap은 독립 scene 수가 작으며 모집단 일반화를 보장하지 않는다. 유의성 검정/다중비교 보정/독립 test가 아니다.

## 부가 지표와 검증

- [baseline/metrics.json](artifacts.json) (외부 산출물: `baseline/metrics.json`), [pilots/metrics.json](artifacts.json) (외부 산출물: `pilots/metrics.json`): XY/yaw median, XY p95 tail, GT nearest-pose rank/NLL, 성공영역 NLL, entropy, task confidence/ECE, subgroup, repair/harm.
- Posterior는 `softmax(score/T=1)` 진단 분포다. nearest candidate/yaw GT NLL은 연속 pose density가 아니며 ECE는 MAP 주변1m/30° mass를 confidence로 정의한다. 고정0.1의 평균 confidence≈0.00775, 실제 joint≈0.74237로 T=1은 심하게 underconfident. NLL/entropy 개선을 localization 개선과 혼동하지 않는다.
- 전체 logits는 수GB 중복 저장 대신 SHA256-검증된 query/FP/windows와 [run.py](artifacts.json) (외부 산출물: `run.py`)의 결정적 재계산식으로 보존한다. 샘플별 GT/예측/오차/rank/NLL/entropy/gate는 각 `samples.json`에 있다. 원본 RGB/RIR SNR·ToA·가구 반사 counterfactual은 이 실행에서 추출하지 않았다.
- 독립 검증 [verification.json](artifacts.json) (외부 산출물: `verification.json`): sample ID·GT 대응1704개, unique IDs, source receipt→config chain, gate SHA256, 저장소 원래 metrics와 수치 일치. [analysis.json](evidence/analysis.json)의 pilot_baseline_exact/repository_metrics_equal도 통과.
- [diagnostics.png](artifacts.json) (외부 산출물: `diagnostics.png`): 전체 성능, repair/harm, scene delta, entropy 비교. [case_studies.json](artifacts.json) (외부 산출물: `case_studies.json`): 결정적 사례 선택과 전체 측정.
- [recent_prediction_replay.json](artifacts.json) (외부 산출물: `recent_prediction_replay.json`): 최근 CNN/STFT self/cross 모델 seed42/43의 선택 epoch 저장 예측을 동일Val3에서 재측정했다. 이 모델은 adapted query heads/vision과w=.15를 사용하므로 위 표와 matched training comparison이 아니다. 이 재측정은 checkpoint inference 재실행도 아니다.

## 재현과 비용

- 실행 원문 [commands.sh](artifacts.json) (외부 산출물: `commands.sh`), [run.py](artifacts.json) (외부 산출물: `run.py`), [verify.py](artifacts.json) (외부 산출물: `verify.py`), [analyze.py](artifacts.json) (외부 산출물: `analyze.py`). 새 출력 폴더만 허용하며 기존 결과가 있으면 실패한다. 재실행은 새 폴더 복사 후 writable mount와 분석 경로를 바꿔 원본을 보존한다.
- CPU 기준 baseline 43.06초,4 pilot 공동 계산52.06초, 추가 검증·분석 수초. 동일 score 계산을 공유했으며 pilot별 시간을 합산해 비용을 부풀리지 않는다. GPU0GB, fitting0회. `avfploc:cu124-clean` image ID는 [environment.json](artifacts.json) (외부 산출물: `environment.json`)에 기록.
- [experiment_log.csv](artifacts.json) (외부 산출물: `experiment_log.csv`), [hypothesis_register.md](archive/hypothesis_register.md), [selection.json](artifacts.json) (외부 산출물: `selection.json`): 전 시도·선택 규칙·중단 이유. 확인용 추가Val5와 Test를 이번에 열지 않았다.
- 필수 산출물: [감사](archive/fusion_system_audit.md), [진단](archive/failure_diagnosis.md), [가설](archive/hypothesis_register.md), [실험 로그](artifacts.json) (외부 산출물: `experiment_log.csv`), 이 요약, [연구자 판단](archive/researcher_decisions.md).

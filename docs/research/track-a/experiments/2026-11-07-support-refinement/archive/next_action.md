> 역사적 연구 기록. 원본: `astra_fusion_v2_20261107_0225/next_action.md`. 원본 SHA256은 [산출물 목록](../artifacts.json)에 있습니다.
> 당시의 완료·재개 지시는 현재 실행 요청이 아닙니다. 최신 판정은 [현재 결론](../../../../current_findings.md)을 먼저 확인하세요.

# 연구 상태 — 검증 개선 재현 완료

- **가장 강한 비교 기준:** B2 epoch8, V.85/A.15, Val8·4190관측·seed42/43. R@1m 72.566%, 동시 성공률 71.014%, R@0.1m 22.136%.
- **최종 후보:** `SupportFusion(refine=True)`. R@1m 74.224%, 동시 성공률 72.601%, R@0.1m 22.279%. 8개 장면의 seed 평균 동시 성공률이 모두 개선됐습니다.
- **중요한 제한:** 시각 성공 훼손은 평균149.5→147.5개로 줄었지만 일부 장면은 증가했습니다. 엄격한 비열등성 검정이나 독립 Test 일반화를 입증하지 않았습니다. 학습 gate의 성공이 아니라 posterior 결정 방식의 개선입니다.
- **확정 파일:** [inference.py](../artifacts.json) (외부 산출물: `inference.py`), [설정](../artifacts.json) (외부 산출물: `inference_config.json`), [동결 기록](../artifacts.json) (외부 산출물: `refinement_frozen.json`), [최종 통계](../evidence/final_summary.json), [검증](../artifacts.json) (외부 산출물: `inference_verification.json`), [실험 로그](../artifacts.json) (외부 산출물: `experiment_log.csv`).
- **기각/열린 가설:** 새 선형·트리·paired-gain 선택기 3종은 Val 개선 실패. 국소 지지도와 영역 내부 정밀화는 개선 재현. 가구 원인, 확률 calibration, 학습형 신뢰도 일반화, 벽 연결성 이웃은 미검증입니다.
- **종료 이유:** v2 §7.2의 ‘요청한 개선을 재현하고 문서화’ 조건을 충족했습니다. 정해진 batch 수나 보고서 작성만으로 종료하지 않았습니다. 로컬 자원 자유 사용 승인은 받았지만, 현재 목표를 달성한 뒤 불필요한 탐색을 계속하지 않습니다.

## 정확한 다음 실행

- 동일 결과를 새 폴더에서 재현하려면 [reproduce.md](reproduce.md)의 명령을 실행합니다. 원본 결과를 덮어쓰지 않습니다.
- 최종 후보의 독립 추론 검사만 재실행하는 명령:

```bash
bash ${WORKSPACE}/experiments/astra_fusion_v2_20261107_0225/run_script.sh verify_inference.py
```

이 명령은 새 모델을 학습하거나 Test를 읽지 않으며 `inference_verification.json`만 다시 기록합니다. 원본 검증 기록까지 보존하려면 먼저 새 출력 폴더로 복사하십시오.

- **다음 새 연구 실험:** 미관측 장면에서 고정 후보를 한 번 평가하고, 사전 정의한 시각 비열등성 기준을 확인하는 것입니다. 이를 위한 장면 목록과 Test 사용 승인은 아직 없습니다. 존재하지 않는 manifest나 실행 명령을 만들어 기재하지 않았습니다.
- **실행 환경:** `avfploc:cu124-clean`, CPU4/RAM8GB, GPU allocator 45% cap. checkpoint와 입력 경로·해시는 각 cache/support receipt에 있습니다. 기존 branch와 checkpoint 변경 없음. Test는 봉인 상태입니다.

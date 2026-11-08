> 역사적 연구 기록. 원본: `astra_fusion_v2_20261107_0225/fusion_system_audit.md`. 원본 SHA256은 [산출물 목록](../artifacts.json)에 있습니다.
> 당시의 완료·재개 지시는 현재 실행 요청이 아닙니다. 최신 판정은 [현재 결론](../../../current_findings.md)을 먼저 확인하세요.

# 재개 감사

- 이전 감사/로그를 먼저 검토했다. [`../astra_fusion_20261107_0200/fusion_system_audit.md`](../../2026-11-07-fusion-audit/archive/fusion_system_audit.md)의 데이터/좌표/전처리 계약과 체크포인트 계보를 재사용한다. 새로 처음부터 진단하지 않는다.
- 동일 Git commit `6ebcf64c22f334542fd623cb3a21005b02a66331`, branch `refactor/hssd-two-branch-fusion`, 기존 Gibson 파일2개 수정 보존. 전용 새 실험 폴더와 read-only source/data mounts로 격리한다. 이전 worktree 도구의 host checkout 인식 실패는 이전 environment.json에 기록됐다.
- Baseline 개선: 원래 E+C+N score+scalar gate만을 최강이라고 보지 않는다. `mbt_remaining_suite_20261106/runs/cap020/B2_seed{42,43}_{foldN/full}_epoch8.pt`를 원래 Exchange/Heads 코드로 재실행. Frozen raw CNN/UnLoc feature→B2 bottleneck exchange→adapted query heads→동일 FP fullbank→z-score→fixed V.85+A.15.
- B2는 Transformer bottleneck을모달리티별로갱신하고latent 평균교환하는구조. 직접 Q=audio/K,V=visual은 `direct_attention_stft_20261107/network.py`에서만 확인된 별도 구현. 단순 projection 평균을 cross-attention으로 부르지 않는다.
- B2 Train12/6387에서는 해당scene을학습에서제외한4fold checkpoint만사용. Val3/1704에서는fullTrain 모델. Train gate 자체의cross-validation은encoder/adaptor까지nested하지않았으며, Val/test leakage와구분해 한계로명시한다.
- checkpoint config hash/held scene 확인, data receipt+SHA256 검증, original sweep의 fixed.15/vision/acoustic index와전query동일성 확인 후 새 모델을 학습한다. 결과 `cache_seed*/<scene>.json`.
- 로컬4090 24GB 중기존서버가7GB 사용. 새 inference memory fraction .45 제한. CPU4/RAM8GB; 무료local사용승인. dependencies는전용deps 디렉터리(scikit-learn1.5.2/joblib1.4.2/threadpoolctl3.5.0), 기존환경을수정하지않음.
- 새후보선택기의inference입력: score/gap/localmean/entropy/relative peak separation/yaw disagreement/predicted depth·scale. GT/sceneID/absolute world position은 없음. 성공target은Train fitting에만 사용. full-bank MAP후보8개가중치에서생성, true pose 주입 없음.
- 비교기준: 현재 strongest fair B2.15; 원래시각/음향 및고정.1, 기존adaptive58와최신CNNcross의동일Val3결과도보고하되학습pipeline 차이를 구분한다. Test읽기금지.

## 최종 감사 결과

- 기준 재현은 성공했다. seed42의 Train OOF/Val과 seed43 Val에서 B2 fixed.15의 full-bank index가 기존 sweep과 일치했다. 새 방법은 기존 후보를 추가하거나 GT에 맞춰 걸러내지 않는다.
- 최종 추론은 `inference.py`의 `SupportFusion`이다. 새 학습 파라미터는 0개이고, 고정 모달리티 점수에서 국소 지지도와 지역 MAP 정밀화를 계산한다. 입력과 설정은 `inference_config.json`을 참고한다.
- 기존 v1 산출물 32개 해시와 기존 Git diff가 모두 그대로임을 확인했다. Parent Train/Val/Test membership도 다시 검증했다. 증거: `final_integrity.json`.
- 최종 방법의 시각 성공 보존은 전체 평균의 관찰이며, 일부 장면 퇴행과 불확실성이 남는다. 이 점을 학습된 reliability나 엄격한 비열등성으로 해석하지 않는다.

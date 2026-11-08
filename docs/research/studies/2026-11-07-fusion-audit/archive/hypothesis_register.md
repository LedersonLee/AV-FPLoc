> 역사적 연구 기록. 원본: `astra_fusion_20261107_0200/hypothesis_register.md`. 원본 SHA256은 [산출물 목록](../artifacts.json)에 있습니다.
> 당시의 완료·재개 지시는 현재 실행 요청이 아닙니다. 최신 판정은 [현재 결론](../../../current_findings.md)을 먼저 확인하세요.

# 가설 등록 — pilot 결과 열기 전 고정

근거: [baseline/metrics.json](../artifacts.json) (외부 산출물: `baseline/metrics.json`), [baseline/samples.json](../artifacts.json) (외부 산출물: `baseline/samples.json`). 초기Val3/1704, 지표는 sample-micro 및 scene-macro R@1m, joint(1m&30°). 한 개입씩 변경하며 encoder, 후보, GT, 입력, 지표 구현을 고정한다. 모든 pilot은 fitting 없이 seed42, CPU4/RAM8GB/VRAM0GB, 약1~3분 예상. 최대4 pilot과 후보1개 확인; 총30분 상한.

1. **H1: 방향 모호성이 음향의 유용한 XY 정보를 방해한다.**
   - 근거: 음향 R1=35.80%와 joint=26.47% 간격9.33%p; 시각의 간격은0.70%p. 두 단독 실패605개 중 융합267개 복구: 단독 성공 여부만으로 음향을 버리면 안 된다.
   - 개입 P3 `yaw_marginal`: 각 XY에서 `logsumexp(A_yaw)` 후 XY z-score, 각 yaw에 동일하게 broadcast, `.9V+.1A_xy`. 이로써 최종 yaw는 선택 XY의 visual yaw가 된다.
   - 예측: joint 개선/시각 성공 harm 감소. 반증: fixed01보다 macro joint가 개선되지 않거나 R1 이득 없이 harm 증가.
   - 대조: fixed01, vision. 제한: yaw 정보 제거와 XY 정규화가 묶인 개입이므로 양성이라도 순수 yaw의 인과 효과라고 단정 불가.
2. **H2: 현재 가중치는 일부 query에서 음향을 과신한다.**
   - 근거: equal harm201 vs fixed01 harm49(시각 joint 성공 기준), learned gate harm58. 기존 adaptive의 weight median=.222, p90=.548. adaptive가 fixed01보다51 repair/50 harm으로 순이득1개.
   - 개입 P1 `calibration005`: 고정 w=.05, 나머지 동일. 별도 온도/weight sweep 없음.
   - 예측: 시각 성공 harm 감소, macro joint와 R1 유지/개선. 반증: rescue 감소가 harm 감소보다 커서 두 주지표 하락.
   - 위험: 전역 음향 억제이며, 성공해도 selective fusion 학습의 증거가 아니다.
3. **H3: 전체 점수 엔트로피를 모달리티 신뢰도로 쓰면 선택에 실패한다.**
   - 근거: acoustic joint는26.47%로 낮지만94.78% query에서 visual보다 엔트로피가 낮다. T=1의 task ECE는 fixed01=.735로 calibration이 안 됐다.
   - 개입 P2 `entropy_gate`: normalized entropy(A)<entropy(V)이면 w=.1, 아니면0. 임계값 fitting 없음, GT/scene ID 사용 없음.
   - 예측: 대부분 음향을 유지하고 고정 융합보다 개선되지 않거나 퇴행. 반증: 주지표 개선과 reliable-vision non-regression이 추가Val에서도 재현됨.
   - 대조: fixed01; 위험: entropy는 후보수 정규화만 했고 서로 다른 분포 모양/score scale을 보정하지 않는다.
4. **H4: 고정 융합의 복구는 query에 대응하는 음향 정보에 의존한다.**
   - 근거: fixed01이 시각보다386 repair/49 harm. 두 단독 실패 구간에서267 repair.
   - 개입 P4 `shuffled`: 같은 scene query 음향 embedding을 seed42의 fixed-point 없는 순열로 치환. 후보 위치/시각/query GT는 고정.
   - 예측: 원래 fixed01보다 joint/R1 및 rescue 크게 감소. 반증: 대응을 깨도 이득 유지.
   - 위험: 같은 scene 내 pose 분포가 남으므로 완전 OOD/노이즈 대조가 아니다. 일반화/가구 원인의 증거도 아니다.

## 선택과 중단 규칙

- P1~P3 중 fixed01 대비 **macro R1과 macro joint 모두 개선**, visual-success joint harm이 증가하지 않는 방법만 promising. 통과하면 macro R1, macro joint, method 이름 순으로 하나를 고른다. P4는 진단 전용이며 선택 불가.
- 통과 후보를 코드/규칙/가중치 그대로 freeze하고 추가Val5에서 한 번 확인. 임계값/온도 재탐색 없음. 후보 없으면 confirmation 미사용, 추가 실행 중단.
- 추가Val5는 과거 개발 자료라 독립 test 일반화로 주장하지 않는다. 별도 test 실행은 하지 않는다.
- 원인 A(정보 손실), E(가구 mismatch)는 여기서 직접 개입하지 않음. B(점수/보정), C(gate), F(yaw geometry), G(비교 confound)를 우선 진단한다. 원시특징/attention 재학습은 결과 후 연구자 결정 사항.

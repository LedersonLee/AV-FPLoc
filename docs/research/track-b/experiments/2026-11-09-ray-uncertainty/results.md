# 결과 — 아직 미측정

입력 감사와 U0 지표 재현이 통과했다. 아래 모든 조건은 동일19개 재사용 Val 장면에서 평가할 예정이며, U1–U4 재추론 결과는 아직 없다.

| Arm | Normal | Occlusion25/50/75 | Brightness0.2 | Audio shuffle | RIR noise20dB | Ray calibration | Latency |
|---|---|---|---|---|---|---|---|
| U0 | 재현 통과; 아래 수치 | NA | NA | NA | NA | NA | NA |
| U1 | NA | NA | NA | NA | NA | NA | NA |
| U2 | NA | NA | NA | NA | NA | NA | NA |
| U3 | NA | NA | NA | NA | NA | NA | NA |
| U4 | NA | NA | NA | NA | NA | NA | NA |

각 cell은 V/A/fusion의 Joint/Fine/raw5·10·20/diverse20/NLL/ECE 및 추가 지표로 확장한다. U3−U1/U2/U4/U0의19-scene paired CI도 NA다. 효과/반증/다음 연구 선택은 모든 선언 조건을 완료한 후 판단한다. 학습 실패 기록은 현재 없으며 학습 자체가 미실행이다.

## U0 재현 완료

19장면×3seed×4방법의14,592행 전체가 원래 평가와 주요14지표에서 허용 오차 내 일치했다(불일치0). F1 자체는3,648행이며 Joint34.4024%, Fine3.0154%, raw20 70.4770%, diverse20 89.6656%, NLL9.463879이다. [검증](u0_replay_validation.json), [요약](u0_replay_summary.json). 이는 기존 기준선 재현이며 uncertainty의 효과는 아직 NA다. 역사적 full score 배열/선택 index의 bitwise 일치를 뜻하지 않는다.

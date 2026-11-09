# Study 30 — 방향별 관측 불확실성 진단

**완료: 주 가설 지지 실패, Exploratory.** 정상 및 가림25/50/75%, brightness×0.2, audio shuffle, Gaussian20dB의 모든 U0–U4×V/A/fixed 평가를 완료했다. 정상 U3 Joint29.688%는 U0 34.402%보다 낮고, U2 대비 추가 pose 이득을 확인하지 못했다. 음향 열화의 U0 대비 개선은 고정 L1 대조에서도 나타나 방향별 uncertainty 고유 효과로 볼 수 없다. severe visual occlusion에서는 음향 단독 대비 negative transfer가 남는다.

- [전체 결과·표·CI·실패와 한계](results.md)
- [고정 protocol](protocol.json), [preflight](preflight_report.md), [최종 검증](validation.json)
- [재현·source/checkpoint 출처](reproduction.md), [실행 registry](experiment_registry.csv)
- [사전 결정 및 다음 Study 제안](decisions.md)

Train112/reused Val19×64, seeds42/43/44. Test 미사용. 기존 upstream/Val 노출과 반복 평가 때문에 독립 검증이 아니다. head1000updates 외 frozen F1을 보존했다. 이전 진행 문서들은 당시의 상태 기록이며, 최종 상태는 이 페이지와 results.md/validation.json을 따른다. 기존 Study21/24/28 실패를 덮어쓰지 않는다.

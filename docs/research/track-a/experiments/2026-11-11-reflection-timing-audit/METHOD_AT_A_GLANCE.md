# 방법 한눈에 보기 — 벽이 바뀌면 소리의 어느 부분이 바뀌는가?

소리는 벽까지 갔다가 microphone으로 돌아온다. 평평한 벽 뒤로 source를 거울처럼 옮기면, 그 가상 source에서 microphone까지의 길이가 이상적인1차 반사 길이와 같다. **길이÷음속**이 기대 도달 시각이다. 이 예측은 소리 파형을 보지 않고 계산한다.

1. **기하를 먼저 고정한다.** 실제 렌더러가 읽은 벽과 source/6mic 좌표를 export·ray query로 확인한다. 다른 수치 최소화로 반사 길이를 재검산한다.
2. **예측과 관측을 분리한다.** 예상 시각 주변에서 peak를 골라 정답으로 부르지 않고 공통1–80ms 전체에서 onset, peak, centroid를 구한다. Pressure waveform에서 만든 envelope와 renderer 내부 energy histogram을 혼동하지 않는다.
3. **벽에 개입한다.** 제거하면 응답이 없어지는지,1m 멀리 옮기면 약5.83ms 늦어지는지, 같은 위치를 반복해도 유지되는지 검사한다. Indirect off는 반사 계열의 증거이지 특정1차 경로 GT는 아니다.
4. **렌더러와 관측법을 나누어 검사한다.** Habitat를 우회해도 늦은 응답은 남는다. Peak 대신 onset을 쓰거나 temporal coherence를 켜도 해결되지 않는다. 실제 acoustic mesh가 알려준 기본 scattering0.5를0으로 바꾸면 큰 시간 오차가 줄어든다.
5. **전체 기준으로 판정한다.** Relative wall shift가 가까워진 것만으로 성공이라 하지 않는다. 모든 채널 절대오차0.10ms, 이동0.20ms 등 사전 gate와 path identity 한계를 함께 본다. Default 관측의 gate 실패로 HSSD를 재평가하지 않는다.

가장 중요한 발견은 **‘materials off’가 완벽한 정반사 벽이라는 뜻이 아니었다**는 점이다. Scattering0.5→0 개입은2m peak를약2.48ms,6m peak를약6.05ms 앞당겼다. 그러나 point receiver에서도 잔여 오차가 있고 내부 경로는 보이지 않는다. 따라서 결론은 **Outcome C: 설정에 의존하는 관측**, 정확한 first-order alignment 검증은 보류다.

[인터랙티브 화면](interactive.html)에서 다음 순서로 비교할 수 있다.

- `Previous · single_d2.0_base → removed → moved → repeat`: 벽 의존성은 있지만 이상적 시간 이동은 불일치하는 실패 사례.
- `New · d2.0_explicit_default_c0 → specular_c0`: 같은 geometry에서 scattering만 바꾼 인과 대조.
- `specular_moved` 및6m, microphone1–6: 상대 변화의 부분적 성공과 잔여 절대오차를 함께 확인.
- `indirect_off`, `temporal_on`, `simultaneous`: 각각 무엇을 제거하거나 바꿨는지 확인. 새 파형과 이전 파형은 출처가 구분된다.

화면은 저장된 실제 RIR을 표시한다. Min/max bin은 그림 크기만 줄이며 추정은 원시 sample에서 계산했다. 거울경로는 analytic prediction이고 path GT가 아니다. 독립 analytic impulse 성공은 geometry 검사로만 보고하며 이 화면의 실제 결과에 섞지 않는다.

이 발견으로 예측할 수 있는 것은 **같은 geometry라도 renderer의 scattering/radius 설정을 바꾸면 pressure peak timing이 달라질 수 있다**는 것이다. 실제 환경에서 후보별 wall consistency가 유효한지는 이번 연구가 답하지 못한다. 다음 연구는 자동 시작하지 않는다.

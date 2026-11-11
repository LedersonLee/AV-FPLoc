# Physical Verification — first-order arrival correspondence

## 물리 원리

유한 수직 벽선(a,b)의 단위 normal n에 대해 source s의 image는 s′=s−2n[n·(s−a)]다. 채널 receiver rᵢ에서 image로 이어지는 선이 벽 내부 qᵢ를 통과하고 양 leg가 가려지지 않아야 한다. 예상 길이 Lᵢ=‖s′−rᵢ‖,시간 tᵢ=Lᵢ/343. source와receiver가 벽의 같은 쪽이어야 하며 point가 segment 밖이면 제외한다. 마이크별 별도 qᵢ와시간을 사용한다. 임의 radial ray 거리2d를 모든 방향 반사로 지정한 proxy와 구분한다.

시각화: [interactive](interactive.html). 메인 분석: [report](RESEARCH_REPORT.md). 설정: [protocol](PROTOCOL.md).

## 실제 renderer 개입

Single-wall,8×8m room,2.4×18m corridor; front-wall거리0.3/2/6m; baseline/repeat/front-wall 제거/front-wall1m 이동의36rings. source(0.17,1.5,d),radius0.05m 6mono positions. 벽 y−8..12m,원격 바닥 y−20m를 모든 조건에 둬 navigation bounding box를 유효하게 했다. 해당 바닥 monostatic return은 약125ms 이후이므로1–80ms 해석에서 제외된다. 모든 mesh의 simulator-loaded bounds를 vertex bounds와1e−4m 이내로 검증했다. 두면 panel의 우려는 closed-box/one-sided 후속으로 점검했다.

경로 GT를 export하는 API는 사용 가능한 binding에 없다. 따라서 벽이 사라지거나 이동했을 때 response가 변하는지를 검증했다. Room/corridor 개입은 고차 경로도 바꾸므로 차분 전체를 '첫 반사 GT'라고 부르지 않는다. Single-wall은80ms 전에 다른 큰 reflecting surface가 없어 wall-dependent response의 가장 강한 대조다.

## 예상 창 내 모든 조건의 결과

아래 값은 **예상±0.5ms 창**의 absolute mean-squared amplitude 비율이다. 작은 baseline/removed noise floor끼리의 비율은100dB도 가능하므로 반사 검출 성공 지표가 아니다.

| 환경 | 앞벽 거리 | 제거 power ratio(dB) | repeat 차이(dB) | 이동 후 new/old(dB) |
|---|---:|---:|---:|---:|
| single | 0.3m | 0.001 | -0.000 | -43.630 |
| single | 2.0m | 4.459 | 0.763 | -3.369 |
| single | 6.0m | 102.824 | 0.014 | -0.000 |
| room | 0.3m | 0.001 | 0.000 | -43.616 |
| room | 2.0m | 3.196 | 0.035 | -3.298 |
| room | 6.0m | -2.482 | 0.075 | -0.747 |
| corridor | 0.3m | 80.526 | -3.025 | 78.434 |
| corridor | 2.0m | -0.063 | 0.010 | 11.406 |
| corridor | 6.0m | -2.605 | -0.380 | 0.184 |

단일 벽의 양의(base energy−removed energy) 차분 중 예상±0.5ms에 포함된 비율은0.3m에서3.29e−13,2m6.63e−16,6m4.29e−16이다. 거의0이며 작은 부동소수점 잔차를 반사로 부르지 않는다. 이 차분은 stochastic waveform 변동을 포함하므로 정확한 path-energy decomposition은 아니다. Repeat absolute sample-energy 차이는 removed effect의0.19/1.45/1.49배로,2/6m에서 세부 waveform shape 변동이 크다. 반복 peak timing은 훨씬 안정적이다.

각 채널 expected/observed times,window peak power,sharpness,coherence,timing residual과0.25/0.5/1ms 분석은 [controlled_analysis.json](controlled_analysis.json)에 보존했다. Mean peak를 임의 GT로 지정하지 않았다. 'Observed peak'는전체1–80ms에서 계산한 기술통계이며, 벽 귀속은 removal/movement와 함께만 해석한다.


| 단일 벽 거리 | 예상 mean(ms) | baseline observed mean(ms) | 이동 후 observed mean(ms) | observed shift(ms) |
|---|---:|---:|---:|---:|
| 0.3m | 1.7523 | 2.3438 | 9.0035 | 6.6597 |
| 2.0m | 11.6623 | 14.0104 | 21.1042 | 7.0937 |
| 6.0m | 34.9856 | 41.1493 | 47.7014 | 6.5521 |

Expected wall-move shift는약5.83ms다. 관측시각은각채널1–80ms최대점의평균이며reflection-path GT가아니다.

## 추가60rings: 대안 설명을 가르는 검증

| 조건 | 2m timing bias 평균±repeat SD(ms) | 6m timing bias 평균±repeat SD(ms) |
|---|---:|---:|
| baseline | 2.3597 ± 0.0107 | 6.1510 ± 0.0161 |
| fs24000 | 2.3493 ± 0.0033 | 6.1441 ± 0.0182 |
| fs44100 | 2.3450 ± 0.0064 | 6.1432 ± 0.0108 |
| fs96000 | 2.3250 ± 0.0103 | 6.1551 ± 0.0196 |
| no_diffraction | 2.3586 ± 0.0028 | 6.1741 ± 0.0170 |
| depth1 | 2.3759 ± 0.0177 | 6.1695 ± 0.0427 |
| rays200k | 2.3204 ± 0.0057 | 6.1290 ± 0.0085 |
| closed_box | 2.3458 ± 0.0128 | 6.2216 ± 0.0254 |
| one_sided | 2.3586 ± 0.0098 | 6.1765 ± 0.0256 |
| free_direct | -0.3289 ± 0.0000 | -0.3884 ± 0.0000 |

각 조건은3fresh repeats. SD는3repeat의 기술통계이며 독립 scene CI가 아니다. free_direct는 source-array center4/12m의 벽 없는 직접 경로로,같은 길이의 반사와 비교한 clock audit이다. 직접 경로는c343예상보다약0.33/0.39ms 빠르고 반사는약2.35/6.15ms 늦다. 따라서 단일 전역 sample-rate scaling 오류만으로 두 현상을 설명할 수 없다. 초기 근거리 direct peak는sample7이며 전체 시간축을 query peak에 맞춰 shift하지 않았다.

24/44.1/48/96kHz,time in seconds,회절off,depth1,200k rays,closed-box,one-sided 모두 지연을 제거하지 못했다. 이 결과는 samplerate·diffraction·higher-order-only·duplicate-face-only 설명에 반대되는 증거다. 반사 backend가 어떤 변환을 적용하는지는 **미확정**이다. 실측 음속이나 backend path GT를 확보하지 않은 채 c를 fitting하거나 시간 보정 공식을 학습하지 않았다.

## 실패 조건과 결론

좁은 expected window의 최고점은 noise가 될 수 있고,coherence가 높아도 공통 low-level filter tail일 수 있다. 특히 near-wall direct/early overlap,room/corridor의 여러 경로 중첩,weak/distant response에서는 잘못된 wall attribution 위험이 크다. 이번 production renderer와 설정에서 물리 예상창 대응은 음성이다. 벽 의존적 음향 response의 존재와 HSSD 조건부 에너지 연관성은 이 부정 결론과 양립한다.

독립적인 실제 실내 녹음·재질 다양성·정밀 wall-path annotation·3D ceiling/floor reasoning으로 일반화하지 않는다. 해당 항목은 이 실험의 확보된 입력에 없으며 path export/실측 센서가 필요하다.

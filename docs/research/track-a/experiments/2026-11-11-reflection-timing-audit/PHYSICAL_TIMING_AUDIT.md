# Physical timing audit

최종 판정: **Outcome C**. 설정 효과는 확인, 내부 acoustic path timing 및 wall ID는 미확인이다.

## Analytical path와 실제 로드 geometry

Source S=(0.17,1.5,d), R_i=S+(0.05 cos θ_i,0,−0.05 sin θ_i), θ_i=π+iπ/3. 벽z=w의 image source S′=(Sx,Sy,2w−Sz). R→S′ 선과 벽의 교점 Q가 finite wall 범위(x±12,y−8..12)에 있는지 확인하고 L=|S−Q|+|Q−R|, t=L/343을 계산한다. Source와 receiver는 벽 같은 쪽에 있고 반사점은 패널 안이며 distant floor(y−20)는 두 leg를 가리지 않는다. 이 geometry에서는 각 leg에 다른 장애물이 없다. Wall moved는 w=−1로 거리1m 증가다.

기존 vectorized solver, 새 scalar image-source, 독립 수치 Fermat wall-point 최소화를36경로(0.3/2/6m×base/moved×6ch)에서 비교했다. 시간 차이<1e−9ms. 독립 minimizer로 생성한 fractional two-tap impulse의 peak는1sample(0.02083ms) 이내다. 이는 analytic reference 검증이며 SoundSpaces 파형이 아니다. Reference는 로컬 파일에 명시적으로 분리했고 실제 RIR viewer에 섞지 않았다.

C API가 export한72개의 acoustic OBJ에서8geometry vertices 및 wall/base-floor 좌표를 검사했다. OBJ 뒤쪽 source/listener 표시용 vertices는 geometry 계산에서 제외했다. Ray firsthit 거리2/6m 및 moved3/7m는1e−5m 이내이고 normal은+Z(반환 norm480, 정규화 필요)였다. 이 API는 기하 교차이지 acoustic path/order output이 아니다. Source/listener float receipt와 analytical ring도1e−6m 이내다. `unitScale=1`이며 cm/m 뒤바뀜 근거는 없다.

## Source/receiver 및 시간 원점

Habitat Python `_get_audio_observation`은 sensor node absolute translation과 rotation으로 listener를 설정하고 simulation/getIR을 호출한다. 설치된 C++ `AudioSensor.cpp`의 source/listener setter는 좌표 float를 RLR에 전달하고, mesh는 joined acoustic mesh vertices/indices를 전달한다. `getIR`은 `RLRA_GetIRChannel` samples를 복사한다. 이 경로에 추가 resampling, delay buffer 또는 convolution은 없다. RLR 내부 합성은 별도다.

Raw sample0을 공통 원점으로 유지했다. 기존 ring 직접음 peak7/48k=0.1458ms는0.05m의 이론 전파시간과 일치한다. 그러나 default RLR listener radius0.1m(헤더 명시)는 ring 반경0.05m와 다르다. Point receiver로 바꾸면 default onset이 약+0.287/+0.279ms 늦어지고 specular peak가약+0.288/+0.295ms 늦어진다. 수 ms bias의 전체 원인은 아니다. 장거리 free-direct diagnostic도 완벽한 절대시간 calibration이 아니며 fitted correction을 만들지 않았다.

## Direct / first-order / late와 pressure IR

[SoundSpaces2 공식 구현 안내](https://github.com/facebookresearch/sound-spaces/blob/main/SoundSpaces2.md)와 [공식 논문 Appendix C](https://papers.nips.cc/paper/2022/file/3a48b0eaba26ba862220a307a9edb0bb-Paper-Datasets_and_Benchmarks.pdf)는 geometric propagation 후 pressure IR 합성을 설명한다. 논문은 early reflections(최대2bounce)의 plane grouping과 이산 응답, diffuse intensity envelope에 frequency-filtered noise/random phase를 사용하는 경로를 구분한다. 따라서 ‘모든 early reflection이 random-phase histogram이다’라고 단정하지 않는다. 이 설명은 설치 binary의 개별 path-time trace가 아니다.

설치 wrapper를 우회한 documented RLR C API에서 늦은 응답을 재현했다. Direct off도 늦은 return 유지, indirect off에서는 없어졌다. 따라서 원인은 추가 Habitat 후처리가 아니라 RLR 내부 indirect 응답/설정에 위치한다. Scattering0.5→0만 바꾸면 dominant timing이2m −2.4815ms/6m −6.0544ms 바뀌었다. Mixed-reflection setting dependence는 인과 대조로 확인했지만, diffuse energy weighting과 early-cluster representative time 및 synthesis filter의 개별 기여는 아직 분해할 수 없다.

Material JSON의 `speed=1483.961...`는 transmission medium의 값이다. Public header는 이를 재료 내부 전파용이라고 정의한다. **이를 공기 음속이나 fitted effective speed로 쓰지 않는다.** Transmission=false이고 이 값·density·damping·absorption은 explicit_default와 specular에서 동일했다. 바꾼 것은 scattering 값 두 개뿐이다. 실제 export의 absorption0.1/scattering0.5 또는0/transmission0을 확인했다. 이번 설정 개입은 renderer 원인 분리이며 재질 추정/Experiment4가 아니다.

## Internal path GT 접근 가능 여부

감사 범위: 설치 Habitat Python/C++ source, `RLRAudioPropagation.h`, RLR package repository/header/binary, exported/demangled symbols, public project 안내. [공개 RLR 저장소](https://github.com/facebookresearch/rlr-audio-propagation)의 배포 형태는 binary+public header다. 설치본에서 직접 읽을 수 있는 engine implementation source는 확보하지 못했다.

| 접근 대상 | 결과 / 의미 |
|---|---|
| Public C API | source/listener/radius/config, multi-listener, mesh OBJ, geometric firsthit, pressure IR channels/metrics를 사용 가능 |
| Path length/order/wall ID/travel-time | public header에 per-path export 없음; acoustic GT 미확보 |
| Internal intensity histogram | `ovrAudio_SceneResultGetPairIntensityIRBand/Format`, `outputEarlyIntensityIR...` 등 binary symbols는 존재하나 supported signature/header 및 안전한 scene-result ownership 접근이 없음 |
| Pressure synthesis | `buildPressureIR`, `IntensityIR::addPath` 등 이름 확인; 이름만으로 정확한 연산/지연을 증명할 수 없음 |
| Unsafe ABI 호출 | opaque layout/signature를 추측하지 않았으며 실행하지 않음 |
| 필요한 조건 | 버전 일치 internal debug API와 context/result access 또는 renderer source, path IDs·order·length·pre-pressure time bins export |

Viewer의 energy envelope는 **pressure²의 causal moving average**이며 내부 path energy histogram이 아니다. Histogram peak는 접근 불가로 정량 비교를 수행하지 못했다. 이것을 waveform-derived envelope로 대체했다고 표기하지 않는다.

## 원인 규명 수준

- **Verified:** geometry/units/wrapper passthrough; default material coefficients; wall/indirect dependence; scattering-only timing effect; receiver radius effect; stationary waveform/timing variability; no validated default estimator under frozen gate.
- **Consistent:** diffuse/early weighting or clustering plus pressure synthesis could explain timing movement. Scattering0 relative wall-shift error≤0.107ms supports limited tracking of intervention.
- **Unresolved:** exact internal acoustic path time/order/identity, synthesis component별 error, physical reality transfer, stable cross-mic complex phase interpretation. RLR 설정 효과를 현실 벽 반사 법칙이나 확정 backend bug로 일반화하지 않는다.

[정량표와 실패 조건](RESEARCH_REPORT.md) · [재현 출처](REPRODUCIBILITY.md) · [검산](verification.json)

# Step0 — existing research and input audit

| 기존 연구 | 실제 다룬 항목 | 특정 실제 벽의 예상 반사 시간 검증 여부 |
|---|---|---|
| [Acoustic Cues11/07](../2026-11-07-acoustic-cues/archive/METHODS.md) | E/C/N cues,raw windows,GCC,repeat/reverse render | 아니오. 명시적으로 peak에 wall identity를 할당하지 않음 |
| [Directional Structure11/09](../2026-11-09-directional-structure/AUDIT.md) | 방향 extent예측,pureV 후보,6ch raw controls | 아니오. Blocked footprint targets이며 echo-wall 대응 아님 |
| [Ray Joint11/09](../2026-11-09-ray-joint/PHASE1_REPORT.md) | ray capacity/learnability,oracle와joint 비교 | 아니오. Geometry oracle과acoustic predictability를 구분; 물리 path time annotation 없음 |
| [Track B Acoustic Geometry11/10](../../../track-b/experiments/2026-11-10-acoustic-geometry/pair_results.md) | architectural distance targets,furnished/stage/opening-preserved intervention | 아니오. Object removal이 directional prediction에 미치는 효과이며 특정 wall arrival 검증 아님 |
| Gibson planar likelihood (`scripts/gibson_runtime/planar_acoustic_likelihood.py`) | 72radial first blocked-cell returns,8kHz,100ms proxy,energy/peak likelihood | 실제 wall-path GT 검증 구현 아님. Docstring도 topology proxy임을 명시. 이번 finite specular path/intervention으로 대체 검증 |

## RIR와좌표 convention

관측은HSSD native furnished3D,FP reference는canonical map의2.8m extrusion. Sample48kHz; source는6mic ring 중심,반경0.05m; 각mic은omni mono로 순차 렌더. RING=[180,240,300,360,60,120]deg,offset=(.05cosθ,0,−.05sinθ). NativeY-up,X/Z수평. Raw base acoustic yaw와 query channel_roll은historical schedule에서 읽고,후보worldyaw에서θ=RING−yaw로예측한다. 원점/부호/회전 mapping과유한벽 time solver를별도 numerical check로검증한다.

Observed source=receiver center와1.5m height를metadata로확인. Actual waveform384개 전부 기존receipt SHA256과대조했다. Source XY/yaw는감사 metadata에서확인할수있지만prepared inference hypotheses에는넣지않았고,geometry selector는candidate pose만받는다. Direct sample7≈0.146ms는공통근거리source-ring구조이며위치별거리센서가아니다.

실제선택RIR는6×15751..40835,약328–851ms이다. 기존encoder의2–128ms제한과raw저장길이는다르다. 신규통제RIR는전체variable-length배열을보존했다. maxIRLength default4s는항상4s가저장된다는뜻이아니다.

Acoustics:20k indirect/512source rays,depth50,diffraction on,maxorder10,2threads,materials off,transmission off;temporal coherence defaultoff. 동일pose반복으로samplewisephase동기성은보장되지않음을확인했다. [공식API설명](https://github.com/facebookresearch/sound-spaces/blob/main/SoundSpaces2.md)은temporal smoothing의의미를설명한다. [SoundSpaces2논문](https://papers.nips.cc/paper/2022/file/3a48b0eaba26ba862220a307a9edb0bb-Paper-Datasets_and_Benchmarks.pdf)의energy-to-pressure합성설명은위상해석의한계에대한배경이며,이번backend지연의원인을입증하지는않는다.

## FP / architecture alignment

FP metadata:1cm raster,+Xright/+Zdown,native origin,wall source stage_mesh_slice,door/window/glass는free portal,semantic outside는폐쇄된다. FP GLB는recenter없이동일nativeframe을유지한다.0.1mwallthickness와semanticgapclosure0.2m로geometry오차가발생할수있다. Black cell은blocked/unknown이며physical wall로자동해석하지않는다.

FP GLB1.5m slice를정확한coplanarsegment로합쳤다. Stageverticalslice는이전TrackBtargets_v3의`segments_1.5`만사용하며pose/depth라벨을읽지않는다. parallel tolerance10deg,pointdistance0.20m로support를감사한다. 이tolerance는arrival시간약1.17ms왕복차이를허용할수있어±0.5ms검증을엄격하게만들고실패원인이된다. Nativearchitecture고정물/개구부재질은semanticwallGT가아니다.

Test는scene/family식별메타데이터로제외확인만했다. 기존TrackB공식Test산출물은열거나재사용하지않았다. Currentproject에이미있던Gibson두파일수정은보존하고이번Commit에서제외했다.

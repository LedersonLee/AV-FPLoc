# Experiment4A — Material-dependent Acoustic Observability

2026-11-11 · Track A · ID `material_observability_20261111` · **Outcome B — Material-dependent but Ambiguous**. A/B/C/D의 실험·분석을 수행했다. 증거 수준 **Exploratory**:3종 설계 기하의6위치, 재사용 Train HSSD2장면이며 independent Test나 실측 일반화가 아니다. 최종 산출물·브라우저 검증은 completion audit를 함께 읽는다.

## 핵심 결론

**재질 계수는 관측을 체계적으로 바꾸지만, 관측만으로 target wall의 존재와 재질 영향을 충분히 분리하지 못했다.** 흡수 증가의 에너지 감소와 high-frequency absorption의 spectral tilt 감소는6위치 모두 같은 방향이다. Scattering 효과는 기하에 따라 크기·방향이 달랐다. 고정 reliability rule의 target-wall 제거 오검출률70.6%는 다른 벽·바닥·천장 응답을 목표 벽의 증거로 오인하는 문제를 보여준다.

이것은 모든 reliability 방법의 불가능성 증명이 아니다. Rule fit을 single-wall2m에 한정했으므로 **fit-domain shift와 다중 반사**도 주요 경쟁 설명이다. 평가를 본 뒤 더 넓은 fit으로 바꿔 성공을 만들지 않았다. Absorption/material 효과의 원인은 교차 개입 A/B에서 분리했지만, reliability 실패를 material 하나만의 인과 효과로 쓰지 않는다.

## Phase0 — 기존 연구와 renderer 감사

[Experiment1](../2026-11-11-reflection-observability/RESEARCH_REPORT.md)은 geometry window의 energy 연관성과 first-order timing 미검증을 구분했다. [1B](../2026-11-11-reflection-timing-audit/RESEARCH_REPORT.md)는 default scattering0.5→0에 따른 dominant timing 변화를 발견했으나 path ID를 검증하지 못했다. [Acoustic Cues](../2026-11-07-acoustic-cues/results.md)의 E/C/N gain 불변성과 [TrackB Acoustic Geometry](../../../track-b/experiments/2026-11-10-acoustic-geometry/conclusion.md)의 방향 구조/거리 규모 구분을 검토했다. 역사적 Test 결론은 배경 감사로만 읽었고 이번 선택·평가에 Test payload를 사용하지 않았다.

설치된 RLR의 default absorption0.1/scattering0.5/transmission0,listener radius0.1m와5cm ring의 차이를 재확인했다. `materials off`는 nonsemantic mesh에 default acoustic material을 쓰는 동작이다. `Material name`이 아니라 category와 `labels` substring mapping이 적용된다. 실제 acoustic OBJ1980개에서 적용 coefficient를 검사했다.4개 backend band 중심은83.910/369.251/1624.911/7150.517Hz이며 full run profile을 여기에 맞췄다. [물리 감사](PHYSICAL_MATERIAL_AUDIT.md)

## Dataset / Evaluation Protocol와 대조

- Single-wall2/6m, room8×8×3m의2위치, corridor2.4×18×3m의2위치. Front panel present/removed/moved1m를 모두 렌더링했다. 방/복도의 floor·ceiling은 고정 default이며 vertical surfaces만 계수를 바꾼다. 제거벽은 target front panel 제거이지 모든 반사체 제거가 아니다.
- Absorption .05/.30/.70 × scattering0/.5/1의 필수9조건에 .99흡수 stress를 추가했다. Uniform/low/high profile3종 ×3geometry states×6poses×3fresh contexts = **1944rings**. Direct-only/indirect-only36rings를 추가해 **1980rings /11,880mono**. Context가 독립 RNG seed인지는 보장하지 않는다.
-48kHz,20kindirect/512source rays,depth50,diffractionon,transmissionoff,temporaloff,default4s maxIR 및0.1m listener radius를 고정했다. 실제 길이는 renderer가 반환한 길이를 보존했다. Source는5cm6mic ring 중앙이고 sample0을 이동하지 않았다.
- C fit은 single2 uniform108조건뿐이다. 다른 위치·frequency profile1836조건에 같은 rule/threshold를 적용했다. 이는 전체 pipeline 미노출 test가 아니다. 고정 raw amplitude gate1e−6 direct-peak power와 학습한 peak-threshold rule은 서로 다른 진단이다.
- 원시 절대 energy와 direct-normalized energy, unit-total spectral shape를 구분했다. E/C/N의 알려진 global gain 불변성 때문에 절대 energy 효과를 기존 encoder가 보존한다고 주장하지 않는다. 이번에 encoder를 학습하거나 전체 E/C/N을 재평가하지 않았다.

## A — Absorption × Scattering 효과

| Pose | 흡수 .05→.70 Δenergy dB | 산란0→1 Δenergy dB | interaction dB | High−Low tilt dB |
|---|---|---|---|---|
| corridor_far | -3.967 | +2.207 | -0.715 | -5.091 |
| corridor_near | -4.673 | +1.539 | +0.048 | -6.733 |
| room_far | -2.078 | -0.126 | -0.322 | -2.877 |
| room_near | -2.067 | +0.130 | -0.063 | -2.883 |
| single2 | -4.872 | +3.052 | -0.382 | -6.053 |
| single6 | -5.030 | +2.580 | +0.118 | -6.050 |

Energy는1–80ms의 channel 평균 power sum이다. dB는 mean log10 energy 차이×10으로, 각pose에서 균형 잡힌 조건 평균을 비교했다. Absorption effect는scattering3수준, scattering effect는필수absorption3수준(.99stress 제외), interaction은s=1과s=0에서의 absorption 효과 차이다. Profile tilt는a=.3/s=.5 high−low. 모든 비교는3fresh contexts를 평균하되 독립 장면 수로 세지 않는다.

흡수 효과 범위−5.03..−2.07dB, spectral tilt−6.73..−2.88dB로 방향은 일관됐다. 산란 energy 효과는room_far−0.126dB에서single2+3.052dB까지 달랐다. 산란에 따른 peak-error 변화도 single2/6에서는+3.64/+8.73ms, corridor에서는−1.39/−3.81ms로 반대다. 이 값은 dominant peak의 교체/혼합을 포함하며 특정 경로의 travel-time 변화가 아니다.

단일벽 정반사에서 α .70/.05 energy ratio는2m0.347/6m0.309였다. Energy fraction 관계(1−.70)/(1−.05)=0.316에 가깝고, pressure fraction을 그대로 제곱한0.100과 다르다. 이것은 제한된 관측 response 대조이며 전체 입사/반사 flux 보존을 측정한 것은 아니다. Reflection amplitude를 별도 slider로 독립 조절하지 않는다.

거리별 decay, direct/indirect suppression, channel 상대energy, band late/early ratio는 [추가 진단](diagnostics.json)에 있다. Channel timing/phase는 순차 mono 렌더의 일관성 진단이지 동시 수음 phase GT가 아니다. Repeat SD와 wall-movement error72교차 조건, aligned/shifted/matched-time window 결과는 [전체 집계](quantitative_summary.json)에 보존했다. Matched-time donor는 서로 다른2/6m군을 교환해 시간 분포를 유지하고 정렬은 깨뜨렸다.

## B — Frequency-dependent reflection

각profile은 실제 backend4center에서 평균 coefficient α를 같게 맞췄다. Low는낮은2band에α+δ,높은2band에α−δ; High는반대, δ=.8min(α,1−α). Incident-spectrum-weighted absorbed energy는 같다고 보장하지 않는다. Native interpolation kernel은 공개되지 않아 band center에 직접 anchor했다.

| 분리 조건 | Full energy | Band energy | Normalized shape | Profile shuffle |
|---|---|---|---|---|
| context_holdout | 33.3% | 55.6% | 72.2% | 27.8% |
| hold_single | 33.3% | 33.3% | 50.0% | 0.0% |
| hold_room | 33.3% | 44.4% | 44.4% | 61.1% |
| hold_corridor | 33.3% | 66.7% | 100.0% | 16.7% |

알려진3개의 **시뮬레이션 profile**에 대한 nearest-centroid 진단이다. Context holdout은동일pose/geometry의2contexts fit→1context 평가(18건)라 독립 scene test가 아니다. Leave-family-out도 설계 기하3종뿐이며 각평가18건이다. Full energy만으로는33.3%, normalized shape는context holdout72.2%였으나 held-room44.4%로 떨어졌다. Shuffle baseline도 소표본에서변동이 크므로 실제 재질 식별이나 강한 일반화를 주장하지 않는다.

Normalized shape 거리의 중앙값은 동일geometry·다른profile0.0465, 다른geometry·동일profile0.0291이며 범위가 겹친다. Spectrum 차이와 geometry 차이는 분리되지 않는다. Band별late/early ratio도 profile별로 기록했다. 이는 유한 window의 decay proxy이지 band RT60 GT가 아니다.

## C — Observability / Reliability

| Rule | wall-state TPR | 제거벽 FPR | FNR | AUC | Brier |
|---|---|---|---|---|---|
| peak | 100.000% | 70.588% | 0.000% | 0.4985 | 0.2512 |
| sharp | 100.000% | 70.588% | 0.000% | 0.6596 | 0.3323 |
| time | 99.837% | 70.588% | 0.163% | 0.6204 | 0.4110 |
| band | 99.837% | 70.588% | 0.163% | 0.6541 | 0.2964 |
| channel | 99.837% | 70.588% | 0.163% | 0.6867 | 0.3837 |
| combined | 99.755% | 70.588% | 0.245% | 0.5737 | 0.3731 |

위 TPR/FPR의 GT는 **target-wall intervention 상태**다. 실제 first-order path GT가 없어 ‘진짜 특정 반사 경로 검출률’은 미검증이다. Room/corridor 제거조건의 다른 반사는 실제로 존재하므로 FPR은 소리 자체가 가짜라는 뜻이 아니라 목표 벽 귀속의 실패다. Pooled1836을 독립 표본으로 세지 않으며 pose/material/profile별 표가JSON에 있다.

Combined는5scores의 fit empirical percentile 평균이다. Single2 uniform에서FPR≤5%를 만족하며 FNR을 최소화하는 threshold를 고정했다. 다른조건에서 peak rule 대비 FPR은 줄지 않았고 Brier는0.251→0.373으로 나빠졌다. H3의 ‘결합하면 reliability가 더 좋다’는 강한 형태는 **이번 규칙·fit 범위에서 지지되지 않았다**. 평가를 본 뒤 threshold를 재선택하지 않았다.

Known α/scattering으로 threshold를 나눈 **Oracle**도TPR82.1%,FPR70.6%다. 이것을 material-free inference와 섞지 않았다. Unknown material의 단순 RIR rule과 oracle 모두 대상 벽 귀속을 해결하지 못한다.

필수 .05/.30/.70 clean present 조건에서 raw gate 미검출은 없었다. 약한 신호 반례는 사전 지정 .99stress의 **single6,wall moved(실제거리7m),s=0,high-profile**에서나왔다:3contexts 중2개에서6채널 모두 raw gate 미검출, 나머지조건의 peak-rule 통과와도 구분한다. 이 극단조건을 일반 재질의 실패율로 외삽하지 않는다. Viewer에는context0 실제 실패를 포함했다. 미검출을 벽 부재로 읽지 않고 Unknown/Unobservable로 남긴다.

### Noise 보조 대조

| Noise | Peak FPR | Combined FPR | Combined AUC |
|---|---|---|---|
| clean | 0.667 | 0.667 | 0.658 |
| moderate | 1.000 | 0.833 | 0.656 |
| severe | 1.000 | 1.000 | 0.647 |

108개 동일 clean recording에 합성 Gaussian noise를 추가한 별도 보조 분석이다. Moderate/severe는pose별고정 baseline early RMS 대비20/0dB이며 material별실효SNR을 따로 기록했다. 실제 renderer output처럼 표기하지 않았고 explorer의 waveform에는 noise를 섞지 않았다. Clean에서도다중반사 FPR이높고 noise는 이를더높인다. Spectral tilt 변화분포도 별도기록했다. Material과noise를 같은기전으로 단정하거나 어느쪽이실환경에서더크다고 주장하지 않는다.

## D — HSSD transfer feasibility

Train2scenes×2poses×6coefficients×3contexts=**72native-stage rings**를 실제 생성했다. Native GLB source좌표를 보존하고 |normal_y|<.25인vertical face proxy에만계수를지정했다. Stage/object assets는분리되지만 이proxy는검증된architectural-wall segmentation이아니다. Furniture/object instances는이번쌍에서제외했다. Stock semantic-material mapping은이setup에서검증되어있지않으며 furnished전체로확장성공을주장하지않는다.

| Scene/query | α .70/.05 energy ratio | s .5/0 energy ratio | 6조건 최고 evidence 후보 rank |
|---|---|---|---|
| 102344250/0 | 0.448 | 1.074 | [13, 13, 13, 13, 13, 13] |
| 102344250/145 | 0.436 | 1.185 | [15, 16, 15, 15, 16, 15] |
| 102816615/16 | 0.280 | 1.339 | [3, 7, 7, 7, 7, 7] |
| 102816615/36 | 0.391 | 1.431 | [19, 8, 12, 8, 8, 8] |

조건 순서는(.05,0,uniform),(.05,.5,uniform),(.70,.5,uniform),(.30,.5,uniform/low/high)이다. 기존pure-visual Top20의후보geometry를사용하고GTpose로window를고르지않았다. FP·RGB·native source자산을수정하지않았다. 동일pose/material변경만으로earlyenergy와candidate evidence 순위가바뀐다. 최고evidence후보는4poses중3곳에서조건에따라바뀌었다. Rank는untrained window-energy 진단의순위일뿐 정답후보·localization개선이아니다. Supported/unknown geometry별evidence와top-margin은JSON에보존했다.

## 대표성과 Interactive Explorer

전체6poses에서absorption log-energy/scattering log-energy/scattering peak-error/profile tilt4효과벡터를먼저계산했다. 각차원의중앙값과MAD로표준화해중앙에가장가까운 **corridor_near**를기본화면으로선정했다(거리1.656). 가장극적인pose를선정하지않았다. Room_far는효과구조가가장다른예외(5.276),single6는실제weak-wall nondetection을보여주는추가예외다. **한pose만으로충분하지않다.** 원래6poses분포·거리·선정식은[대표성집계](quantitative_summary.json)에있다. HSSD는별도native-stage감사이며대표toy가HSSD전체를대표한다고하지않는다.

[Interactive HTML](interactive.html)은3poses×108실제조건의context0,6channels를제공한다. A/B coefficient/profile/wall selection,6개별/전체channel,같은시간원점·공통amplitude,선택적정규화,wave/envelope/EDC/spectrum/bandenergy/metrics/실패사례를갱신한다. 전송계수0,reflection독립slider없음. 데이터는저장RIR의display extrema로새신호생성·보간은없다. RT60은충분한길이·dynamic-range·fit gate를통과한T20 extrapolation에만조건부표시하고나머지는N/A다.

## Supported / Rejected / Unverified

- **Supported:** 같은geometry에서absorption/scattering/profile이RIR을변화시킴;흡수·spectral tilt방향이6poses에서일관됨;native-stage surface assignment기술적가능성;targetwall과무관한반사로인한귀속모호성.
- **Not supported under tested protocol:** 단순결합rule이peak보다reliability/calibration을일반적으로개선한다는주장;material정보oracle만으로targetwall ambiguity해결.
- **Unverified:** per-path first-order identity/timing,실제재질분류,physicalsimultaneousphase,실환경일반화,furnishedHSSD exact semanticmapping,LocalizationUtility. 후자는Experiment2범위이며실행하지않았다.

6개위치/3설계geometry와2개재사용Train scene밖에없다. 반복context의RNG독립성·upstream노출은배제하지못한다. Late window는662records에서400ms전에끝나며censored로표기했다. Feature totalenergy는동일ring내최단채널공통support,viewer EDC는각채널native support이다. Small interaction을확정적인물리법칙으로승격하지않는다. 새Network/GEOQ/fusion학습·Experiment2/3은수행하지않았다. Outcome B는정상적인부정/제한결론으로보존하고이실험에서종료한다.

[재현·비용·실패기록](REPRODUCIBILITY.md) · [원리](METHOD_AT_A_GLANCE.md) · [검증](verification.json)


## 전체 측정값과 단위

[648개 조건별 측정 요약](measurement_summary.json)은 각 조건의3개 fresh context에 대해 peak/onset, first–strongest peak 차이, sharpness, temporal spread, direct/early/late energy, early-to-late ratio, 채널 timing/band/phase 일관성의 평균·범위·context SD를 제공합니다. 주파수 대역별 energy와6채널 상대 energy도 포함합니다. Context SD는 독립 장면에 대한 신뢰구간이 아닙니다.

여기서 absolute energy는 정규화하지 않은 renderer float sample의 제곱합입니다. 보정된 Joule이나 dB SPL이 아닙니다. 화면의 spectrum dB도 동일 raw power 기준이며 실제 음압 레벨로 해석하지 않습니다.

## HSSD 후보 간 evidence contrast
아래 값은 supported-window energy의 `(최대−차상위)/최대`입니다. 정답 후보와의 구별 정확도는 아니며, GT-free 에너지 대비가 얼마나 달라지는지만 진단합니다.
| Scene/query | α=.05,s=.5 | α=.70,s=.5 | 변화 |
|---|---:|---:|---:|
| 102344250/0 | 0.2667 | 0.2770 | +0.0102 |
| 102344250/145 | 0.0697 | 0.1491 | +0.0794 |
| 102816615/16 | 0.0994 | 0.1037 | +0.0042 |
| 102816615/36 | 0.3730 | 0.0854 | -0.2876 |

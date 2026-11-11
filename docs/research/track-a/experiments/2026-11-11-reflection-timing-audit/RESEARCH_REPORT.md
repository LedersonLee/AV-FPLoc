# Experiment 1B — Reflection Timing & Renderer Physics Audit

ID: `reflection_timing_audit_20261111` · 2026-11-11 · Track A · 완료 · **Outcome C — Renderer-dependent Artifact**. 증거 수준 **Exploratory**: 재사용된 통제 장면과 사후 renderer 원인 진단이며 독립 Test 검증이 아니다. 설정 개입의 인과 근거와 현실 음향 일반화는 구분한다.

## 1. 핵심 결론

기존 큰 시간 불일치는 단순 peak 선택 문제가 아니라 **설치된 렌더러의 기본 mixed-reflection 설정에 강하게 의존**했다. 실제 export에서 기본 scattering=0.5를 확인했고, 다른 material 필드·좌표·ray budget을 유지한 scattering=0 대조에서 지배적 peak 오차가 2m **+2.347→−0.134ms**, 6m **+6.168→+0.114ms**로 변했다. JSON을 명시적으로 읽힌 default=0.5 대조는 기존 늦은 응답을 재현했다. 설정을 바꾼 인과 효과는 확인됐지만, 내부 path travel-time → intensity histogram → pressure IR 중 어느 연산이 잔여 오차를 만드는지는 확인되지 않았다.

**정확한 first-order path / wall identity 검증은 아직 성립하지 않는다.** Scattering=0에서도 point-receiver peak 오차 +0.154/+0.409ms, 기본 radius에서 최대 채널 오차 0.142/0.139ms로 사전 절대 허용치0.10ms를 넘는다. 벽 이동 상대 변화는 좋아져도 절대 시간 정렬·내부 경로 귀속을 대신하지 못한다. D의 물리 gate 실패에 따라 HSSD 재평가는 생략했다. 이것은 요청된 정상 종료 조건이다.

## 2. 이전 실험에서 남은 문제

[Experiment 1](../2026-11-11-reflection-observability/RESEARCH_REPORT.md)의 결과는 그대로 보존했다. 기존96개6ch RIR 해시를 전부 확인하고 diagnostic60개의 기존 peak를 오차0으로 재생했다. 2m 기대11.662ms/관측14.010ms, 6m 기대34.986ms/관측41.149ms, 벽1m 이동의 기대약5.83ms와 관측 차이를 이번 신규 발견으로 세지 않는다. Sample rate/depth/diffraction/ray count/mesh 대조도 재렌더링하지 않았다.

기존 HSSD12장면384query/7,680후보의 matched-marginal power 차이 +0.26194 [scene bootstrap 95% CI +0.16714,+0.36814] 및 coherence 개선 미확인도 **이전 결과**다. 그 통계적 에너지 연관성은 물리적 first-order timing 증명이 아니다. 이번에는 HSSD payload나 Test를 재평가하지 않았다.

## 3. 가설별 검증 결과

| 가설 | 새 검증 | 판정 |
|---|---|---|
| H1 peak 해석 오류 | 12추정량, 3고정 threshold, 제거/이동/반복/거리 비교 | Onset도2m +2.327ms/6m +6.143ms. Peak를 onset으로 바꾸는 것만으로 기존 큰 지연은 해결되지 않음. Causal band filter의 직접음 ringing은 제거벽에서도 검출됨. |
| H2 renderer 합성/설정 | Habitat 우회 C API, listener radius, indirect/direct suppression, scattering-only 대조, 공개 소스/API/심볼 감사 | 큰 mismatch의 **설정 의존성 Verified**. 내부 histogram/pressure 연산별 지연 기여는 **Unresolved**. |
| H3 geometry/좌표 | 실제 acoustic OBJ, ray first hit, source/listener receipt, 독립 Fermat 최소화36경로, analytic fractional reference | 좌표·단위·기하 solver는 검산 통과. 외부 wrapper resampling/delay 없음. acoustic path time/order GT는 미확보. |
| H4 stochastic/channel | 동일 위치5회, 3fresh contexts, temporal on/off, 6listener 동시context | timing/파형 변동 확인. Temporal on 및 동시context도 default timing bias 제거 못함. 복소 채널 유사도를 물리 동시 수음 coherence로 해석 불가. |

환경은 48kHz, unitScale1, source=(0.17,1.5,d)m, 반경0.05m의6mic, 벽z=0 또는−1m, c=343m/s(독립 물리 예측값)다. Source radius0, **renderer listener radius0.1m는 mic ring radius와 다른 개념**이다. 신규72contexts에서420mono outputs(70ring-equivalents; stationary repeats 포함)를 생성했다. 전체 설정·원시 해시는 [재현 문서](REPRODUCIBILITY.md)에 있다.

기존 single-wall2m는 diagnosis/development,6m 및0.3m/removed/moved는 threshold를 고르지 않은 보고 조건이다. 모두 과거 관찰 이력이 있으므로 독립 holdout이 아니다. Primary threshold는 direct-reference power의1e−6, sensitivity1e−8/1e−4를 모두 보존했다. 탐색은 공통1–80ms이며 예상 도달 시각 주변만 검색하지 않았다. Acceptance는 절대오차≤0.10ms, 이동오차≤0.20ms, 반복SD≤0.10ms, 제거FDR≤5%다. 조건별 best threshold 선택이나 effective speed/delay fitting은 없다.

## 4. Timing / Reflection Evidence 정량표

### 새로 확인한 scattering 설정 개입

| 벽 거리 m | 설정 | 평균 peak 오차 ms | 최대 절대 오차 ms | 채널별 반복 SD 평균 ms | 3 context 평균 오차 범위 ms |
|---|---|---|---|---|---|
| 2 | explicit_default | +2.3470 | 2.4617 | 0.0295 | +2.3447, +2.3516 |
| 2 | specular | -0.1345 | 0.1424 | 0.0040 | -0.1380, -0.1310 |
| 2 | specular_radius0 | +0.1537 | 0.1701 | 0.0000 | +0.1537, +0.1537 |
| 2 | specular_moved | -0.0833 | 0.0983 | 0.0080 | -0.0833, -0.0833 |
| 6 | explicit_default | +6.1684 | 6.4101 | 0.0866 | +6.1186, +6.2193 |
| 6 | specular | +0.1140 | 0.1393 | 0.0080 | +0.1082, +0.1186 |
| 6 | specular_radius0 | +0.4091 | 0.4309 | 0.0020 | +0.4068, +0.4103 |
| 6 | specular_moved | +0.1754 | 0.2251 | 0.0160 | +0.1731, +0.1801 |

각 행은3fresh contexts×6channels. 마지막 열은 context별 평균의 min/max이며 신뢰구간이 아니다. 채널/반복을 독립 장면으로 세지 않았다. RNG seed를 명시 제어하는 API를 사용하지 않았으므로 fresh context를 독립 seed로 보장하지 않는다.

Scattering=0의 wall+1m 이동에서 peak shift error 평균/최대는2m **+0.051/0.065ms**,6m **+0.061/0.107ms**로 상대 이동 기준을 충족한다. 그러나 moved6m peak 최대 절대오차0.225ms, onset 최대0.204ms이며 전체 절대 기준은 실패한다. 이 대조의 별도 specular-wall-removed 렌더는 수행하지 않았으므로 default-wall-removed 결과를 specular FDR 검증으로 전용하지 않는다.

### 기존 RIR에 적용한 새 추정량 비교

| 거리 m | 추정량 | 평균 오차 ms | 최대 이동 오차 ms | 제거 FDR | 반복 SD 평균 ms |
|---|---|---|---|---|---|
| 2.0 | wave_peak | +2.3482 | 1.3150 | 0% | 0.0283 |
| 2.0 | energy_onset | +2.3273 | 1.2944 | 0% | 0.0283 |
| 2.0 | short_increase | +2.3273 | 1.2944 | 0% | 0.0283 |
| 2.0 | band_low_onset | -10.6623 | 5.8308 | 100% | 0.0000 |
| 2.0 | band_high_onset | -10.6623 | 5.8308 | 100% | 0.0000 |
| 2.0 | energy_centroid | +2.4363 | 1.3539 | 0% | 0.0282 |
| 2.0 | envelope_peak | +2.4315 | 1.3150 | 0% | 0.0283 |
| 2.0 | removed_difference_onset | +2.3273 | 1.2944 | 0% | 0.0283 |
| 2.0 | band_low_difference_onset | +2.6190 | 1.5650 | 0% | 0.0263 |
| 2.0 | band_high_difference_onset | +2.3482 | 1.3361 | 0% | 0.0283 |
| 6.0 | wave_peak | +6.1637 | 1.1066 | 0% | 0.0434 |
| 6.0 | energy_onset | +6.1429 | 0.3983 | 0% | 0.0434 |
| 6.0 | short_increase | +6.1429 | 0.3983 | 0% | 0.0434 |
| 6.0 | band_low_onset | -33.9856 | 5.8309 | 100% | 0.0000 |
| 6.0 | band_high_onset | -33.9856 | 5.8309 | 100% | 0.0000 |
| 6.0 | energy_centroid | +6.2180 | 0.6942 | 0% | 0.0433 |
| 6.0 | envelope_peak | +6.2471 | 1.1066 | 0% | 0.0434 |
| 6.0 | removed_difference_onset | +6.1429 | 0.3983 | 0% | 0.0434 |
| 6.0 | band_low_difference_onset | +6.6221 | 0.4608 | 0% | 0.0434 |
| 6.0 | band_high_difference_onset | +6.2262 | 0.3983 | 0% | 0.0434 |

FDR은 제거벽6채널에서의 검출 비율로, 일반 환경의 false-discovery probability 추정이 아니다. 제거 waveform 자체와 차분한 방법의 FDR0은 구조상 당연하며 독립 음성대조 성능으로 해석하지 않는다. 0.3m onset 오차+0.571ms/이동 오차최대0.861ms로 가까운 거리도 실패했다. 나머지 relative1%/10%, threshold sensitivity 및 모든 조건은 [전체 요약](estimator_summary.json)에 보존했다. 6m onset의 평균 이동 오차−0.011ms만 보면 성공처럼 보이나 최대채널0.398ms로 기준을 넘는다.

### Backend 및 채널 대조

Default C API onset 평균 오차2m +2.325ms/6m +6.156ms로 Habitat를 우회해도 재현된다. Radius0에서는+2.612/+6.435ms; direct off +2.323/+6.158ms; temporal on +2.422/+6.267ms; simultaneous6listener +2.310/+6.163ms. Indirect off에서는1–80ms amplitude-gated detection0/18채널(각 거리). All-indirect suppression이지 order1 isolation이 아니다. [집계](backend_summary.json)

동일위치5회×3context의 mean onset SD는2m temporal off/on **0.060/0.041ms**,6m **0.147/0.077ms**. 시간축을 재정렬하지 않은 파형 cosine 평균은2m **0.069/0.170**,6m **0.068/0.042**였다. Temporal coherence가 모든 파형 일관성을 개선한다는 주장은 지지되지 않는다. Context 간 첫 파형 일부는 거의 동일하여 seed 독립성을 가정할 수 없다. 다른 mic 사이에는 물리적 전파차도 있으므로 phase/cosine을 정답 coherence로 사용하지 않는다. [채널 감사](channel_audit.json)

## 5. 밝혀진 원인과 여전히 가능한 대안

**Verified:** 기본 mixed reflection의 scattering 설정이 dominant return timing에 수 ms 인과 효과를 가진다. 명시 default loading, export coefficient, 입력 해시, 동일 geometry로 확인했다. 이 결론은 설치된 binary/geometry/budget에서의 설정 효과다. **Consistent:** early-path clustering, diffuse-path weighting 및 pressure synthesis가 함께 관측 peak를 바꿀 가능성은 공개 설명과 부합한다. **Unresolved:** 내부 travel-time 자체 오류인지, path grouping/선택인지, intensity-to-pressure 필터/phase인지의 기여 분리는 공개 경로 GT 없이 불가능하다. Scattering이 바뀌면 실제 반사 에너지 분포도 바뀔 수 있으므로 모든 차이를 ‘버그’로 단정하지 않는다.

Point receiver 전환은 약0.29ms 변화를 만들지만 기존+2–6ms 전체를 설명하지 못한다. Direct ring peak는sample7=0.1458ms로0.05m/343≈0.1458ms와 맞지만, 장거리 직접 경로의 이전 diagnostic은 약−0.33/−0.39ms 오차가 있어 전체시간축 calibration으로 사용할 수 없다. 어떤 값도 보정에 쓰지 않았다.

실패는 보존했다: full-RIR band onset은 직접음 filter-memory로100% 제거 검출, paired-band도 물리 오차 지속; 첫 C API 초기화/경로 처리 실패; 필수 material 필드 누락2회; 분석 필드 수정·OBJ marker 검증 실패 후 정정. [실패와 수정 이력](REPRODUCIBILITY.md#실패-시도와-정정)

## 6. HSSD 활용 가능성

D **미수행**: A–C에서 default observation에 적용 가능한 물리적 arrival evidence를 확보하지 못했다. Scattering0 새 렌더가 기존 HSSD RIR을 검증하거나 교정하지 않는다. 기존 candidate energy enrichment는 보존하되, 이를 wall별 first-order consistency로 승격하지 않는다. Candidate GT pose 선택·새 localization score·GEOQ 변경·네트워크 학습은 없다. HSSD teacher Train116/Val19 및 acoustic Train12/Val3 노출과 과거 평가 재사용 한계도 이전 결과에 그대로 적용된다.

## 7. 다음 실험 진행 여부 및 근거

Experiment1B는 완료하고 종료한다. Experiment2/3/4는 실행하지 않았다. 다음 연구를 하려면 내부 early-path/intensity-time export를 지원하는 검증 가능한 backend나 실제 측정이 필요하다. 예측 가능한 전이는 ‘좌표가 맞아도 scattering/default radius가 다르면 pressure peak timing이 달라질 수 있다’이며, 현실의 wall timing 일반화는 아직 미검증이다.

Baseline은 retained single-wall/default backend, 독립 analytic reference, removed/moved/repeat/suppression 대조다. B2/GEOQ/UnLoc accuracy baseline과 localization ablation은 이번 물리 감사에 적용 불가하다. 실패·한계로 종료할 수 있다는 사용자 조건을 충족하며 framework 개발 성공을 종료 조건으로 삼지 않았다.

[물리 감사](PHYSICAL_TIMING_AUDIT.md) · [원리 설명](METHOD_AT_A_GLANCE.md) · [실제 RIR 인터랙티브 화면](interactive.html) · [검증 결과](verification.json)

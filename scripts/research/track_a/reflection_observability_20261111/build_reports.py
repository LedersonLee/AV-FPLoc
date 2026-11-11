"""Build manuscript-style reports from immutable local evidence summaries."""
import json,argparse,shutil,subprocess,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--docs',type=Path,required=True);a=p.parse_args();root=a.root;docs=a.docs;docs.mkdir(exist_ok=True,parents=True)
 read=lambda n:json.loads((root/n).read_text());s=read('hssd_summary.json');c=read('controlled_analysis.json');t=read('timing_diagnosis.json');groups=read('matched_subgroups.json')
 def contrast(control,metric='power_ratio',rank='all20',half=.5,support=True):return next(x for x in s['contrasts'] if (x['control'],x['metric'],x['ranks'],x['half_ms'],x['supported'])==(control,metric,rank,half,support))
 def ci(x):return f"{x['mean']:.4f} [{x['ci95'][0]:.4f}, {x['ci95'][1]:.4f}]"
 def table(rank):
  rows=['| 대조군 | 정렬 창 평균 | 대조군 평균 | paired scene 차이 [95% CI] |','|---|---:|---:|---:|']
  for ctl in ['random','shifted','other_candidate','matched_marginal','full_early']:
   x=contrast(ctl,rank=rank);rows.append(f"| {ctl} | {x['aligned_mean']:.4f} | {x['control_mean']:.4f} | {ci(x['difference'])} |")
  return '\n'.join(rows)
 subgroup='\n'.join(['| 조건 | 정렬−분포보존 대조 power 차이 [95% CI] |','|---|---:|']+[f"| {x['field']}: {x['group']} | {ci(x['matched_power_difference'])} |" for x in groups])
 coherent=contrast('matched_marginal','coherence');timing=contrast('matched_marginal','timing_residual_ms')
 report=f'''# Track A Experiment 1 — Geometry-guided Reflection Observability

2026-11-11 KST · **완료: 핵심 물리 시간 정렬 가설 음성, HSSD 조건부 에너지 연관성 양성** · 증거 수준 **Exploratory**. 독립 Test 검증 아님. Experiment 2–4, network 학습, localization scorer·reranking·accuracy 평가는 수행하지 않았다.

## 결론

**현재 SoundSpaces 생성 설정에서 c=343m/s로 계산한 좁은 first-order reflection window가 실제 벽 반사를 포착한다는 핵심 가설은 지지되지 않았다.** 단일 벽 제거·이동에 따라 변하는 음향 응답은 존재하지만, 예상 시간보다 늦게 나타난다. 2m·6m 단일 벽에서 반복 측정한 peak timing bias는 약+2.36ms·+6.15ms다. 반면 HSSD의 후보 기하 정렬 창은 같은 시간·거리 분포를 보존한 다른 질의 대조보다 평균 정규화 power +0.2619 [0.1671,0.3681] 높았다. 이 결과는 **기하와 관측 RIR의 조건부 연관성**이며, 정확한 벽 반사 대응이나 localization 개선의 검증이 아니다.

[물리 검증](PHYSICAL_VERIFICATION.md) · [원리와 pipeline](METHOD_AT_A_GLANCE.md) · [Interactive HTML](interactive.html) · [입력 감사](INPUT_AUDIT.md) · [재현과 비용](REPRODUCIBILITY.md) · [사전 protocol과 후속 개입](PROTOCOL.md).

## 가설·경쟁 설명·증거 구분

- 관찰: 벽 제거/이동으로 단일 벽 early response가 달라지고 dominant return이 늦어지는 경향이 있다. c343±0.5ms 창에는 attributable energy가 사실상 없다.
- 통제에 기반한 결론: 단일 벽 응답은 벽 의존적이지만, 현재 renderer의 좁은 예상 시간 대응은 실패한다. 신호의 peak를 벽 GT로 지정하지 않았다.
- 관찰: HSSD에서 정렬 창의 energy enrichment가 random/shift/matched-marginal 대조보다 높다. coherence는 matched 대조보다 낮다.
- 해석: 후보 기하가 관측의 시간별 에너지 구조와 연관될 가능성이 있다. 시각 후보 선택, 가구/천장/바닥·고차 반사·공통 early decay·FP 정합 오차가 대안 설명이다.
- 미검증 인과 가설: renderer 내부 반사 시간 생성 또는 합성 방식이 지연을 만든다. 공개 API에는 path GT/내부 travel-time trace가 없어 특정 backend 원인을 확정하지 못했다. 지연을 학습·보정해 HSSD 창을 이동시키지 않았다.

## Dataset / Evaluation Protocol

기존 directional-structure의12 Train 장면에서 순수 visual NMS Top20을 그대로 사용했다. 장면별128 질의 중 균등32개를 사전 선택해 총384질의·7680후보를 분석했다. 후보의 XY/yaw만으로 반사 경로를 계산했고, query GT XY/yaw를 시간창 선택·filter에 사용하지 않았다. 질의의 채널 회전은 acquisition schedule의 관측 좌표 convention이다. 센서 높이1.5m는 알려진 acquisition 가정이다.

FP extrusion의 유한 수직 segment를 image-source로 반사하고, specular point가 벽 내부에 있으며 두 leg가 다른 FP 벽에 가리지 않은 경로만 남겼다. 반사점이 평행한 native architectural vertical slice에서0.20m 이내이면 `supported`; 나머지는 `unknown/closure`로 분리했다. 이는 architectural geometry의 추가 확인을 사용한 **감사된 FP 경로**로, FP-only 배포 성능 주장이 아니다. Stage fixed built-in도 포함될 수 있어 semantic wall GT와 같지 않다.

전체 유효 경로28902개 중 공통 분석 범위(각 채널 중심2–79ms)에22295개 supported,4749개 unknown/closure가 있었고1858개는 범위 밖이었다. 실패한 finite/visibility proposals는 생성 단계에서 제거됐으며 그 전체 개수는 따로 계수하지 않았다. Raw RIR 길이는15751–40835 samples (328.15–850.73ms), 모두80ms 분석에 충분했다. 기존 HSSD를 재렌더할 필요가 없었다.

Bootstrap10000회, seed20261111, query 내 candidate 평균→scene 평균→paired scene 재표집. 경로·채널·seed는 독립 표본 수가 아니다. 세 random/donor seed42/43/44를 평균했다. Teacher Train116/Val19 노출과 재사용 연구 장면을 명시한다. Scene ID 및 family 기준으로 Test 제외를 확인했으며 Test payload/결과는 열지 않았다. CI는 다중 비교 보정 없는 탐색적 scene 변동성이다.

## Baseline / Ablation과 정량 결과

평균 squared amplitude를 각 관측 full early1–80ms 평균으로 나눈 **power ratio**를 비교한다. Total energy는 창 길이가 달라 full early와 직접 비교하지 않는다. c343,반폭0.5ms가 primary이며0.25/1ms는 사전 민감도 분석이다. No-guidance full early는 power ratio=1로 정의된다. Peak energy,sharpness=peak/mean,aligned waveform absolute correlation,inter-channel peak residual도 함께 계산했다. 약한 창의 argmax는 noise maximum일 수 있어 detection으로 해석하지 않는다.

### 모든 visual 후보, architecture-supported FP paths

{table('all20')}

다른 후보는 rank를 한 칸 순환한다. 모든20후보 평균에서는 이 대조가 대수적으로 거의 상쇄되므로 작은 차이를 후보 구분의 증거로 해석하지 않는다. 일부 후보의 경로 부재 때문에 paired subset 평균이 살짝 다르다.

### 사전 고정된 Visual Top1 후보만

{table('top1')}

Top1 vs 다음 후보 power 차이는 양수지만, 실제 정답/오답 분별률이나 localization 성능을 계산한 결과가 아니다. 후보 rank 선택은 visual-only이며 best acoustic rank를 택하지 않았다.

Matched 대조 대비 coherence 차이는 {ci(coherent['difference'])}; timing residual 표준편차 차이는 {ci(timing['difference'])}ms다. 낮은 timing residual은 좁은 창 내부 argmax의 기술통계이고 실제 wall timing GT 정확도가 아니다. **에너지 이득이 위상 일관성 이득을 동반하지 않았다.**

### 거리·방향·중첩별 실패 조건

{subgroup}

방향은 candidate-relative X/Z bearing quadrant이며 yaw 오차·시각 후보 편향과 혼합된다. 물리적 센서 방향성으로 일반화하지 않는다. <1m·1–3m strata CI는0을 포함한다. ≥3m는 작은 양의 차이이며 near/far 비교는 서로 다른 path/query 구성이다. Strata는 path→query→scene 평균으로 계산해 primary candidate-balanced 평균과 수치가 같지 않을 수 있다. 중첩 조건의 더 큰 에너지는 여러 응답의 합일 수 있으며 특정 벽 식별이 더 쉽다는 뜻이 아니다. 대표 scene별 값은 [전체 집계](hssd_summary.json)에 있다.

Supported matched-marginal power 차이는 반폭0.25ms +0.2333 [0.1128,0.3753],0.5ms +0.2619 [0.1671,0.3681],1ms +0.3672 [0.2938,0.4501]다. Unknown/closure 반폭0.5ms는+0.1374 [−0.0172,0.2714]로 불확실하며1ms에서 양수다. 따라서 넓은 창의 양성 결과만으로 물리 wall support를 증명할 수 없다.

## 필수 질문에 대한 답

1. **실제 벽 반사의 acoustic evidence가 있는가?** 통제 renderer에서 벽 의존적 return은 있다. 그러나 좁은 c343 first-order window와의 대응은 실패했다. HSSD에서는 wall identity GT가 없어 첫 반사라고 확정하지 못한다.
2. **어느 거리·방향·장면에서 관측 가능한가?** 단일 벽0.3/2/6m 모두 벽 의존적 response가 있고 timing mismatch도 있다. HSSD12장면과 위 strata에서 조건부 power 차이를 측정했다. 장면 밖 일반화는 미검증이다.
3. **약하거나 겹치면?** 수치 noise floor에도 argmax·큰 dB ratio가 생긴다. 룸/복도에서는 다른 벽·고차 응답 때문에 wall removal이 예상 창 power를 오히려 높일 수 있고 특정 벽 귀속이 불가능하다. Sequential mono 렌더의 phase 변동도 남는다.
4. **단순 시간 선택보다 geometry 정보가 추가되는가?** 같은 시간/거리 marginal을 보존한 query permutation보다 energy가 높아 단순 pooled early-time prior만으로 모두 설명되지는 않는다. 다만 시각 후보와 실내 구조의 상관, broad reverberation을 배제하지 못해 specular mechanism은 미검증이다.
5. **후보별 reflection consistency 가능성?** Top1-vs-next energy 차이는 진단적 가능성이다. 정확한 reflection consistency로 쓰려면 backend timing/path 검증과 FP-wall registration이 먼저 해결돼야 한다. 이번 결과로 scorer를 개발하거나 다음 실험을 시작하지 않았다.

## 실패·반증·일반화 한계

통제36rings와 후속60rings 전부 보존했다. 단일 벽 성공 사례만 고르지 않았으며9 geometry×distance조건의 실패 수치를 모두 공개한다. Trimesh 미설치,PLY loader/degenerate OBJ 오류는 로컬 로그에 보존하고 GLB+원격 바닥으로 수정했다. 최초 direction subgroup에 floating modulo로360°가 별도 bin이 된 오류는 원본 JSON을 보존한 뒤0° bin으로 정정했다. 이는 primary contrast를 바꾸지 않는다. Random sensitivity 창이 최대0.15ms 분석 경계를 넘을 수 있던 구현도 수정해 전 대조를 재계산했고, 수정 전 결과를 보존했다. Non-random contrasts는 전부 동일함을 검증했다.

물리1차 검증이 실패했으므로 HSSD 단계는 '통제 환경에서 검증 완료된 검출기 적용'으로 부를 수 없다. 사용자 요청의 feasibility 비교는 모두 실행하되 **미검증 물리 시간창의 적용 실험**으로 격하한다. Raw6ch RIR는 현장 녹음이 아니라 SoundSpaces 합성이다. 재질·transmission을 비활성화했고 실제 음속/표면 scattering/path export를 직접 확인할 공개 binding이 없다. 반사 경로 GT 대조는 미수행이며 이를 위해 내부 path length/order/wall ID와 합성 전 energy histogram export가 필요하다. wall 제거/이동 개입으로 가능한 대체 검증을 수행했다.

## 종료 판단

Step0 감사,Step1 세 환경/개입·반복·timing falsification,Step2 HSSD 실제 visual 후보·6대조군,Step3 정량/실패/가능성 분석과 모든 필수 산출물이 완료됐다. 실패한 물리 가설을 성공으로 바꾸는 것이 종료 조건은 아니다. 불가능한 path-GT/backend 원인 규명은 위 조건과 함께 명시했다. 이 보고서 작성만으로 종료하지 않았으며 재현 검증·문서 검증 후 지정 branch에 Commit/Push한다. Docs Commit SHA는 commit 이후 최종 응답에만 기록한다.
'''
 (docs/'RESEARCH_REPORT.md').write_text(report)
 rows=['| 환경 | 앞벽 거리 | 제거 power ratio(dB) | repeat 차이(dB) | 이동 후 new/old(dB) |','|---|---:|---:|---:|---:|']
 for x in c['contrasts']:rows.append(f"| {x['kind']} | {x['distance']:.1f}m | {x['removal_db']:.3f} | {x['repeat_change_db']:.3f} | {x['move_new_vs_old_db']:.3f} |")
 diagrows=['| 조건 | 2m timing bias 평균±repeat SD(ms) | 6m timing bias 평균±repeat SD(ms) |','|---|---:|---:|']
 for label in ['baseline','fs24000','fs44100','fs96000','no_diffraction','depth1','rays200k','closed_box','one_sided','free_direct']:
  cells=[]
  for dist in [2.,6.]:
   v=[r['mean_error_ms'] for r in t['diagnostic'] if r['distance']==dist and r['label']==label];cells.append(f'{np.mean(v):.4f} ± {np.std(v):.4f}')
  diagrows.append('| '+label+' | '+' | '.join(cells)+' |')
 phys=f'''# Physical Verification — first-order arrival correspondence

## 原理と実験 / 물리 원리

유한 수직 벽선(a,b)의 단위 normal n에 대해 source s의 image는 s′=s−2n[n·(s−a)]다. 채널 receiver rᵢ에서 image로 이어지는 선이 벽 내부 qᵢ를 통과하고 양 leg가 가려지지 않아야 한다. 예상 길이 Lᵢ=‖s′−rᵢ‖,시간 tᵢ=Lᵢ/343. source와receiver가 벽의 같은 쪽이어야 하며 point가 segment 밖이면 제외한다. 마이크별 별도 qᵢ와시간을 사용한다. 임의 radial ray 거리2d를 모든 방향 반사로 지정한 proxy와 구분한다.

시각화: [interactive](interactive.html). 메인 분석: [report](RESEARCH_REPORT.md). 설정: [protocol](PROTOCOL.md).

## 실제 renderer 개입

Single-wall,8×8m room,2.4×18m corridor; front-wall거리0.3/2/6m; baseline/repeat/front-wall 제거/front-wall1m 이동의36rings. source(0.17,1.5,d),radius0.05m 6mono positions. 벽 y−8..12m,원격 바닥 y−20m를 모든 조건에 둬 navigation bounding box를 유효하게 했다. 해당 바닥 monostatic return은 약125ms 이후이므로1–80ms 해석에서 제외된다. 모든 mesh의 simulator-loaded bounds를 vertex bounds와1e−4m 이내로 검증했다. 두면 panel의 우려는 closed-box/one-sided 후속으로 점검했다.

경로 GT를 export하는 API는 사용 가능한 binding에 없다. 따라서 벽이 사라지거나 이동했을 때 response가 변하는지를 검증했다. Room/corridor 개입은 고차 경로도 바꾸므로 차분 전체를 '첫 반사 GT'라고 부르지 않는다. Single-wall은80ms 전에 다른 큰 reflecting surface가 없어 wall-dependent response의 가장 강한 대조다.

## 예상 창 내 모든 조건의 결과

아래 값은 **예상±0.5ms 창**의 absolute mean-squared amplitude 비율이다. 작은 baseline/removed noise floor끼리의 비율은100dB도 가능하므로 반사 검출 성공 지표가 아니다.

{chr(10).join(rows)}

단일 벽의 양의(base energy−removed energy) 차분 중 예상±0.5ms에 포함된 비율은0.3m에서3.29e−13,2m6.63e−16,6m4.29e−16이다. 거의0이며 작은 부동소수점 잔차를 반사로 부르지 않는다. 이 차분은 stochastic waveform 변동을 포함하므로 정확한 path-energy decomposition은 아니다. Repeat absolute sample-energy 차이는 removed effect의0.19/1.45/1.49배로,2/6m에서 세부 waveform shape 변동이 크다. 반복 peak timing은 훨씬 안정적이다.

각 채널 expected/observed times,window peak power,sharpness,coherence,timing residual과0.25/0.5/1ms 분석은 [controlled_analysis.json](controlled_analysis.json)에 보존했다. Mean peak를 임의 GT로 지정하지 않았다. 'Observed peak'는전체1–80ms에서 계산한 기술통계이며, 벽 귀속은 removal/movement와 함께만 해석한다.

## 추가60rings: 대안 설명을 가르는 검증

{chr(10).join(diagrows)}

각 조건은3fresh repeats. SD는3repeat의 기술통계이며 독립 scene CI가 아니다. free_direct는 source-array center4/12m의 벽 없는 직접 경로로,같은 길이의 반사와 비교한 clock audit이다. 직접 경로는c343예상보다약0.33/0.39ms 빠르고 반사는약2.35/6.15ms 늦다. 따라서 단일 전역 sample-rate scaling 오류만으로 두 현상을 설명할 수 없다. 초기 근거리 direct peak는sample7이며 전체 시간축을 query peak에 맞춰 shift하지 않았다.

24/44.1/48/96kHz,time in seconds,회절off,depth1,200k rays,closed-box,one-sided 모두 지연을 제거하지 못했다. 이 결과는 samplerate·diffraction·higher-order-only·duplicate-face-only 설명에 반대되는 증거다. 반사 backend가 어떤 변환을 적용하는지는 **미확정**이다. 실측 음속이나 backend path GT를 확보하지 않은 채 c를 fitting하거나 시간 보정 공식을 학습하지 않았다.

## 실패 조건과 결론

좁은 expected window의 최고점은 noise가 될 수 있고,coherence가 높아도 공통 low-level filter tail일 수 있다. 특히 near-wall direct/early overlap,room/corridor의 여러 경로 중첩,weak/distant response에서는 잘못된 wall attribution 위험이 크다. 이번 production renderer와 설정에서 물리 예상창 대응은 음성이다. 벽 의존적 음향 response의 존재와 HSSD 조건부 에너지 연관성은 이 부정 결론과 양립한다.

독립적인 실제 실내 녹음·재질 다양성·정밀 wall-path annotation·3D ceiling/floor reasoning으로 일반화하지 않는다. 해당 항목은 이 실험의 확보된 입력에 없으며 path export/실측 센서가 필요하다.
'''
 (docs/'PHYSICAL_VERIFICATION.md').write_text(phys.replace('## 原理と実験 / 물리 원리','## 물리 원리'))
 (docs/'METHOD_AT_A_GLANCE.md').write_text('''# Method at a glance — 기하가 예측한 시간과 실제 음향 증거를 분리한다

**FP의 벽까지 가까우면 돌아오는 경로가 짧아진다.** 단, 임의 방향의 왕복거리만으로 반사를 만들 수는 없다. 입사/반사 법칙을 만족하는 유한 벽점과 가시 경로가 있어야 한다.

```mermaid
flowchart LR
 V[기존 visual Top20 후보 XY/yaw] --> G[FP 유한 벽 segment]
 G --> P[image source와 채널별 specular point]
 P --> F[벽 내부·양 leg 가시성 확인]
 F --> T[예상 길이 / 343 → 시간창]
 A[건축 stage slice] --> S[FP 지점의 물리 표면 support 감사]
 S --> T
 R[원본6ch RIR · 공통 sample0] --> E[예상창 energy / sharpness / coherence / timing]
 T --> E
 E --> C[Random · shift · 다른 후보 · 분포보존 · full early 비교]
 C --> I[장면 단위 불확실성·실패 분석]
```

1. **Geometry hypothesis:** 각 후보의 위치·방향에서5cm ring을 배치한다. 실제 query pose로 창을 고르지 않는다. Architectural support는 FP 자체의 검증 보조 정보이며 배포 시 공짜 입력이라고 가정하지 않는다.
2. **Physical constraint:** 벽에 source를 대칭시킨 image와 receiver를 직선으로 연결하면 반사 경로 길이를 구할 수 있다. 벽 밖 specular point나 가려진 경로는 제외한다.
3. **Clock alignment:** RIR sample0을 유지한다. 마이크별 peak에 맞춰 창을 옮기지 않는다. 48kHz에서±0.5ms는각방향24samples다.
4. **Evidence:** 강한 peak가 있는지,창 내부에 sharp한 변화가 있는지,예상 delay 정렬 후 채널 waveform이 닮는지 측정한다. Noise에도 argmax가 있으므로 이것만으로 reflection detection을 선언하지 않는다.
5. **Discrimination:** 동일 시간·거리 marginal을 유지한 다른 query RIR을 대입한다. 이 대조를 이겨야 단순 '이른 시간은 원래 에너지가 큼'을 넘어선 연관성을 말할 수 있다. Full early는 길이가 길어 per-sample power로 비교한다.
6. **Causal verification:** 통제 벽을 제거·이동한다. 신호가 벽 때문에 생겼는지와 계산한 시각에 도착하는지는 별개 질문이다.

## 이번에 배운 물리적 구분

벽 의존적 return은 있지만 좁은 예상 시간 대응은 실패했다. HSSD의 matched-control energy 차이는 양수였지만 coherence는 개선되지 않았다. 따라서 **geometry-associated energy ≠ verified first-order reflection identity**다.

새 예측: 정확한 path-time export를 지원하는 renderer에서 same-wall causal arrival이 확인되면,offset을 임의로 맞추지 않아도 single-wall response는벽1m 이동에 약5.83ms 지연해야 한다. 반대로 broad temporal association만 존재하면 넓은 창의 energy 차이는 유지돼도 세밀한 channel timing·wall identity는 실패할 수 있다. 이는 향후 검증 가능한 예측이며 이번 작업에서 다음 실험을 시작한 것은 아니다.

[정량 결론](RESEARCH_REPORT.md) · [물리 검증](PHYSICAL_VERIFICATION.md) · [상호작용 보기](interactive.html)
''')
 (docs/'INPUT_AUDIT.md').write_text('''# Step0 — existing research and input audit

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
''')
 # Compact scientific figure, not a screenshot of the viewer.
 fig,ax=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
 for v,col in [('base','#1767b0'),('removed','#15956f'),('moved','#db5b45')]:
  raw=np.load(root/f'controlled/single_d2.0_{v}/rir.npy').astype(float);power=(raw[:,:3840]**2).mean(0);ax[0].plot(np.arange(384,1201)/48,power[384:1201],label=v,color=col,lw=1)
 ax[0].axvspan(11.66-.5,11.66+.5,color='#efb654',alpha=.5,label='predicted window');ax[0].set(xlim=(8,25),xlabel='Time (ms)',ylabel='Mean squared amplitude',title='Single wall, 2m: intervention response is late');ax[0].legend(fontsize=8)
 controls=['random','shifted','other_candidate','matched_marginal','full_early'];vals=[contrast(x)['difference'] for x in controls];y=np.arange(len(vals));means=[x['mean'] for x in vals];errs=np.array([[x['mean']-x['ci95'][0],x['ci95'][1]-x['mean']] for x in vals]).T;ax[1].errorbar(means,y,xerr=errs,fmt='o',color='#1767b0',capsize=4);ax[1].axvline(0,color='gray',lw=1);ax[1].set_yticks(y,controls);ax[1].set(xlabel='Aligned − control normalized power (95% scene CI)',title='HSSD: association, not verified wall identity');fig.savefig(docs/'evidence.png',dpi=180);plt.close(fig)
 for f in ['hssd_summary.json','controlled_analysis.json','timing_diagnosis.json','matched_subgroups.json','viewer_verification.json']:shutil.copyfile(root/f,docs/f)
 shutil.copyfile(root/'interactive.html',docs/'interactive.html')
 (docs/'README.md').write_text('# Experiment1 — Reflection observability\n\n완료·혼합/핵심물리가설음성. [연구보고서](RESEARCH_REPORT.md),[물리검증](PHYSICAL_VERIFICATION.md),[원리](METHOD_AT_A_GLANCE.md),[Interactive](interactive.html),[재현](REPRODUCIBILITY.md).\n\n![주요증거](evidence.png)\n')
 print('reports written',docs)
if __name__=='__main__':main()

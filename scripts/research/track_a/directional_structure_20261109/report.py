"""Render factual tables from saved scene-level aggregates; no training."""
import json
from pathlib import Path
P=Path(__file__).resolve().parent;a=json.loads((P/'analysis.json').read_text())
lines=['# Directional Acoustic Structural Evidence — Track A','',
'## 결론과 증거 수준','',
'6-channel RIR의 방향별 구조 단서는 순수 visual 후보 사이의 판별에 유용했다. 후보 좌표를 그대로 유지하고 visual score와 구조 일치 점수를 0.85/0.15로 결합하면 New12 joint가 50.20%→53.32%(+3.125%p, paired scene bootstrap 95% CI [2.018,4.297])로 개선됐다. 고정 방법의 Transfer11 결과도 49.08%→52.34%(+3.267%p, CI [2.202,4.332])였다. 그러나 기존 cosine fusion과 B2+GEOQ를 능가하지 못했다. 전체 공간에서 구조 점수로 pose를 다시 선택하면 정밀도가 떨어졌다. 따라서 기여는 **시각 후보의 위치를 유지한 모호성 해소**이며 기존 강한 시스템의 대체 성공이 아니다.','',
'증거 수준은 **Exploratory**다. New12의 3개 outer fold는 8개 fit scene/4개 held scene으로 분리되지만, 모든 장면은 과거 연구와 visual teacher에 노출되었다. Transfer11도 새 독립 Test가 아니다. 2개 seed를 scene 내 평균한 뒤 12개 scene(transfer는11개)을 20,000회 bootstrap했다. 결정론적 ridge의 seed 복제는 독립 증거가 아니다. 다중 비교 보정은 하지 않았다. Test 관측·라벨·결과는 사용하지 않았다.','',
'## 기존 Stage A 감사 및 물리적 조건','',
'[AUDIT](AUDIT.md)와 `prior_replay.json`은 기존 24개 probe checkpoint의 96개 held prediction을 오차0으로 재현한다. 기존 manifest 3,723개 항목을 검증했다. 예전 raw_ridge는 raw waveform이 아니라 E/C/N 552차원이었다. Frozen embedding은128차원, combined ridge는680차원, 작은 MLP는680→64→4이며60epoch였다. 기존 MLP와 probe를 다시 학습하지 않았다.','',
'실제 raw RIR을 query별로 추적해6채널·48kHz·수평반경5cm·중앙source·channel roll 및 metadata를 확인했다. 여러 yaw query는 동일 base RIR의 회전을 재사용하므로1536query가1536독립 rendering을 뜻하지 않는다. 공유 시간 원점을 보존한84차원 descriptor를 만들었다. Direct path 부근 peak는 별도 통계로 기록하며, peak를 특정 벽 반사로 간주하지 않는다. CNN token 순서를 방위각으로 해석하지 않았다.','',
'목표는 각 body 방향 ±20°의5개 FP ray 중앙값, 최대10m이다. Front/Left/Back/Right는 logical raster [0,27,18,9]다. XZ forward=(cos yaw,sin yaw), body-left는 world yaw−90°이며 up×forward로 확인한다. Metric 목표는 log1p 거리; near<3m, ordinal 경계2m/5m를 고정했다. 2D blocked footprint는 감독 신호이며 3D 반사면 정답이 아니다. Query target은 연속 pose/1cm raycast, candidate geometry는10cm DESDF/20cm pose grid/36yaw여서 불일치가 남는다.','',
'## 방향별 예측','',
'아래 MAE/median/P90은 meter이며 scene별 통계를 평균했다. 특히 median/P90은 pooled quantile이 아니다. 거리 출력은 metric 계산에0–10m로 clip한다. BA는 balanced accuracy이며 기존77.07% ordinary accuracy와 다르다.','']
def table(section,rows,columns):
 lines.extend(['### '+section,'','| Model / arm | '+' | '.join(label for key,label,scale in columns)+' |','|'+'---|'*(len(columns)+1)])
 for name,row in rows:
  vals=[]
  for key,label,scale in columns:
   v=row[key];v=v['mean'] if isinstance(v,dict) else v;vals.append(f'{v*scale:.3f}')
  lines.append('| '+name+' | '+' | '.join(vals)+' |')
 lines.append('')
cols=[('mae_m','MAE m',1),('median_m','Median m',1),('p90_m','P90 m',1),('balanced_accuracy','BA %',100),('auroc','AUROC',1),('rank_correlation','Rank rho',1),('pair_order_accuracy','Ordering %',100)]
table('Prediction baselines',[(k,v) for k,v in a['prediction'].items() if k.endswith('/matched') or k=='raw_multi/prior'],cols)
for name,row in a['direction_metrics'].items():
 table(name+' / directions',[(d,{k:v[i] for k,v in row.items()}) for i,d in enumerate(['Front','Left','Back','Right'])],[(k,label,scale) for k,label,scale in [('direction_mae_m','MAE m',1),('direction_median_m','Median m',1),('direction_p90_m','P90 m',1),('direction_balanced_accuracy','BA %',100),('direction_auroc','AUROC',1)]])
lines+=['![Direction GT vs prediction](figures/direction_gt_prediction.png)','![Openness confusion](figures/openness_confusion.png)','![Scene-wise prediction](figures/scene_prediction.png)','',
'## 반증 대조와 불확실성','',
'Raw multi와 같은84차원/동일 ridge/동일4출력의 canonical channel-sort 대조는 절대 channel identity를 제거한다. Permutation 불변성을 raw에서 검사했다. Multi−symmetric BA +8.300%p [5.646,10.849], ordering +20.773%p [17.855,23.553]이다. 이는 단순 room prior만으로 모든 결과를 설명하기 어렵게 하지만 physical DoA를 입증하지 않는다. Cyclic probe는6회 회전 증강과36방향 supervision을 사용하여4출력 probe와 감독량이 다르다.','']
table('Raw controls',[(k,v) for k,v in a['prediction'].items() if k.startswith(('raw_multi/','cyclic_ecn/'))],cols)
table('Feature isolation',list(a['feature_ablation'].items()),cols)
lines+=['E/N은 각각48차원, coherence early/full은 각각228차원이다. Early는50ms미만, full은2–128ms로 late-only 대조가 아니다. E의 약한 성능은 정규화/표준편차floor/regularizer에 의존할 수 있어 energy 정보 부재로 해석하지 않는다. Complex channel 관계의 유용성은 특정 반사 경로 식별과 다르다.','',
'불확실성은 fit scene의 inner-OOF residual로 학습했다. Held fixed90 interval의 방향별 coverage는91.28/91.47/90.95/91.93%, log halfwidth는0.6255/0.6488/0.6127/0.6158이다. Learned error scale은 calibrated posterior가 아니다. Fit uncertainty 출력은 regressor에 대해 in-sample이므로 GEOQ 결합 feature에서는 제외했다.','',
'Repeat diagnostic은 기존4개 acoustic-training scene의8개Observed pose와32개FP pose를 재사용한다. Observed의 반복간 평균 거리 차이는0.418m, near-label agreement82.81%; FP는0.386m/87.50%다. 실제 센서 반복성이나 새로운 scene 일반화 증거가 아니다.','',
'## 순수 Visual 후보와 B2 후보 판별','',
'Pure visual fullbank 점수만으로 NMS(1m/30°) Top20을 구성했다. B2 Top20과 별도 평가한다. 쌍 선택은 fit-only visual gap 하위1/3, 후보 간 거리≥1m, FP log-geometry 차이≥0.15이며 GT는 이후 good/bad label에만 사용했다. High ambiguity는 fit-only top1–top2 visual gap 하위1/3이다. Candidate 내 z-score fusion과 fullbank z-score fusion은 다른 실험이다.','']
cc=[('joint','Joint %',100),('fine','R@0.1m %',100),('pair_accuracy','Ambiguous pair %',100),('ranking_auroc','Ranking AUC',1),('GT_success_rank','Success rank',1),('high_ambiguity_joint','High ambiguity %',100),('repair','Repair %',100),('harm','Harm %',100)]
for kind in ['pure_visual','B2']:
 table(kind,[(k.split('/')[1],v) for k,v in a['discrimination'].items() if k.startswith(kind+'/')],cc)
lines+=['Pair tie는0.5, Top1 tie는 첫 후보로 처리한다. Query-only constant는 pose 정보가 없고 pureV의 첫 후보를 유지한다. Success rank는 성공 후보가 있는 query에 조건부다. Repair/harm은 동일 후보 집합의 visual 대비 전체query 비율이다.','',
'![Ambiguous pair repair](figures/ambiguous_pair_repair.png)','![Ambiguous pair harm](figures/ambiguous_pair_harm.png)','',
'사례는 고정 GT-free pair 조건을 만족하는 첫 repair/harm을 사후 선택한 설명 그림이다. 빈도 근거는 전체 집계 표이며 harm도 제시한다.','',
'## Full SE(2) structural matching と統合','',
'Structural scoreはquery predicted log geometryと候補FP geometryの平均絶対残差の負値。推論にFP-RIR reference bankを必要としないが、frozen query encoderの過去contrastive学習にはFP-RIRが使われている。このbaselineをTrack B独立frameworkの達成と扱わない。Ordinal、direction-centered similarity、uncertainty weighting、geometry prior、donor、direction shuffleも評価した。GT-neighborhood score massはtemperature1の記述統計で、確率校正されていない。','']
table('Full bank',list(a['fullbank'].items()),[('joint','Joint %',100),('fine','Fine %',100),('R1','R@1m %',100),('top5','Top5 %',100),('top20','Top20 %',100),('gt_neighborhood_score_mass','GT score mass',1)])
table('Full bank / high visual ambiguity',[(k.split('/')[1],v) for k,v in a['fullbank_ambiguity'].items() if k.startswith('high/')],[('joint','Joint %',100),('top5','Top5 %',100),('top20','Top20 %',100)])
table('Full bank / preservation',list(a['fullbank_downstream'].items()),[('conditional_joint','Conditional joint %',100),('repair','Repair %',100),('harm','Harm %',100),('visual_success_preservation','Visual preservation %',100)])
table('Matched-budget GEOQ integration',list(a['integration'].items()),[('joint','Joint %',100),('fine','Fine %',100),('pair_accuracy','Pair %',100)])
lines+=['E0는 기존GEOQ, E1은 V+structure, E2는 cosine+structure, E3는 기존156feature에 구조8+상대8을 추가한172feature GEOQ 재학습이다. E4는 direction-centered matching 대안이다. 별도로 donor/nuisance 동일용량 대조와 고정0.05score addition도 수행했다. 150iteration/15leaf/최소leaf100의 동일 설정이다. Matched 결합은 GEOQ 대비 joint−1.432%p CI[−2.962,−0.130]였다. 고정addition은 joint가 같고 미세한 fine 차이를 확정 개선으로 보지 않는다. 기존GEOQ를 유지한다.','',
'추론에 사용할 수 없는 true4방향 geometry oracle도 fullbank joint51.11%, Top20 98.96%였다. 4방향 요약은 유일한 pose 식별에 불충분하며 예측 오차도 큰 추가 요인이다. Oracle은 진단용이고 추론에 넣지 않는다. Fullbank V+structure는 전체query에서 joint−2.409%p/fine−5.273%p였지만 후보 위치를 고정한 재정렬은 개선됐다. 구조 신호는 거친 mode 판별을 지원하지만 연속 pose peak를 움직이기에는 불충분하다는 해석과 일치한다.','',
'![Fullbank fields](figures/structural_score_field.png)','![Localization comparison](figures/localization_ablation.png)','',
'## 固定方法のTransfer11','']
table('Transfer',list(a['transfer'].items()),[('joint','Joint %',100),('fine','Fine %',100),('pair_accuracy','Pair %',100),('ranking_auroc','Ranking AUC',1),('repair','Repair %',100),('harm','Harm %',100)])
table('Transfer existing baselines',list(a['transfer_baselines'].items()),[('joint','Joint %',100),('fine','Fine %',100)])
lines+=['계수·fit scene·checkpoint hash·ambiguity threshold를 transfer score 계산 전에 고정했다. 11scene 모두 V+structure−V joint 차이는 양수였지만 기존cosine fusion/GEOQ를 넘지 못했다. 새로운 미사용scene 독립검증은 미실시다.','',
'## 비용·재현성과 한계','',
'아래 시간은 측정 구간이 달라 speedup 비율을 주장하지 않는다. Structure는 CPU4방식 합계, cosine은 GPU query×reference 행렬곱, visual은 CPU score 계산이다. Encoder 전처리·RIR 생성 비용은 제외한다. Full wall에는 NMS/IO가 포함된다. FP embedding과4방향geometry byte는 dtype을 포함한 표현 크기이며 raw reference RIR 총량이 아니다.','']
lines+=['| Scene | Poses | Struct4 CPU s | Cosine GPU s | Visual CPU s | FP embedding bytes | Geometry bytes |','|---|---:|---:|---:|---:|---:|---:|']
for r in a['costs']:lines.append(f"| {r['scene']} | {r['bank_poses']} | {r['structural_score_seconds']:.3f} | {r['cosine_matmul_seconds']:.3f} | {r['visual_score_seconds']:.3f} | {r['fp_embedding_bytes']} | {r['geometry4_log_bytes']} |")
lines+=['','## 가설 판정과 다음 반증 예측','',
'- H1: 거친 방향별 FP구조 관계를 지지한다. 정밀DoA/벽별 반사 식별은 미증명이다.',
'- H2: scene prior만이라는 설명은 donor/channel/동일용량 비방향 대조로 약해진다. Dataset 고유 shortcut 전반을 배제할 수는 없다.',
'- H3: 고정pureV 후보 재정렬에 한정해 지지한다. Fullbank metric localization 개선으로 일반화하지 않는다.',
'- H4: cosine/GEOQ를 넘는 추가 정보의 실용 가치는 미확인이다. Matched 재학습은 부정 결과다.',
'- H5: metric 불확실성·먼 거리 과소추정·fine 저하는 지지 증거다. Ordinal로만 대체하면 된다고 말할 수 없다.',
'- H6: 현 조건에서 방향 정보를 전혀 관측할 수 없다는 강한 주장은 대조 결과와 모순된다. 물리적 방향 분해능·실측 강건성은 미해결이다.','',
'전이 가설: 거친 acoustic 구조는 시각이 제안한 떨어진 mode 사이 선택에 유용하고, mode 내부 정밀 위치를 움직이면 harm이 증가한다. 새로운 미사용scene/실측RIR에서 visual 좌표 고정과 자유 재최적화를 사전등록하여 비교하면 반증 가능하다. 이번 결과를 미실시 실험의 결론으로 대체하지 않는다.','',
'새 checkpoint73개를 생성했다(prediction24, uncertainty6, integration24, symmetric6, feature12, transfer1). 가중치·waveform·raw 예측은 로컬에 보존한다. 기존 연구·strong baseline·Track B는 변경하지 않는다. 실행 코드·hash와 재현 절차는 최종provenance/재현 문서로 추적한다. 최종 완료는 model replay·산출물manifest·요건감사·Docs/index 검증과 동기화 후 판정한다.']
# Tables are generated directly from saved aggregates.
table('Ordinal and ordinary openness accuracy',list(a['ordinal_openness'].items()),[('ordinal_accuracy','Ordinal %',100),('openness_accuracy','Ordinary openness %',100)])
table('Candidate conditional metrics and margins',list(a['discrimination_extended'].items()),[('R1','R@1m %',100),('conditional_joint','Conditional joint %',100),('visual_success_preservation','Preservation %',100),('success_minus_failure_margin','Success-failure margin',1)])
lines += ['','Signal audit: 1,536 query rows reuse 1,201 base RIRs. All 9,216 channel early peaks occur at sample 7. This is centered-source direct timing, not directional wall evidence.']
sg=json.loads((P/'signal_audit.json').read_text()); ca=json.loads((P/'candidate_preparation.json').read_text())
lines += ['','## 측정한 전처리 구간','',f"Raw descriptor/ECN audit wall: {sg['seconds']:.3f}s (1536query). PureV candidate preparation: {sum(r['seconds'] for r in ca['records']):.3f}s (12scene; cosine controls, IO, NMS included). Renderer generation and complete encoder/GEOQ end-to-end latency were not measured; no speedup claim."]
(P/'RESEARCH_REPORT.md').write_text('\n'.join(lines)+'\n')
print('Report tables written',len(lines))

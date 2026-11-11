"""Archive compact evidence and write complete Experiment4A reports."""
import argparse,json,hashlib,subprocess,shutil
from pathlib import Path
import numpy as np

def table(headers,rows):return '| '+' | '.join(headers)+' |\n|'+'|'.join(['---']*len(headers))+'|\n'+''.join('| '+' | '.join(str(x) for x in row)+' |\n' for row in rows)
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--docs',type=Path,required=True);a=p.parse_args();r,d=a.root,a.docs;d.mkdir(exist_ok=True,parents=True)
 load=lambda name:json.loads((r/name).read_text());R=load('results.json');H=load('hssd_results.json');N=load('noise_results.json');V=load('verification.json');G=load('diagnostics.json');M=load('controlled/manifest.json');HM=load('hssd_render/manifest.json');sel=R['representative'];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 def write(name,s):(d/name).write_text(s)
 for name in ['verification.json','diagnostics.json','explorer_manifest.json','pilot_cost.json']:(d/name).write_bytes((r/name).read_bytes())
 small={k:v for k,v in R.items() if k not in ['predictions','fit_score_references']};write('quantitative_summary.json',json.dumps(small,indent=2));write('noise_summary.json',json.dumps(N['summary'],indent=2));write('hssd_summary.json',json.dumps({k:v for k,v in H.items() if k!='records'},indent=2))
 code=Path(__file__).resolve().parent;revs={k:subprocess.check_output(['git','rev-parse',k],text=True).strip() for k in ['aff5991','50b51b3','47d6740','97b5d71','e54e706','239de7d','9fdbe0e','8587ca3']}
 compact=[dict(name=v['name'],rir_sha256=v['rir_sha256'],channel_bytes_sha256=v['channel_bytes_sha256'],material_sha256=v['material_sha256'],mesh_sha256=v.get('mesh_sha256'),export_sha256=v['export_sha256'],bytes=v['bytes']) for v in M+HM]
 provenance=dict(study='material_observability_20261111',raw_root=str(r),code_commits=revs,source_hashes={p.name:sha(p) for p in code.iterdir() if p.is_file()},original_analysis_execution=load('analysis_execution.json'),final_selection_code=revs['e54e706'],raw_inputs=compact,hssd_input_audit=load('hssd_inputs/manifest.json'),renderer_image='sha256:c2907a52a8d425fc853beaa8c2fb202b577538acaee7f512b9cee9a5b8d26fc1',library_sha256='31e948eef4908d8cbb403b5f445d9d0eab59fc81b05a658538f8795984f9bfb4',artifact_hashes={name:sha(r/name) for name in ['features.json','results.json','noise_results.json','hssd_results.json','diagnostics.json','verification.json']},cost=dict(controlled_rings=len(M),controlled_bytes=sum(v['bytes'] for v in M),controlled_wall_sum_s=sum(v['wall_seconds'] for v in M),hssd_rings=len(HM),hssd_bytes=sum(v['bytes'] for v in HM),hssd_wall_sum_s=sum(v['wall_seconds'] for v in HM)),dependencies='Prior renderer image, local native HSSD assets, prior pure-visual candidate cache; no neural checkpoint loaded')
 # Do not copy detailed candidate path arrays into provenance.
 for scene in provenance['hssd_input_audit']:
  for q in scene['queries']:q.pop('hypotheses',None)
 write('reproducibility_manifest.json',json.dumps(provenance,indent=2))
 effects=table(['Pose','흡수 .05→.70 Δenergy dB','산란0→1 Δenergy dB','interaction dB','High−Low tilt dB'],[[v['pose'],f"{10*v['log_energy']['absorption_70_minus_05']:+.3f}",f"{10*v['log_energy']['scattering_1_minus_0']:+.3f}",f"{10*v['log_energy']['interaction']:+.3f}",f"{v['profile_high_minus_low_db']:+.3f}"] for v in R['effects']])
 reliability=table(['Rule','wall-state TPR','제거벽 FPR','FNR','AUC','Brier'],[[v['rule'],f"{v['tpr']:.3%}",f"{v['fpr']:.3%}",f"{v['fnr']:.3%}",f"{v['auc']:.4f}",f"{v['brier']:.4f}"] for v in R['reliability'] if v['group']=='evaluation'])
 classifier=table(['분리 조건','Full energy','Band energy','Normalized shape','Profile shuffle'],[[split]+[f"{next(v['accuracy'] for v in R['profile_classification'] if v['split']==split and v['arm']==arm):.1%}" for arm in ['energy','bands','shape','shuffle']] for split in ['context_holdout','hold_single','hold_room','hold_corridor']])
 hssdtable=table(['Scene/query','α .70/.05 energy ratio','s .5/0 energy ratio','6조건 최고 evidence 후보 rank'],[[f"{v['scene']}/{v['query_id']}",f"{v['absorption70_over05']:.3f}",f"{v['scattering05_over0']:.3f}",str(v['top_candidate_by_condition'])] for v in H['contrasts']])
 noisec=table(['Noise','Peak FPR','Combined FPR','Combined AUC'],[[level]+[f"{next(v[key] for v in N['summary'] if v['noise']==level and v['rule']==rule):.3f}" for rule,key in [('peak','fpr'),('combined','fpr'),('combined','auc')]] for level in ['clean','moderate','severe']])
 write('RESEARCH_REPORT.md','''# Experiment4A — Material-dependent Acoustic Observability

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

'''+effects+'''
Energy는1–80ms의 channel 평균 power sum이다. dB는 mean log10 energy 차이×10으로, 각pose에서 균형 잡힌 조건 평균을 비교했다. Absorption effect는scattering3수준, scattering effect는필수absorption3수준(.99stress 제외), interaction은s=1과s=0에서의 absorption 효과 차이다. Profile tilt는a=.3/s=.5 high−low. 모든 비교는3fresh contexts를 평균하되 독립 장면 수로 세지 않는다.

흡수 효과 범위−5.03..−2.07dB, spectral tilt−6.73..−2.88dB로 방향은 일관됐다. 산란 energy 효과는room_far−0.126dB에서single2+3.052dB까지 달랐다. 산란에 따른 peak-error 변화도 single2/6에서는+3.64/+8.73ms, corridor에서는−1.39/−3.81ms로 반대다. 이 값은 dominant peak의 교체/혼합을 포함하며 특정 경로의 travel-time 변화가 아니다.

단일벽 정반사에서 α .70/.05 energy ratio는2m0.347/6m0.309였다. Energy fraction 관계(1−.70)/(1−.05)=0.316에 가깝고, pressure fraction을 그대로 제곱한0.100과 다르다. 이것은 제한된 관측 response 대조이며 전체 입사/반사 flux 보존을 측정한 것은 아니다. Reflection amplitude를 별도 slider로 독립 조절하지 않는다.

거리별 decay, direct/indirect suppression, channel 상대energy, band late/early ratio는 [추가 진단](diagnostics.json)에 있다. Channel timing/phase는 순차 mono 렌더의 일관성 진단이지 동시 수음 phase GT가 아니다. Repeat SD와 wall-movement error72교차 조건, aligned/shifted/matched-time window 결과는 [전체 집계](quantitative_summary.json)에 보존했다. Matched-time donor는 서로 다른2/6m군을 교환해 시간 분포를 유지하고 정렬은 깨뜨렸다.

## B — Frequency-dependent reflection

각profile은 실제 backend4center에서 평균 coefficient α를 같게 맞췄다. Low는낮은2band에α+δ,높은2band에α−δ; High는반대, δ=.8min(α,1−α). Incident-spectrum-weighted absorbed energy는 같다고 보장하지 않는다. Native interpolation kernel은 공개되지 않아 band center에 직접 anchor했다.

'''+classifier+'''
알려진3개의 **시뮬레이션 profile**에 대한 nearest-centroid 진단이다. Context holdout은동일pose/geometry의2contexts fit→1context 평가(18건)라 독립 scene test가 아니다. Leave-family-out도 설계 기하3종뿐이며 각평가18건이다. Full energy만으로는33.3%, normalized shape는context holdout72.2%였으나 held-room44.4%로 떨어졌다. Shuffle baseline도 소표본에서변동이 크므로 실제 재질 식별이나 강한 일반화를 주장하지 않는다.

Normalized shape 거리의 중앙값은 동일geometry·다른profile0.0465, 다른geometry·동일profile0.0291이며 범위가 겹친다. Spectrum 차이와 geometry 차이는 분리되지 않는다. Band별late/early ratio도 profile별로 기록했다. 이는 유한 window의 decay proxy이지 band RT60 GT가 아니다.

## C — Observability / Reliability

'''+reliability+'''
위 TPR/FPR의 GT는 **target-wall intervention 상태**다. 실제 first-order path GT가 없어 ‘진짜 특정 반사 경로 검출률’은 미검증이다. Room/corridor 제거조건의 다른 반사는 실제로 존재하므로 FPR은 소리 자체가 가짜라는 뜻이 아니라 목표 벽 귀속의 실패다. Pooled1836을 독립 표본으로 세지 않으며 pose/material/profile별 표가JSON에 있다.

Combined는5scores의 fit empirical percentile 평균이다. Single2 uniform에서FPR≤5%를 만족하며 FNR을 최소화하는 threshold를 고정했다. 다른조건에서 peak rule 대비 FPR은 줄지 않았고 Brier는0.251→0.373으로 나빠졌다. H3의 ‘결합하면 reliability가 더 좋다’는 강한 형태는 **이번 규칙·fit 범위에서 지지되지 않았다**. 평가를 본 뒤 threshold를 재선택하지 않았다.

Known α/scattering으로 threshold를 나눈 **Oracle**도TPR82.1%,FPR70.6%다. 이것을 material-free inference와 섞지 않았다. Unknown material의 단순 RIR rule과 oracle 모두 대상 벽 귀속을 해결하지 못한다.

필수 .05/.30/.70 clean present 조건에서 raw gate 미검출은 없었다. 약한 신호 반례는 사전 지정 .99stress의 **single6,wall moved(실제거리7m),s=0,high-profile**에서나왔다:3contexts 중2개에서6채널 모두 raw gate 미검출, 나머지조건의 peak-rule 통과와도 구분한다. 이 극단조건을 일반 재질의 실패율로 외삽하지 않는다. Viewer에는context0 실제 실패를 포함했다. 미검출을 벽 부재로 읽지 않고 Unknown/Unobservable로 남긴다.

### Noise 보조 대조

'''+noisec+'''
108개 동일 clean recording에 합성 Gaussian noise를 추가한 별도 보조 분석이다. Moderate/severe는pose별고정 baseline early RMS 대비20/0dB이며 material별실효SNR을 따로 기록했다. 실제 renderer output처럼 표기하지 않았고 explorer의 waveform에는 noise를 섞지 않았다. Clean에서도다중반사 FPR이높고 noise는 이를더높인다. Spectral tilt 변화분포도 별도기록했다. Material과noise를 같은기전으로 단정하거나 어느쪽이실환경에서더크다고 주장하지 않는다.

## D — HSSD transfer feasibility

Train2scenes×2poses×6coefficients×3contexts=**72native-stage rings**를 실제 생성했다. Native GLB source좌표를 보존하고 |normal_y|<.25인vertical face proxy에만계수를지정했다. Stage/object assets는분리되지만 이proxy는검증된architectural-wall segmentation이아니다. Furniture/object instances는이번쌍에서제외했다. Stock semantic-material mapping은이setup에서검증되어있지않으며 furnished전체로확장성공을주장하지않는다.

'''+hssdtable+'''
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
''')
 write('PHYSICAL_MATERIAL_AUDIT.md','''# Physical material audit

## Verified contracts

Installed SoundSpaces/RLR image and library hashes are in [manifest](reproducibility_manifest.json). Public header `RLRAudioPropagation.h` and installed Habitat `AudioSensor.cpp` were inspected. `enableMaterials=false` chooses joined nonsemantic mesh/default acoustic material; it does not disable absorption or scattering. Default coefficients areα=.1,s=.5,T=0. `setAudioMaterialsJSON` is consumed by the semantic-loading branch; simply passing a JSON while keeping the stock nonsemantic path does not establish HSSD assignment.

This study directly uses the same installed RLR C API and supplies mesh indices grouped under `audit_surface` / `fixed_background`. Header documentation maps category strings through `labels` substring matches; human-readable `name` is ignored. Every controlled acoustic OBJ export was checked against the provided coefficient values. A removed single-wall case has no varying-surface face, so only background material should appear; that exception is verified explicitly.

4log-spaced bands are reported by actual `RLRA_WriteIRMetrics`:83.910034,369.251007,1624.911255,7150.516602Hz. JSON coefficients use alternating `[frequency,value,...]`. The exact interior interpolation kernel is not documented in the public header; engine source is not supplied. Profiles use those reported centers to avoid relying on assumed linear/log interpolation between widely different anchors. Native binary's internal pre-pressure histogram/path ID remains unavailable; coefficient export is assignment evidence, not acoustic path GT.

## Coefficients and energy accounting

All variable surfaces:α in.05/.30/.70/.99,scattering0/.5/1,transmission0. Uniform hasα atall4centers;low/high useα±δ,δ=.8min(α,1−α),two low/two high bands with opposite signs. Mean atcenters equalsα for everyprofile. Fixed background retains bundled default damping/density/material-medium speed andα.1,s.5,T0. No ambient speed adjustment or per-condition delay is fitted.

Public header comments call the coefficient a pressure fraction, while standard absorption energy accounting and our single-wall response favor an energy-fraction interpretation:α.70/.05 yieldsenergy ratios.347/.309 vs.316energy prediction and.100pressure-squared prediction. This is **consistent**, not full implementation or global-flux proof. AtT0,reflected andabsorbed energy share the incidentbudget, so reflection cannot be an independent UI variable. Scattering redistributes the surviving reflection; higher measuredpoint energy can occur without higher totalreflected energy. The material-medium speed field(~1484m/s) belongs to material transmission and is not air c343.

## Source / receiver / geometry

Source height1.5m;6mic0.05m ringθ=π+iπ/3;source atarraycenter. Renderer listener radius0.1m is a separate receiver-capture setting and is fixed.48kHz,unitScale1,globalVolume1,default maxIR4s;20kindirect/512source rays,depth50,2threads,diffractionon,transmissionoff,temporal coherenceoff. Sequential mono channels are not guaranteed phase-synchronous.3fresh contexts percondition are not independent scenes or guaranteed seeds.

Single-wall panelx±12,y−8..12,z0 plusdistant floor. Room8×8×3 andcorridor2.4×18×3 havefixed floor/ceiling. Front panel isremoved ortranslatedz−1 withoutchangingotherpanels; gaps appear atcorners in moved cases. Allvariableverticalsurfaces sharecoefficient, so deletingfrontwallstillleavesmaterial-dependentotherreflectionsinroom/corridor. No detected peak is designated as wall GT. Independent scalar Fermat minimization and108raw energy/onset checks verifyexpected target-panel geometry and arithmetic.

## HSSD native mapping

TwoTrain native stageGLBs areusedat4previouslyrecordedposes. Original GLB and FP hashes verifiedunchanged. Native stageface proxy|normal_y|<.25 receivesvariablematerial;otherfacesstaydefault. Scene102344250:4677faces/2438verticalproxy;102816615:3263/1683. Theproxyincludesverticalarchitecturalgeometry butisnotexactsemanticwall identity. HSSDobjectinstancesareseparatefiles andtheircountsarearchived;furniture/openingobjectsarenotrenderedinthisstage-onlyaudit. RGB/FPassetsarenotmodifiedorregenerated.

A limited surface-specific nativeinterventionistherefore**Verified**, stock furnishedHSSDsemantic-materialmappingis**Unverified**. Extendingitrequiresvalidatedtriangle/categoryassignment,opening/furnituretransformpreservation,andcross-checkagainsttheoriginaljoinedacousticmesh. ControlledgeometryresultsarenotrelabeledasHSSDresults.

## Physical mechanism versus renderer effect

- **Verified:** parameterassignment,stableabsorption/profilechanges,geometry-heldmaterialinterventions,andwall-stateambiguityunderretainedotherreflectors.
- **Consistent:** fixedfloor/ceilinghelps explain smallerabsorption effect andpersistentearlyreturnsinroom/corridor;energy responseisclosertoanenergybudgetthanpressure-squaredmodel.
- **Unresolved:** exactearlypath identity,internalintensity-to-pressurefilter/grouping,physicaltransferofdominantpeaktiming,scene-generalreliability. Experiment1B'srenderer-dependenttimingfindingremainsvalid;thisstudydoesnotrepairit.

Primary implementation references: [RLR distribution](https://github.com/facebookresearch/rlr-audio-propagation), [SoundSpaces2](https://github.com/facebookresearch/sound-spaces/blob/main/SoundSpaces2.md). The authoritative evidence here is the installed-version source/header, actualexports andhashes, not an assumption about latest source.
''')
 write('METHOD_AT_A_GLANCE.md','''# Acoustic material과 reflection evidence

같은벽이라도반사되어돌아오는소리의크기·모양·스펙트럼은다를수있다. **Geometry는가능한경로와시간을제안하고,재질은그경로가얼마나관측되는지를바꾼다.**

1. **Geometry를고정한다.** Source/6mic/벽을움직이지않은채absorption과scattering을교차해바꾼다. T=0이므로absorption으로사라진에너지와reflection의관계가연결된다.
2. **흡수와산란을구분한다.** 흡수는돌아오는에너지를줄인다. 산란은반사에너지의방향·시간분포를바꾼다. 측정위치에서커진peak가총반사에너지증가를뜻하지않는다.
3. **주파수별모양을분리한다.** 평균계수를맞춘Low/High absorption profile을비교한다. Full energy와normalized spectrum을함께보아전체감쇠와형태차이를구분한다.
4. **벽에개입한다.** 제거·이동·반복을통해현재증거가targetwall을따르는지본다. 방의다른벽·바닥반사가남으면소리가있다는사실만으로targetwall존재를판단할수없다.
5. **관측의한계를검사한다.** Peak·shape·time·band·channel rule을fit조건에서고정하고다른조건에서평가한다. 이번결합rule은제거벽오검출70.6%여서검증된벽검출기가되지못했다. 미검출은Unknown이지벽부재가아니다.

[Explorer](interactive.html) 기본은전체효과의중앙에가까운corridor_near다. Room_far로바꾸면fixedbackground반사의영향이더큰예외를볼수있다. Single6에서A를α=.99,s=0,High,Wall moved로설정하면실제벽이있어도raw gate가반사를검출하지못하는사례가나온다. A/B를같은조건으로두면파형이일치하며,각조건은같은context0저장결과를가리킨다.

기본waveform은공통amplitude/time origin이다. 정규화checkbox는형태비교용이며절대energy를없애므로감쇠차이를읽을때끄는것이맞다. Spectrum은early RIR의스펙트럼이지검증된한반사경로의스펙트럼이아니다. EDC와조건부T20추정도실측RT60이나무한한late tail을뜻하지않는다.

배울수있는예측은‘FP가같아도재질이달라지면후보별energy evidence순위가바뀔수있다’는것이다. HSSDnative-stage4poses중3곳에서실제로최고evidence후보가바뀌었지만정답을더잘맞췄다는뜻은아니다. Localization유용성은이번에검증하지않았다.
''')
 cost=provenance['cost']
 write('REPRODUCIBILITY.md',f'''# Reproducibility

Actual code, raw/input hashes and costs: [manifest](reproducibility_manifest.json). Raw root: `{r}`. Code: `scripts/research/track_a/material_observability_20261111/`. No model/checkpoint loaded or trained. Code/Docs branch `refactor/hssd-two-branch-fusion`.

## Actual execution revisions

| Stage | Code SHA |
|---|---|
| C++ renderer / frozen design | `{revs['aff5991']}` |
| Raw packing runner | `{revs['50b51b3']}` |
| Main coefficient profile corrected after pilot | `{revs['47d6740']}` |
| Initial feature/reliability/noise/HSSD analysis | `{revs['97b5d71']}` |
| Final representative exception correction | `{revs['e54e706']}` |
| Independent input/coefficient/scientific verification | `{revs['239de7d']}` |
| Energy/shape/distance diagnostics | `{revs['9fdbe0e']}` |
| Final explorer template geometry | `{revs['8587ca3']}` |

The compiled C++ binary is unchanged after initialrender; exactSHA256isrecorded locally. Existing2dirtyGibsonscriptswere notused/committed. FinalDocsSHAisreportedaftercommit,notinsertedintoitsownindex. Finalselectionrerunonlycorrected exceptionselection toinclude movedwalls (stillphysicallypresent),notrules/thresholds/metrics. Prior`results_selection_v1.json`isretained.

## Configuration / provenance

Image `sha256:c2907a52a8d425fc853beaa8c2fb202b577538acaee7f512b9cee9a5b8d26fc1`; installed RLR librarySHA256 `31e948eef4908d8cbb403b5f445d9d0eab59fc81b05a658538f8795984f9bfb4`. Habitat source`4f61e321477708fa606fbd8f42b4bef41d67c672`,RLRpackage`4fd446b4abb5c71fb7a232a083bbddd65f25fc6f`. ContainerPython3.9/C++14;hostPython3.10.12,NumPy1.21.5,SciPy1.8.0. HSSDpreparationusespriorstudy-localtrimesh3.23.5. No globaldependencychanges.

EachrecordhasmaterialJSON,meshtext,acousticOBJ,rendererreceipt,fullfloat32`rir.npz`with6originalchannel lengths,per-channelbytehashes,rawcontainerhash,andruntime. Compressionislossless;onlyownintermediate`.f32`filesaredeletedafterarray-equalityverification. Originalpriorstudies/assetsareunchanged. RNGcontextisnotaseed;threefreshcontextsaretheunitofrenderrepeat,notindependentphysicalscenes.

Newcontrolledraw:1980rings /11,880mono /{cost['controlled_bytes']:,}compressedbytes;sumper-ringwall{cost['controlled_wall_sum_s']:.3f}s. HSSD72rings /432mono /{cost['hssd_bytes']:,}bytes;sumwall{cost['hssd_wall_sum_s']:.3f}s. Pilot6rings3,445,204bytes/2.522sforecast~1.14GB/832s;pilotexcludedfrommainstatisticsbecauseprovisionalfrequencyanchorswerecorrected. CPUquota4/20percontainer;main+HSSDoverlapmaximum8/20(40%,below80%),2renderthreads each,8GiB memorylimit;noGPU/Node2. AnalysisBLAS2threads. CostsumsexcludeDocker/compile/startup/manifestrewrite/browsingandarenotfullscene-furnishedthroughput.

HistoricalHTML-reports-2referencecheckout(commit`b79e31c6bf1e09ad5d5785f76647a36ed83cae1f`)islocalreferenceonly;its~1.5GBisnotRIRcostandnotcommitted. Itssemantic-compatibilityviewerprovidedlightpanels,pairedviews,scenecontrolsandprovenancepatterns;nohistoricalTestpredictionswereusedforfit.

## Commands

RunfromAV-FPLocroot. Freshrenderdirectoriesmustnotexist;runnerrefusesthem. SourceassetsandDockerimagearelocaldependencies,soGitclonealoneisnotself-contained.

```bash
ROOT=/home/jeongeonlee/track1/experiments/material_observability_20261111
CODE=scripts/research/track_a/material_observability_20261111
OPENBLAS_NUM_THREADS=2 python3 "$CODE/analyze.py" --root "$ROOT"
OPENBLAS_NUM_THREADS=2 python3 "$CODE/results.py" --root "$ROOT"
OPENBLAS_NUM_THREADS=2 python3 "$CODE/auxiliary.py" --root "$ROOT"
python3 "$CODE/diagnostics.py" --root "$ROOT"
python3 "$CODE/verify.py" --root "$ROOT"
python3 "$CODE/build_explorer.py" --root "$ROOT" --out /tmp/material-explorer.html
```

TheseoverwriteonlyderivedJSON;copythestudydirectoryfirstifpreservingallhistoricalderivedsnapshots. Rawinputsremainread-only. To rendernewconditions,compile`render.cpp`againsttheinstalled`RLRAudioPropagation.h`and`libRLRAudioPropagation.so`,thenrun`run.py --out NEW/controlled --exe NEW/render`intheimmutableimagewithrepo`/code:ro`,study`/output`,networknone,`--cpus4 --memory8g --user uid:gid`. Compiler flags: `-O2 -std=c++14 -Wno-deprecated-declarations`, include `/opt/src/habitat-sim/src/deps/rlr-audio-propagation/RLRAudioPropagationPkg/headers`,libraryand rpath `/opt/conda/envs/soundspaces/lib/python3.9/site-packages/habitat_sim/_ext`,link`-lRLRAudioPropagation`. `run.py --pilot`reproducespilotstructurebutcurrentanchorsarecorrected;historicpilotanchorsareat50b51b3.

HSSD: `prepare_hssd.py --out NEW/hssd_inputs`,thencontainer`run_hssd.py --root NEW --exe NEW/render`. PreparationcurrentlyreferencesdeclaredlocalHSSDandpriorcandidatepaths;adjustpathsandcheckhasheselsewhere. Originaluserquerymetadataisusedforsourceposeonly;candidatewindowselectionusescachedpure-visualgeometryandnoGTfilter. ExistingRGB/FP/nativeassetfilesarenotwritten.

## Failures / corrections / limitations preserved

- Pilotusedprovisional125/500/2000/8000Hzanchors. Actualmetricexportrevealed83.91/369.25/1624.91/7150.52;maincorrectedbeforeoutcomesandpilotretainedseparately.
- Firstrepresentativeselectioncheckednondetectiononlyin`present`,excludingmovedbutstillpresentwalls. Finalselectionadds`single6`fromactualmoved-wallfailure;defaultmedoidandallscientificscoresunchanged. BothselectionJSONsretained.
- InitialHTMLwas~30MBbecauseitembeddedfloatdisplayarrays. Finaldisplayuses480early/160fullbinsand16-bitquantizedextrema,160EDCsamplesat.01dB,128float32spectralbins;waveerror≤peak/65534,metricscomputedfromoriginalfloat32. Thisisscreenrepresentation,notRIRinterpolation/synthesis. Originalfullrawisnotuploaded.
- Cleanmandatorygridhadnoamplitude-gatewall-presentmiss;the.99stressmoved7mcaseprovidedtwoactualmisses. ThisisnotgeneralFNRforrealmaterials.
- Reliability/calibrationdidnotgeneralize;highremoved-wallFPRandnoisevulnerabilityarepreserved,notretunedaway. Single2-onlyfitlimitswhatthisfailurecanprove.
- InternalacousticpathGT/interpolationkernelareunavailable. NativeHSSDverticalproxyisnotvalidatedfurnishedsemanticmapping. Exactwallidentity/realmaterialclassification/localizationremainunverified.
-662recordsendbefore400ms,soLateEnergyiscensored;RT60isconditionalT20extrapolationonly. Allfailedinterpretationsandnegativecontrolsremaininthearchive.

## Statistical / exposure policy

Sixdesignedposes/threegeometryfamilies,tworeusedHSSDTrain scenes. NoTestpayloadormethodselection. HistoricvisualteacherTrain116/Val19andacousticTrain12/Val3exposureisinheritedbycachedHSSDcandidates. FreshRIRdoesnotmakeasceneindependent. Pooledconditioncountsarenotindependentscenes;poseeffectrangesandrepeatSDareusedwithoutfalseconfidenceintervals. NoPRmerge/deployment,newnetwork,Experiment2/3orfull135scene/densebankregeneration.
''')
 write('README.md','''# Experiment4A — Material-dependent Acoustic Observability

**Outcome B: Material-dependent but Ambiguous**. 材料계수의영향은관측되지만targetwall의증거와다른반사를안정적으로구분하지못했다.

- [연구 보고서](RESEARCH_REPORT.md)
- [Interactive Acoustic Material Explorer](interactive.html)
- [물리·재질 감사](PHYSICAL_MATERIAL_AUDIT.md)
- [원리 설명](METHOD_AT_A_GLANCE.md)
- [프로토콜](PROTOCOL.md)
- [재현·비용](REPRODUCIBILITY.md)
- [검산](verification.json)

기본corridor_near,예외room_far/single6. 실제저장조건만선택하며A/B공통시간·진폭으로비교한다. 원시RIR는로컬실험폴더에보존하고Git에는집계·설정/해시·화면용데이터만저장한다.
'''.replace('材料','재질'))
 print('reports written',d)
if __name__=='__main__':main()

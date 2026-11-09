> 역사적 연구 기록. 원본: `acoustic_cues_20261107_1830/external_context.md`. 원본 SHA256은 [산출물 목록](../artifacts.json)에 있습니다.
> 당시의 완료·재개 지시는 현재 실행 요청이 아닙니다. 최신 판정은 [현재 결론](../../../../current_findings.md)을 먼저 확인하세요.

# Simulator context and limits

Primary sources consulted 2026-11-07:
- [Official SoundSpaces2 API documentation](https://github.com/facebookresearch/sound-spaces/blob/main/SoundSpaces2.md): stochastic ray tracing; temporalCoherence smooths across previous simulationsteps and is inappropriate for discontinuousmotion. Its documenteddefault isfalse. Runtime here independently confirms0 (renderer_runtime_defaults.json); do notblame temporal smoothing without evidence.
- [SoundSpaces2.0 paper, supplementary acoustic rendering description](https://par.nsf.gov/servlets/purl/10424655): energy path histograms are converted to pressure responses with synthesized phase; direct and low-orderearlypaths are treated separately. This supports caution about equating arbitrary pressurepeaks/phase with measuredphysicalreflection identities, not a claimthat allphaseisrandom or useless.

Local inspected renderer: 3Dto2D_FP/scripts/render_hssd_exp4_ring_rirs.py::_render_ring calls6mono listenerpositions sequentially aroundsourcecenter. No explicitRLRrandomseed control. Localdefaults:temporalCoherence0, frequencyBands4; productionprotocol materialsfalse/transmissionfalse/48kHz/rays20000/source512/depth50/diffractiontrue. These actualsettings takeprecedence over genericpapercapabilities.

Newsamepose FPdiagnostic:2Train scenes×2poses×3renders, includingreversedlistenerprocessingorder. Baselinebankunchanged. Sourcepeak sample7 relativeamplitude differences≈1e-8, whereaswhole0–2mswindowincludesnearreflectionsandislessstable. Mean sameposeembeddingcosine.865 vsdifferentpose.067 acrosssmallselectedposepairs. Rawlatecosine ishigh evenbetweenposes; earlyresponsesdifferentiate more butalso vary acrossrenders. These observations alone donot identify mechanism orprove allrecords robust. Report rawwindow,coherency,embedding variability separately inrepeat_analysis.json. Thisisnot anindependentlocalizationdataset.

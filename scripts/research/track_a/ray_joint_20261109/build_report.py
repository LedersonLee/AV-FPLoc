from pathlib import Path
import json
P=Path(__file__).resolve().parent
a=json.loads((P/'analysis.json').read_text());e=json.loads((P/'extension_analysis.json').read_text());f=json.loads((P/'followup_analysis.json').read_text())
s='''# Ray Representation & Joint Acoustic–Visual Learning

Track A · 2026-11-09 · Evidence level: **Exploratory** · Full-bank evaluation pending; this is an interim report.

## Findings and competing explanations

The geometry oracle shows substantial information beyond four extents, but learned ray precision does not establish localization utility. Frozen contrastive embeddings remain competitive with newly trained STFT models. Under this frontend and budget, neither chirp convolution nor joint attention establishes an advantage over its strongest matched control. The experiment does not reproduce or refute AVLoc's complete system.

A particularly informative negative control concerns vision outside its field of view: restricting late fusion to frontal rays recovers most of the gap between naive late fusion and acoustic-only. This supports avoiding unsupported visual extrapolation; it does not show useful cross-modal interaction. Learned joint attention has not surpassed the strong B2+GEOQ framework baseline.

## Protocol and validity

See [method](METHOD_AT_A_GLANCE.md), [protocol](PROTOCOL.json), [learning protocol](LEARNING_PROTOCOL.md), [joint protocol](JOINT_PROTOCOL.md), [Phase4](PHASE4_PROTOCOL.md), [optimization followup](OPTIMIZATION_FOLLOWUP.md), [Transfer freeze](TRANSFER_PROTOCOL.md) and [FOV diagnostic](FOV_LATE_PROTOCOL.md).

New12 contains128queries per scene,3outer8fit/4heldscene folds and seeds42/43. Transfer11 models fit New12 only;18checkpoints were frozen before Transfer predictions. Both cohorts are reused development scenes with historical visual teacher exposure. Head-held evaluation is not independent end-to-end validation. No Test observations/results were used for method selection. Two seeds are averaged within each scene; reported95%CIs use20,000paired scene bootstrap samples without multiplicity correction. These are exploratory estimates, not confirmatory significance tests.

Joint recall means position≤1m and circular yaw≤30degrees. Fine recall means position≤0.1m without yaw restriction. Pure visual Top20 and full-bank evaluation are distinct. B2/GEOQ uses its own historical candidate pipeline; it is a strong framework comparison, not a same-candidate ablation. All controls preserve the common scene/query grid.

## Phase1: representation capacity

See [complete oracle report](PHASE1_REPORT.md). Full-bank oracle joint recall rises48.893→92.122→93.164→96.289% for4/8/12/36directions; fixedpureV Top20 rises69.206→79.102→80.143→83.919%. Fixedcandidate coverage is90.690%. Front12 yields56.576% on the fixedbank versus80.143% for omnidirectional12. Eight interpolated rays and rounded36-subset8 differ, so angular sampling is not simply output count. Fine recall is nonmonotonic. Oracle GT rays are nondeployable and never inference features for learned methods.

![Oracle](figures/oracle_capacity.png)

Interpretation: more angular coverage distinguishes floorplan shapes that four extents alias. Remaining alternatives include grid quantization and ray-casting mismatch; the oracle does not show that audio can predict those distinctions.

## Phases2/3: fixed pure visual Top20

All table rows below fuse their ray score with the same visual score(.85/.15). The original visual baseline is50.195%; samecandidate visual+cosine is57.487%. These are different from the learned vision-ray row.

| Model | K | Ray MAE(m) | Joint% | Fine% | Ambiguous pair% |
|---|---:|---:|---:|---:|---:|
'''
for key,v in a['summary'].items():
 if key.startswith('acoustic/') and not key.endswith('K36'):continue
 rk=key+'/subset'+key.split('K')[-1];r=a['ray_summary'].get(rk);s+=f"|{key.rsplit('/K',1)[0]}|{key.split('K')[-1]}|{r['mae']['mean']:.3f}|{100*v['joint']['mean']:.3f}|{100*v['fine']['mean']:.3f}|{100*v['pair_accuracy']['mean']:.3f}|\n"
s+='\n![Acoustic](figures/acoustic_representation.png)\n\nThe complete K and common36output-subset comparisons are retained in analysis.json. NativeK targets differ, so lower MAE across K is not a controlled precision claim. Raw/chirp share architecture and updates; E/C/N and frozen embedding have different input/parameterizations. The frozen embedding has a100ms input shift relative to its historical128ms contrastive training.\n\n| Paired contrast | Joint delta(pp) |95%CI(pp)|\n|---|---:|---|\n'
for name,v in a['contrasts'].items():
 x=v['joint'];s+=f"|{name}|{100*x['mean']:.3f}|[{100*x['ci95'][0]:.3f}, {100*x['ci95'][1]:.3f}]|\n"
s+='''
![Contrasts](figures/joint_contrasts.png)

Ordered raw channels outperform energy-sorted and isotropic controls trained with the same CNN. This supports useful spatial-channel information under the simulated microphone convention. It does not isolate IPD from amplitude or establish device-transfer robustness. A fixed channel permutation applied only at inference is a distribution shift, not the same causal test as retraining after removing identity. Donor/shuffle records are preserved. In joint models, `donor` means audio replaced with vision fixed; generic primary `donor17` rolls the complete joint prediction and must not be described as an audio-only intervention.

On fixed candidates, chirp improves mean ray MAE but its joint-recall contrast with raw includes zero. The complete full-bank evaluation below changes the scope of this conclusion. Fixed convolution adds no new scene information; it can change conditioning, spectral weighting and overlap. No frequency-specific mechanism is established without frequency-isolated controls. Attention improves naive late fusion, but not concat or equal-parameter interaction-disabled with decisive scene-level evidence; therefore attention-specific cross-modal benefit is unproven.

### Same36-output model, fewer matching directions

Holding the trained36ray model fixed separates matching-direction information from training capacity. Using all36 rather than4directions changes fixedcandidate joint recall by+1.758pp[0.423,3.092] for frozen embedding,+1.172pp[0.065,2.181] for chirp,+0.749pp[-0.358,1.823] for raw and+0.488pp[-0.456,1.497] for E/C/N. Gains beyond12directions are small and all corresponding intervals include zero. These reused exploratory comparisons support some value of broader angular sampling, not monotonic improvement with ray count.

Training a36output model and evaluating only4/8/12outputs does not clearly outperform training the corresponding nativeK model. K8 is not an exact grid subset: predicted log rays are interpolated while target metric rays are interpolated before log; its comparison includes this declared numerical difference. See remaining_analysis.json for all contrasts.

### Stronger directional falsification

The registered direction-shuffled control is a one-bin(10degree) cyclic shift, not a random permutation. Neighboring rays are correlated, so a small effect under that control cannot establish direction invariance. A subsequent fixed GT-free random36direction permutation diagnostic([protocol](DIRECTION_PERMUTATION_PROTOCOL.md)) retains all ray values but destroys their angular assignment on fixedpureV Top20.

Matched minus permuted joint recall is+1.335pp[0.586,2.116] for raw,+1.986pp[1.335,2.702] for chirp,+2.474pp[1.823,3.255] for embedding, and+0.814pp[-0.065,1.693] for jointattention. This supports usable directional assignment in the acoustic predictors under the current convention, without proving a physical causal cue or interaction mechanism. Full-bank results retain the registered10degree rotation; no random-permutation full-bank claim is made.

RawK36 inference controls additionally yield matched52.799%, donor48.763%,10degree shift52.669%,channel permutation51.335%,isotropic input48.796%,energy-sorted input51.009%joint recall. These inference shifts differ from the trainedsorted/isotropic controls in the main table; both are preserved.

## Optimization and FOV alternatives

Quadrupling training updates to1280 lowers training loss without clear held localization improvement. Attention160−concat160 is+0.879pp[-0.033,1.855]; attention160−disabled160 is+0.130pp[-0.944,1.367]. A wider concat bottleneck uses more parameters and only320updates; it is a diagnostic of bottleneck capacity, not an equal-budget superiority comparison. These results weaken simple undertraining as the sole explanation, while leaving model/frontend capacity and representation alignment unresolved.

| FOV contrast | Joint delta(pp) |95%CI(pp)|
|---|---:|---|
'''
for key in ['development/fov_late minus development/late','development/acoustic minus development/fov_late','transfer/fov_late minus transfer/late','transfer/acoustic minus transfer/fov_late']:
 x=e['fov_contrasts'][key]['joint'];s+=f"|{key}|{100*x['mean']:.3f}|[{100*x['ci95'][0]:.3f}, {100*x['ci95'][1]:.3f}]|\n"
s+='''
FOV-restricted late fusion uses vision only at11frontal ray bins and acoustic-only elsewhere. Its recovery relative to all-direction late fusion does not exceed acoustic-only. The plausible mechanism is removal of unsupported vision extrapolation, not discovered synergy. A transferable prediction is that a calibrated router should reject out-of-view visual structural claims before more complex fusion; this remains a hypothesis for another dataset, not a validated general rule.

## Phase4: matching and residual scales

Inner scene-OOF residual intervals cover approximately91–94% of outer-held log errors at nominal90%. They are global directional scales, not per-query reliability or guaranteed calibrated probabilities. No held labels fit these scales. Uniform matching, inverse-residual weighting and9direction geometry-difference selection are compared on identical predictions. Pair selection uses candidate FP geometry only; no GT picks directions.

![Calibration](figures/calibration.png)

| Prediction | Uniform joint% | Uncertainty joint% | Pair-selection joint% |
|---|---:|---:|---:|
'''
for model in ['acoustic/raw','acoustic/chirp','acoustic/embedding','joint/attention']:
 vals=[100*e['conditioned'][model+'/matched/'+m+'/visual_geometry']['joint']['mean'] for m in ['uniform','uncertainty','pair_selection']];s+=f'|{model}|'+ '|'.join(f'{x:.3f}' for x in vals)+'|\n'
s+='''
There is no clear fixedcandidate benefit from the proposed matching refinements. Geometry differences identify potentially informative directions, but not whether audio predicts those directions accurately. Selecting the largest geometric contrast can amplify prediction error; global residual weights may miss query-specific reliability. These are interpretations consistent with the negative result, not uniquely proven causes. Full-bank anchor selection is evaluated separately and must not be confused with all-pair Top20 Borda scoring.

### Pair-selection case diagnostics

The first repair and harm in protocol scene/query order are retained without selecting attractive examples. Scene102344250 query1229 changes3.9m error to0.1m; scene102816615 query1041 changes0m to1.72m and85degree yaw error. Selected directions for the two displayed choices are illustrative; the actual Borda rank uses all candidate pairs. They show why geometry contrast alone does not guarantee accurate acoustic evidence.

![Repair](figures/pair_repair.png)

![Harm](figures/pair_harm.png)

## Transfer11 (reused, teacher-exposed development)

| Arm | Joint% | Fine% |
|---|---:|---:|
'''
for key,v in e['transfer'].items():
 if key.endswith('/matched/uniform/visual_geometry'):s+=f"|{key.split('/matched')[0]}|{100*v['joint']['mean']:.3f}|{100*v['fine']['mean']:.3f}|\n"
for key,v in e['transfer_baselines'].items():s+=f"|Baseline {key}|{100*v['joint']['mean']:.3f}|{100*v['fine']['mean']:.3f}|\n"
s+='''
![Transfer](figures/transfer.png)

Transfer paired contrasts are chirp−raw+0.391pp[0,0.817], attention−concat+0.639pp[-0.391,1.598], attention−disabled−0.604pp[-1.527,0.178], and attention−acoustic−0.497pp[-1.101,0.036]. Thus these results do not establish a robust attention advantage.

The frozen embedding remains the strongest new ray arm by mean, but below samecandidate visual+cosine and far below B2+GEOQ. The full strong-baseline gap is retained, not presented as a successful framework replacement. Transfer here is predictor-fit scene exclusion, not an independent untouched benchmark. No full-bank Transfer claim is made: the frozen Transfer protocol specifies fixedcandidate evaluation.

## Failures, corrections and preservation

- Strict deterministic CUDA backward rejected adaptive pooling. Replaced the16×16→4×4 operation with mathematically equivalent fixed4×4 average pooling; strict determinism retained, equivalence checked. Earlier vector fits and failed logs preserved.
- Two short3534sample RIRs triggered the initial100ms cache check. Applied the existing zero-tail contract to both raw/chirp inputs, preserving original samples and recording padding. This is not new rendering or additional evidence.
- Initial joint Phase4 donor rolled the complete prediction. Corrected to a forward pass with donated audio and unchanged vision; initial artifacts remain in diagnostics/pre_joint_donor_correction.
- Initial Transfer ambiguity diagnostics used36ray geometry instead of the established coarse4mask. Corrected diagnostic masks, preserving initial files; localization scores did not change.
- Full-bank Phase4 implementation shares candidate-only direction weights across model/control runs to avoid repeated computation. Original stopped logs remain. Scalar checks verify the scoring formula and visual anchor equivalence.

## Reproducibility and unresolved limits

All228checkpoints and1542stored prediction arrays replay exactly(max absolute error0), with matching receipt hashes, disjoint fit/held IDs and matching query IDs. Uniform Phase4 matches primary metrics exactly across1152comparisons; vectorized Spearman and pair scoring agree with independent implementations within2.23e-16. An additional528checks recomputed every inner model normalization, OOF residual scale and held coverage; all matched exactly and excluded outer-held scenes. This verifies saved inference and calibration bookkeeping, not independent retraining. Source dependencies, container and local input hashes must accompany reproduction; caches/checkpoints are not uploaded. Clean-clone reproducibility without historical assets is not claimed.

Results apply to simulated HSSD6channel RIRs, pooled STFT/frozen vision and these budgets. They do not establish real-world, furniture/material/device shift robustness, the full AVLoc Laplace/router method, or Track B performance. Reused cohorts and multiple exploratory comparisons constrain evidence level. Next informative research should target calibrated query/direction reliability and ray-prediction errors that alter pose ordering, rather than assuming more rays, chirp or attention will help. A new independent scene protocol is required before promoting conclusions.

## Full-bank completion status

Pending. Primary all-K/subset evaluation and Phase4 control evaluation are still running. This report does not close the study or authorize a success claim before their aggregation, final provenance and Docs synchronization.
'''
# Add complete primary full-bank results without hiding the earlier fixedcandidate conclusion.
if (P/'fullbank_evaluation.json').exists():
 remaining=json.loads((P/'remaining_analysis.json').read_text());fb=remaining['fullbank_evaluation.json'];s=s.replace('Under this frontend and budget, neither chirp convolution nor joint attention establishes an advantage over its strongest matched control.','On fixed visual candidates, chirp and attention gains over raw/concat remain uncertain. Full-bank evaluation subsequently finds chirp over raw and attention over concat gains, but attention does not clearly beat interaction-disabled and these ray methods remain below strong localization baselines.')
 s=s[:s.index('## Full-bank completion status')]
 s+='''## Full-bank localization: primary evaluation complete

All576model-scene blocks are complete, yielding2880aggregate records across nativeK and36output subsets. The same20cmXY/10degree yaw bank is searched; scores are standardized over the full bank, not the fixed20candidates. Thus differences from fixedcandidate evaluation include both search space and score normalization, and cannot be attributed to candidate extraction alone.

|36ray prediction|Joint%|Fine%|Top20 coverage%|
|---|---:|---:|---:|
'''
 for key,v in fb.items():
  if key.endswith('/36/36/visual_geometry'):s+=f"|{key.split('/36')[0]}|{100*v['joint']['mean']:.3f}|{100*v['fine']['mean']:.3f}|{100*v['top20']['mean']:.3f}|\n"
 for key,v in remaining['fullbank_baselines'].items():s+=f"|Baseline {key}|{100*v['joint']['mean']:.3f}|{100*v['fine']['mean']:.3f}|{100*v['top20']['mean']:.3f}|\n"
 s+='\n|Paired full-bank contrast|Joint delta(pp)|95%CI(pp)|\n|---|---:|---|\n'
 for key,v in remaining['primary_fullbank_contrasts'].items():
  x=v['joint'];s+=f"|{key}|{100*x['mean']:.3f}|[{100*x['ci95'][0]:.3f}, {100*x['ci95'][1]:.3f}]|\n"
 s+='''
**Updated interpretation:** Chirp's same-network intervention improves full-bank joint recall by2.897pp and fine by1.367pp[0.293,2.474] versus raw. Attention improves concat by3.060pp, but attention−interaction-disabled remains uncertain(+0.716pp[-0.391,1.790]). This establishes a task-specific relative gain for chirp and an attention-architecture gain over the chosen concat bottleneck in this reused cohort; it does not establish cross-modal information exchange as the cause or general superiority to stronger localization.

The strongest new ray arm(frozen embedding) has+1.009pp[-2.311,4.590] joint versus visual, with fine−1.628pp[-2.832,-0.456]. Raw and attention both significantly reduce joint and fine versus visual under these exploratory intervals. Full-bank visual+cosine reaches67.708%joint and27.214%fine, versus51.204%/20.898%for embedding rays. The distinct historical B2 pipeline reaches72.103%joint/24.740%fine; B2+GEOQ reaches75.977%joint/25.065%fine on this cohort. GEOQ's better joint recall does not imply superior fine accuracy to full-bank visual+cosine. These regressions prevent a successful framework-replacement claim.

![Fixed versus fullbank](figures/fixed_vs_fullbank.png)

The wider score field exposes false peaks that a visual shortlist filters out. Larger chirp/concat contrasts here may reflect global error-tail behavior and candidate-z normalization rather than a new source of information. A future discriminating study would hold normalization fixed while varying the search field and analyze false-peak geometry; that mechanism remains unverified. Fixedcandidate repairs alone do not justify full-bank deployment.

'''
 if (P/'fullbank_analysis.json').exists():
  finalfb=json.loads((P/'fullbank_analysis.json').read_text());s+='## Full-bank conditioned matching\n\nAll96model-scene blocks and1728records are complete. Uniform metrics agree exactly with the independently evaluated primary implementation. Full-bank direction selection uses a visual Top1 anchor; it is distinct from fixed20pair Borda.\n\n|Prediction|Uniform joint%|Uncertainty joint%|Anchor-selection joint%|\n|---|---:|---:|---:|\n'
  for model in ['acoustic/raw','acoustic/chirp','acoustic/embedding','joint/attention']:
   vals=[100*finalfb['conditioned'][model+'/matched/'+method+'/visual_geometry']['joint']['mean'] for method in ['uniform','uncertainty','anchor_selection']];s+='|'+model+'|'+'|'.join(f'{x:.3f}' for x in vals)+'|\n'
  s+='\nPaired changes, fine/coverage tradeoffs and donor/shuffle controls are retained in fullbank_analysis.json. Interpretation is finalized after inspecting those contrasts; no GT chose anchor directions.\n'
 else:s+='## Remaining completion work\n\nFull-bank conditioned matching is still running. Its comparisons, final audit and final Docs synchronization remain required. This is not a completion claim.\n'

if (P/'fullbank_analysis.json').exists():
 s=s.replace('Full-bank evaluation pending; this is an interim report.','Completed controlled study; mixed findings and negative baseline comparisons retained.')
 s=s.replace('Paired changes, fine/coverage tradeoffs and donor/shuffle controls are retained in fullbank_analysis.json. Interpretation is finalized after inspecting those contrasts; no GT chose anchor directions.','''Uncertainty weighting improves attention full-bank joint recall by2.181pp[1.302,3.027] and fine by1.270pp[0.521,2.051], while acoustic raw/chirp/embedding changes are small and their intervals include zero. Attention with weighting reaches45.638%joint/17.741%fine, still below visual50.195%/22.526%, fullbankvisual+cosine67.708%/27.214%, and the strong B2+GEOQ framework. A locally effective correction is not a superior localization pipeline.

Anchor selection harms joint recall relative to uniform: raw−4.395pp[−6.478,−2.441], chirp−4.720pp[−6.445,−2.767], embedding−4.753pp[−5.794,−3.711], attention−3.678pp[−4.785,−2.701]. Fine recall also decreases for all four. The selected geometry differences can be large precisely where prediction errors or anchor errors are consequential. This is a plausible failure mechanism, not uniquely identified causality.

Matched minus donated-audio uniform joint recall is+15.039pp[11.589,18.979] raw,+18.034pp[14.746,21.875] chirp,+21.029pp[17.806,24.089] embedding,+10.970pp[8.040,14.258] attention. Attention donor retains its original vision input. Thus query-specific acoustic evidence exists, but its usefulness does not guarantee an improvement over strong visual or FP-RIR baselines. The10degree rotation remains a weaker directional control; the separate fixedcandidate random-permutation test addresses that limitation.

Paired fine/coverage tradeoffs and all control contrasts are retained in fullbank_analysis.json. No GT selected anchor directions.''')
 s+='''
## Conclusions and scientific stopping boundary

**Observed:** More geometry directions greatly improve the oracle, while learned ray localization improves much less. Chirp beats raw in fullbank; attention beats the chosen concat but not clearly interaction-disabled; residual weighting helps fullbankattention. Geometry-difference anchor selection fails, and none of these new ray pipelines beats the strong B2+GEOQ baseline.

**Supported interpretation:** A visual shortlist can conceal global false peaks; angular assignment carries useful acoustic information; globally unreliable direction outputs can hurt matching. The controlled interventions support these local statements under the specified frontend and reused cohorts.

**Unverified causal hypotheses:** Chirp may improve conditioning/spectral weighting; uncertainty scales may suppress direction-dependent extrapolation errors; selecting large geometry differences may overemphasize poorly predicted rays. Alternative normalization, bottleneck capacity, frozen teacher priors and observation/model mismatch remain possible explanations. No physical reflection mechanism or cross-modal information-exchange advantage is proven.

**Next discriminating research:** Separate bank normalization from candidate-space expansion, test query-specific direction reliability against global scales, and use a newly defined independent scene evaluation before promoting evidence level. No Test data was consumed to choose these proposals. Real-world/furniture/material/device shifts remain outside scope.

All four experimental phases, required controls and bounded followups are complete. Scientific completion is based on coverage of the requested hypotheses and falsification tests, not on report creation or obtaining a positive result. The report does not claim the broader AV-FPLoc research program is complete. See [completion audit](completion_audit.json), [reproduction](REPRODUCE.md) and source/model inventories; final Docs synchronization is verified after committing and recorded separately to avoid a self-referential Docs SHA.
'''
(P/'RESEARCH_REPORT.md').write_text(s)

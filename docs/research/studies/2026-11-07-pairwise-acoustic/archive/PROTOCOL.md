> 역사적 연구 기록. 원본: `pairwise_acoustic_evidence_20261107_2100/PROTOCOL.md`. 원본 SHA256은 [산출물 목록](../artifacts.json)에 있습니다.
> 당시의 완료·재개 지시는 현재 실행 요청이 아닙니다. 최신 판정은 [현재 결론](../../../current_findings.md)을 먼저 확인하세요.

# Pairwise acoustic evidence — active protocol

User specification: /home/jeongeonlee/.codex/attachments/8abeb694-4447-47b8-9cd6-6c42ab0f0355/붙여넣은 텍스트.txt. Preserve all previous outputs, originalB2/GEOQ andfixedsupport20. Localresourcesfreelyauthorized. NoTest. Noagentsrequested.

Hypotheses before new results:
Hstable: useful early/fullC andT/N have between-pose separation largerthanwithin-pose render variability. Near poses, structurally different poses and visuallysimilar spatiallyseparated hypotheses are different comparators. NeedbothFPandnativeObservedwhenfeasible; no RLRseedcontrolassumed.
Hconditional: query-conditioned nonlinear acoustic residual differences between visuallysimilar hypotheses containinformation beyond existingGEOQ156inputs. Scalaracoustic ranking alreadyinGEOQ; C0notaudiofree.
Hloss: existingcandidatewisehandcraftedsummary may discard fine temporal/channel patterns orpair-relativecontext; objectivechange andfeaturechange mustbe separated.
Hmismatch: FP/native mismatch andstochasticvariation canlimitcueuse. Donorvariationcontrolneeded, notsingledonor.

PhaseBfirst: all190unorderedpairsfromoriginal20, usingGTfreevisualsimilarity+spatial/yawseparation, Trainfixedthreshold. GTlabels only evaluation/supervision. Queryaveragedmetrics before sceneaggregation; pairsnotindependent samples. Full20Top1separatefrommixedlabelpairaccuracy/AUC. RetainfixedTrainambiguity andaddbothpureTop1fail/fusionTop20success (differentfrompreviousfusionTop1group).

PhaseC: originalGEOQ replay; matchedsameutility150x15treecandidate models with candidateconditionedcue(C1), challenger-relativepair evidence(C2), visual-acousticagreement(C3), at least3within-scenequerydonor permutations(C4), deterministicbaseline-onlybasisexpansion(C5), andzeroaddedfeaturecontrol. Same nominalwidth/optimizer/budget; activefeaturecapacitycaveat. Training9scenes inouterfold usespreviousnestedadapter caches, heldsceneoriginalhypotheses. CandidatefeatureconstructionGTfree. Relativeutility joint+.5fine vsoriginalcandidate0 retained. ThresholdsTrainonly, frozenoriginalthresholds primary.

Conditional learning follow-up depends onTrain evidence; comparebaseline-onlypair objective versuscuepair objective if promising or neededtofalsify learnerlimitation. Do notchoosefeaturesfromDev. Val8/Val11 reuseddevelopment; no independentgeneralization. Bootstrapdoesnotremoveexploration/multiplecomparisonbias.

Repeatability: representativeTrain subset first; samepose independent repeats,nearby,structuraldifferent,visuallysimilar candidates. Fixedproductionraycounts/source/micconfig, sequentialmono limitations andnoexplicitacousticseedcontrol documented. CaptureT/N/C, early/fullC, embedding/intermediate, retrieval/ranking andwithinbetweenvariance.

Completion requires controlled/falsificationfollow-ups, notfirstbatch/report. Tenuserdeliverables and reproduciblecode/config/model/results/failedlogs. Negativeperformancefindings validwithsufficient evidence.

Follow-up registered after Train pair opportunity analysis, before new pair learner fits: Train ambiguous low-GEOQ-margin subset has standalone acoustic/cue pair accuracy ~57–58% vs GEOQ52.32%, but entire ambiguous group GEOQ90.26% dominates. Test pair-utility regression factorial with baseline inputs versus identical C1 candidate summaries. Same 150x15 trees/876 nominal dimensions. Train24 closest-visual spatial/yaw-separated unordered pairs per query (fallback remaining separated then allpairs only if insufficient), both orientations, U_i-U_j labels. At inference all190pairs, antisymmetrize predictions and aggregate mean pair utility (Borda), subtractcandidate0 and frozenoriginalswitchthreshold. This tests objective/aggregation effects separately from input additions; pair and candidate learn different target distributions. No Dev-driven arm selection.

Second pair-learning falsification, triggered by Train P0 joint75.12% vs GEOQ76.94% before inspecting P0 development outcomes: close-visual-only training24pairs versus full190pair inference creates a support mismatch. Register P0_broad/P1_broad with same objective, budget, dimensions and aggregation, but12GT-free closevisual pairs plus12uniformremaining pairs, deterministicquery-keyed seed. This tests pair-training distribution separately and retains base/cue factorial. No changes to already-running firstpairmodels. Full190training not required;24sampled pairs retain matchedbudget. Development metrics do not select this rule.

Same-pose domain falsification: prior FP-within-domain retrieval and nativeObserved-to-originalFP ranking differed, but poses/conditions were confounded. Register 8exact Observed physicalsource/receiverposes rendered inFPstage,4freshrepeats each (32renders), identicalyaw/ray/settings, then compare within-domainrepeatdistance to same-pose cross-domain distance. This isolates pose as a confound, not furniturealone: domainstage geometry/materials may also differ. No trainedranking model uses these evaluationposes/labels.

Frozen pair-aggregation falsification registered after close-pair Train P0/P1 underperformed: compare antisymmetric candidate-vs-originalcandidate0 utility directly (19comparisons) against all190pair Borda, keeping the trainedmodel/input and frozen switchthreshold. First inspect TrainOOF; this separates aggregation from trainedpairfeatures without newfitting. Any development follow-up must report both base/cue and be labeled follow-up, not independent selection evidence.

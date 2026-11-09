# Phase4 before execution

RawSTFT exceeds trainedcanonical-sort andisotropiccontrols onfixedpureVjoint by+1.595pp[.716,2.441] and+1.758pp[.814,2.669]. Chirp−raw andattention−concat intervalscrosszero; these are notprovenimprovements. Direction-sensitiveevidence warrants boundedcandidate-conditionedtests.

Models: acousticraw/chirp/embedding andjointattention,K36,bothseeds/all3outerfolds. For eachouter8fitscenes, make2innerfolds(4fit4validation). Train identicalarchitecturefor320updates,batch128 (80epochs over512rows),sameoptimizer/lograyMSE andseed. Predict8outerfitscenes OOF. Noouterheldlabels participate.

Per-direction meanabsolute log residual yields fixedsigma clippedmin.02. Fixed90interval uses OOF90thpercentileresidual. Reportactualouterresiduals, intervalcoverage andrelativeerror; do notcallthem calibratedconfidence merelybecauseestimated. Scalarper-directionuncertaintyisglobalwithinouterfold, notquery-specific.

FixedpureVTop20: uniformnegativeL1; inverse-sigma weightedL1; pair-specific directionselection uses9of36 largestabsoluteFPlog-raydifferences for eachcandidatepair, thenaveragespairwise residualadvantages overopponents. NoGT oracousticprediction choosesdirections. Direction-shuffle andsame-scenedonor17 repeatmatching. Evaluatewithandwithoutfixed.15Vfusion, comparecosine/GEOQ.

Fullbank: uniformanduncertainty asabove. All-to-allpairmatching isquadratic inbankposes; useexplicitlydifferent GT-free anchorextension: eachpose comparedwithoriginalpureVTop1anchor, use9directions oflargestgeometrydifference, weightedresidualadvantage. Reportthisextensionseparatelyfrom20candidateBorda aggregation; retainfullbankposegrid/visualbaseline. Theanchoritselfhaszerogeometrydifference andscore0. Donor/shufflecontrols required. NoanchororK-selectionusingGT. Recordcompute.

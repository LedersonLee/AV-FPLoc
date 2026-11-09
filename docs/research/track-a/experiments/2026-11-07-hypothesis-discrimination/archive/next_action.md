> 역사적 연구 기록. 원본: `hypothesis_discrimination_20261107_1538/next_action.md`. 원본 SHA256은 [산출물 목록](../artifacts.json)에 있습니다.
> 당시의 완료·재개 지시는 현재 실행 요청이 아닙니다. 최신 판정은 [현재 결론](../../../../current_findings.md)을 먼저 확인하세요.

# Research cycle completed

Full active goal audit: completion_audit.md. Scientific report: RESEARCH_REPORT.md. Artifacts and existing user services preserved; no job remains queued. Test sealed. All19evaluation scenes were already used development data; no newvalidation/generalization claim.

Final precision-aware utility ranker: same20support/refinement hypotheses and fixedB2 scores;52features(candidate26+baseline-relative26);150trees/15leaves; target difference joint+.5fine. Train OOF selects thresholds.025(seed42),.1(seed43).
- Val8 baseline72.6014→ranked73.4964% joint, +.8950pp CI[.6816,1.1336]pp. R.1+.2506pp CI[.1356,.3627]pp.
- ReusedVal11 baseline70.0779→ranked70.6753%, +.5974pp CI[.3251,.8671]pp. R.1+.0693pp CI[-.0807,.2489]pp; not a formal fine-precision improvement/noninferiority claim.
- Bothseeds improved onbothsets. Visual-harm counts fell from146→110,149→131(Val8),184→148,196→178(Val11).
-19,930exact inference replays;190nonbaseline-candidate permutations preserve predictions; all evaluatedcandidate sets20distinct; geometry31scenes/maphasheschecked; prior518artifacthashes andoriginalgitdiff unchanged.

Mechanism: absoluteclassification/scores-only insufficient; baseline-relative feature removal largest loss in bothseeds; local/modality/spatial descriptors addinformation. Purejoint gain causedprecisionloss; fine-awareutility reduces unnecessaryswitches1071→273(seed42), improvesnetjointgain andfine. Fixedranker onconnectedvsEuclidean support essentiallysame, so wall-connectivitynotmaincauseinthissetting. Coverage~96% remains faraboveTop1; this doesnotexhaustheadroom.

Future research only: newdata frozenvalidation beforeanygeneralizationclaim; fullscene-nestedupstreamCV; wall-dense data andgeodesicmetriccontrol; structuredrayprofiles/hypothesisattention; context-vs-irrelevant-featurecapacitycontrol. Existing Val8/Val11 mustnotbe relabeledunused. No furtherrequiredwork remains for this empiricalgoal.

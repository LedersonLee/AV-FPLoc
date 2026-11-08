> 역사적 연구 기록. 원본: `structural_evidence_20261107_1730/next_action.md`. 원본 SHA256은 [산출물 목록](../artifacts.json)에 있습니다.
> 당시의 완료·재개 지시는 현재 실행 요청이 아닙니다. 최신 판정은 [현재 결론](../../../current_findings.md)을 먼저 확인하세요.

# Research stage complete, independent confirmation remains

Authoritative report: RESEARCH_REPORT.md.34matched ranker arms×2seeds,12outer-adapter OOF arms×2seeds,72regeneratedoutertrainingcaches,62matchedB1/B2scene evaluations completed. No owned research worker intentionally left running.

Final Train-selected method: GEOQ, seed42threshold0, seed43threshold.01. Frozen hashes/options final_selection_seed42/43.json. Joint improvement vs strongB0 Val8+1.301pp[.656,2.070], Val11+1.143pp[.668,1.724]. These are exploratory reused-development effects; no independent claim. API gain replay exact19930rows. Earliercandidate-onlyGEO atpriorfixedthreshold improves+1.492/+1.169pp; control mechanism is geometry×score conditioning, not new acoustic information or cross-attention.

Causal revision: compact residual gain survives donor retraining; candidate-onlygeometry stronger; samefeatures/capacity interaction restriction losesgain; geometry partly linearlydecodablefromacousticembedding but decodedranker fails to reproduce directgeometrygain. No claim allrawsignalinformation exhausted. Fullmodalityremoval and rawwaveformtraining notdone.

MatchedtrainedB2−B1 fusionMAP Val8+.346pp CI[0,.701],Val11+.061pp CI[-.256,.354]; small exchangegain, not universalfailureproof. Priorfrozen routes-off distinguished.

Existing437coverageartifacts plus332hypothesisrankerartifacts and originaluserGitdiff preserved. All current receipts/modelhashes/candidateidentity checked in final_verification.json. artifact_manifest.json hashes nonlog artifacts, excludingitself and __pycache__. Any later edits require rerunning final_verification.py to update manifest.

Follow-up only if continuing research: independent scenes first under frozenmethod; then matchedsimulation/furniture and fullRIRearlyphase/amplitude probes on remainingoracle-present failures. Test remainedunopened. Pretrainedencoderexposure stilllimits OOF. Val8/Val11 reused; do not rename untouched. Currentstage achieves reproducibleimprovement plus controlledmechanisticdiagnosis, not proof ofgeneralization.

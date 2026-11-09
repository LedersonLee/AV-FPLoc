> 역사적 연구 기록. 원본: `pairwise_acoustic_evidence_20261107_2100/next_action.md`. 원본 SHA256은 [산출물 목록](../artifacts.json)에 있습니다.
> 당시의 완료·재개 지시는 현재 실행 요청이 아닙니다. 최신 판정은 [현재 결론](../../../../current_findings.md)을 먼저 확인하세요.

# Research state and resumption

Current specification: Acoustic Evidence for Pairwise Hypothesis Discrimination, attachment8abeb694. Latest success criterion is reproducible knowledge about acoustic evidence/stability/conditionalvalue, not mandatory architecture or performance gain. Local compute was unrestricted; no resource-limit claim.

Completed research loops: preserve/audit → ambiguouspair evidence → candidate/pair/agreement matchedlearning → three donor retrainings and frozen removal → close-pair utility factorial → broaderpair-training distribution factorial → frozenaggregation falsification → repeatedscorepooling → exact-pose native/FP domain contrast. Thirteen fitted arms × two seeds × (four outerfolds + fullfit) =130 models. Main160 plus domain32 fresh renders. No Test.

Conclusion to retain: acoustic evidence distinguishes visuallysimilar hypotheses, but tested added blocks/learners do not show a reproducible downstream gain beyond GEOQ. Same-pose native/FP stage differences exceed within-stage variability on this small matchedsubset. This supports investigating reference alignment, not claiming furniturecausality or impossibility of better acoustic learning. Keep original GEOQ baseline and existing checkpoints.

Stop rationale under current specification §9: controlled/falsification follow-ups provide sufficient diagnostic evidence; more architecture/capacity/threshold searches on repeatedlyused development scenes have low information value. This is not a budget exhaustion or blocked task. Existing independentgeneralization and physicalcausality questions remain explicitly open.

Resume with new evidence: obtain additional preregistered scene/pose observations (not repurposed previouslyexamined validation), replicate paired-domain/repeat findings, then test confidence-aware referencealignment under Train-only selection. Keep candidate coverage changes separate from fixed20ranking. If a newrepresentation gives Trainconditionalgain, matchcapacity/objective and retain multiplequerydonor controls before developmentconfirmation. Avoid tuning on Test.

Re-run artifacts: follow REPRODUCE.md. Models skip completedresults; isolate a new outputdirectory for freshreplication ratherthan deleting currentcheckpoints. PROTOCOL.md records hypothesis timing. RESEARCH_REPORT.md, paired_statistics.json and final_verification.json are authoritative after finalization.

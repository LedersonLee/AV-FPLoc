> 역사적 연구 기록. 원본: `structural_evidence_20261107_1730/PROTOCOL.md`. 원본 SHA256은 [산출물 목록](../artifacts.json)에 있습니다.
> 당시의 완료·재개 지시는 현재 실행 요청이 아닙니다. 최신 판정은 [현재 결론](../../../../current_findings.md)을 먼저 확인하세요.

# Candidate-conditioned structural evidence study

Preserve original B2 scores,20support/refinement indices,52-feature utility models and all previous outputs. All Val19 reused development; Test sealed. Train12 scene-grouped OOF selects thresholds; architecture/features are hypotheses, not established causes. Upstream B2 OOF features are not fully nested with ranker outer folds: features for ranker-training scenes can have upstream adapters trained on ranker-held scenes. Treat this as a leakage limitation and address matched full/fold-feature regeneration if promising effects warrant it. No independent-data claim.

First controlled batch keeps original150iterations/15leaves/minleaf100/L2=10/lr.07 HistGBRegressor and utility target joint+.5fine minus candidate0. FrozenB0 remains separate. RefitB0 with identical fixed objective and fixed threshold alongside B1 visual per-ray evidence, B2 candidate acoustic embedding evidence, B3 concatenation, B4 explicit cross-feature/context. All fixed-threshold comparisons use prior seed thresholds.025/.1; TrainOOF adaptive thresholds separately reported. Capacity controls use same leaf/iteration budget; feature-dimensionality/search effects additionally controlled by deterministic GT-free noise or permutation blocks. No new threshold tuning on Val.

Initial features: signed/absolute normalized residual at11camera rays, expected/measured rays and uncertainty, angular differences; candidate FP128 and adapted query128 embeddings with per-dimension products and absolute difference. These acoustic features are learned embedding evidence, not physical reflection correspondences. Intermediate visual token projections and acoustic token stream summaries are separately labeled stages. Additional raw E/C/envelope comparisons follow availability audit, not assumed equivalent to rawRIR.

Baseline replay requires previous predictions/candidate arrays/labels exact. New features computed before labels and without truth, absolute GT pose or sceneID. New data directory only; source mounts read-only. Candidate-conditioned inference latency measured separately from frozen score/extraction cost.

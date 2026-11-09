> 역사적 연구 기록. 원본: `fusion_vs_posterior_20261107_1300/findings_checkpoint.md`. 원본 SHA256은 [산출물 목록](../artifacts.json)에 있습니다.
> 당시의 완료·재개 지시는 현재 실행 요청이 아닙니다. 최신 판정은 [현재 결론](../../../../current_findings.md)을 먼저 확인하세요.

# Development checkpoint (not final)

Matched Val8,4190 observations,2 seeds. Same weights passed to each decoder. Joint means:
fixed MAP .7101431981; adaptive MAP .7147971360;
fixed support .7278042959; adaptive support .7221957041;
fixed refined .7260143198; adaptive refined .7207637232.
Difference-in-differences for adaptive vs fixed support gain is -.0102625298, scene bootstrap95% [-.0139016740,-.0065660891]. This is descriptive development evidence, not independent generalization. It contradicts unconditional claims that adaptive selection always fails, but also shows the tested MAP-trained adaptive selector reduces the support decoder's benefit. Objective mismatch is a plausible mechanism, not established causality yet. A follow-up should fit matched support-success labels with identical gate architecture/features and compare.

Eight-weight MAP oracle joint: seed42 .8241050,seed43 .8183771. Fixed Top20 oracle hypothesis recall .9315036/.9288783; Top100 .9599045/.9584726. Restricted weight optimization has less headroom than choosing among score hypotheses. Adjacent-pose redundancy means these TopK values should be supplemented by spatially separated candidates; support-field TopK is particularly redundant (Top20~79.7%). Never deploy oracle labels.

Train42 categorical nearest-cell NLL on13 predeclared temperatures selects .05, the existing support temperature. Macro NLL3.8793, vs .03:4.5376 and .075:4.0549. Calibration hypothesis did not propose a new model in this search. Separate posterior support-field point/region NLL and ECE findings from v2 remain relevant; normalized local support is not automatically a better calibrated posterior.

Five baseline methods (V,A,fixed MAP,support,refine) reproduced bit-exact indices on all8380 seed-query rows (`factorial_replay.json`). Prior v2 artifacts255 unchanged. Input preparation matches original64 acoustic batch and4 vision batch; smoke64 reproduced all tensors exactly.

Independent Val11 input preparation is running. No new Val11 method outcomes have been inspected yet; confirm.py writes method/scene/model/code contract before inference. It should only be launched after all11 input receipts are ready, or run resumably with the same contract. Do not tune on Val11. Existing users' services remain untouched.

## Decoder-label intervention and frontier
Both seeds completed matched target intervention. Support-trained gate joint means: MAP .7173031026, support .7272076372, refine .7255369928. MAP-trained corresponding .7147971360/.7221957041/.7207637232. Target-only change recovers much of lost support benefit but does not beat fixed support .7278042959. It also improves MAP, so robust target labels compete with a decoder-alignment explanation.

Seed42 continuous scalar-weight oracle (positive-width winners plus endpoints, not isolated interior ties) .8458233890;8-grid oracle .8241050119. Separated fixed Top5/10/20 recall .9107398568/.9431980907/.9587112172; support .9143198091/.9441527446/.9594272076. Therefore ordinary support TopK weakness largely reflects spatial duplication, while hypothesis selection retains substantially more diagnostic headroom than continuous scalar weighting. Seed43 frontier still running; do not claim both-seed replication yet.

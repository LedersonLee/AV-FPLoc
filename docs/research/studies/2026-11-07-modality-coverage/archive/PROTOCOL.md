> 역사적 연구 기록. 원본: `modality_coverage_20261107_1700/PROTOCOL.md`. 원본 SHA256은 [산출물 목록](../artifacts.json)에 있습니다.
> 당시의 완료·재개 지시는 현재 실행 요청이 아닙니다. 최신 판정은 [현재 결론](../../../current_findings.md)을 먼저 확인하세요.

# Protocol recorded before diagnostic results

Goal: separate candidate availability, observation-informed coverage and actual ranking. All Val8/Val11 are reused development data. No Test observations accessed. Local resources authorized freely; preserve existing processes/checkpoints.

Implementation audit: `mbt_remaining_suite_20261106/model.py` Exchange.run projects both modalities, passes each through modality Transformers, averages latent tokens at configured routes, then subsequent blocks produce residual dv/da. Heads applies visual residual strength1, acoustic residual.5 and adapted acoustic decoder blend.5. Thus historical vision/acoustic labels denote exchanged branches, not pure modality inference. Pure visual uses original query.npz mu/scale; pure acoustic uses original observed embedding against the SAME frozen floorplan bank. These are pure inference inputs, not claims of matched training objectives or causal isolation of exchange: B2 also adds trained residuals/decoder adaptation. Exact baseline replay required.

Score: float64, visual negative ray residual sum/40, acoustic dot product; independently full-bank population z normalization. Index cell*36+yaw/10, original yaw0..350. Fixed fusion=.85 exchanged visual+.15 exchanged acoustic. NMS joint suppression XY<=1m and circular yaw<=30deg. Supports exp((score-max)/.05) summed within .4m and +/-10deg. Same extraction for every branch. Save centers and local-max refinement separately; refinement can reduce center separation. Success strictly <=1m AND <=30deg; fine strictly <=.1m irrespective yaw.

A: six fields (pure V,A,fusion and exchanged V,A,fusion), identical NMS/refinement/support variants, K1,2,3,5,10,20. No tuning temperatures; ambiguity entropies at .05,.2,1 assess sensitivity, not calibrated uncertainty claims.

B: full bank oracle;100 GT-free random-score NMS sets/scene,10 draws/query, mean/variance;20 GT-free farthest-XY sets with same SE2 suppression; same-scene half-cycle donor observation permutation preserves candidate score distribution but breaks query alignment (retains scene/location-distribution bias). Grid subsets .4m/20deg and .8m/40deg anchored original XY minima; same subset for all modalities, recompute branch normalization for fixed exchanged fusion. Record cell/pose counts,bounds,reachability; original NMS thresholds fixed in physical units. These subsets vary density and yaw together, do not identify individual causal contributions.

C: entropy at3temperatures, top2 mode margin, count within.2z of best, XY dispersion; ambiguity bin thresholds derived Train OOF only. Complementarity intersection/union versus fused K and equal-total-budget union V(K/2)+A(K/2). Posterior mass near GT, uniform baseline success fraction and best-success fullbank percentile diagnostic only; no GT inference features.

D: raw coverage gaps, frozen existing utility ranker on exactly its original20support/refined candidates, fixed-set V/A/fused reranking. Correct-vs-wrong candidate evidence contrasts diagnostic only. Further experiment chosen based on results, no validation threshold optimization. Scene paired bootstrap, seed-wise results, random sampling variability separate from scene variation.

Scope limits: pure-vs-exchanged differs trained adapters and acoustic head as well as exchange. Cached upstream encoders have prior validation exposure. Train OOF adaptation does not imply fully nested encoder independence. Simulator/furniture causal mismatch cannot be established solely by cached within-scene scores.

## Evidence-triggered follow-up log

Initial scene results showed large observation-vs-random coverage gaps and small visual-branch sensitivity to pairing. Added frozen-weight noexchange and within-scene donor interventions to separate interaction from other B2 adaptation; unchanged weights/decoder, matched full replay required. Added within-scene ambiguity rank correlations to test scene-scale confounding. Added ordinary Top20 with stable original-index tie breaking to quantify NMS diversity on identical scores/queries (Val19 bothseeds), because prior ordinary TopK used argpartition with different tie choice. Added a strictly diagnostic .1m boundary-tolerance audit; headline success definition unchanged. No new training/hyperparameter selection was justified or performed.

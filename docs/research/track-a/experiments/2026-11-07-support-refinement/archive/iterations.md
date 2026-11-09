> 역사적 연구 기록. 원본: `astra_fusion_v2_20261107_0225/iterations.md`. 원본 SHA256은 [산출물 목록](../artifacts.json)에 있습니다.
> 당시의 완료·재개 지시는 현재 실행 요청이 아닙니다. 최신 판정은 [현재 결론](../../../../current_findings.md)을 먼저 확인하세요.


## linear_seed42
- Observation: baseline joint 0.795188, candidate 0.792254.
- Hypothesis: Candidate score geometry can predict success (linear model).
- Intervention: Train-only candidate risk fitting; OOF threshold 0.1.
- Outcome: joint repair/harm 31/36; visual harm 31 vs 41.
- Remaining uncertainty: scene generalization and nonlinear candidate interactions.

## tree_seed42
- Observation: baseline joint 0.795188, candidate 0.794014.
- Hypothesis: Candidate score geometry can predict success (tree model).
- Intervention: Train-only candidate risk fitting; OOF threshold 0.1.
- Outcome: joint repair/harm 37/39; visual harm 22 vs 41.
- Remaining uncertainty: scene generalization and nonlinear candidate interactions.

## paired_seed42
- Observation: baseline joint 0.795188, candidate 0.787559.
- Hypothesis: Candidate score geometry can predict success (paired model).
- Intervention: Train-only candidate risk fitting; OOF threshold 0.
- Outcome: joint repair/harm 26/39; visual harm 31 vs 41.
- Remaining uncertainty: scene generalization and nonlinear candidate interactions.

## H7 support seed42 Val3
- Observation: 3 learned selectors failed to transfer gains from Train to Val.
- Hypothesis: isolated MAP peaks discard local spatial/yaw support.
- Intervention: sum posterior mass at radius .4m/yaw10deg; T=.05 selected Train only.
- Outcome: joint79.519→81.045%, R1 80.516→82.218%, visual harm41→35; R.1 drops .939pp.
- Remaining uncertainty: second seed, extra scenes, general smoothing effect, localization precision.

## support_seed43_val
- Observation: baseline joint 0.782277.
- Hypothesis: frozen support rule generalizes across upstream seeds / scenes.
- Intervention: identical temperature/radius, independent seed or extra Val scenes.
- Outcome: joint 0.805164; visual harm 37 vs 44.
- Remaining uncertainty: all8scene combined non-regression/precision and controls.

## support_seed42_extra_val
- Observation: baseline joint 0.658085.
- Hypothesis: frozen support rule generalizes across upstream seeds / scenes.
- Intervention: identical temperature/radius, independent seed or extra Val scenes.
- Outcome: joint 0.675382; visual harm 110 vs 104.
- Remaining uncertainty: all8scene combined non-regression/precision and controls.

## support seed43 extraVal5 / full8 confirmation
- Observation: same frozen operator tested on second seed and additional scenes.
- Hypothesis: local support gains survive seed and scene changes.
- Intervention: no parameter changes; seed43 extraVal5.
- Outcome: extraVal joint65.447→67.056%, visual harm110→107. Full8 seed-mean joint71.014→72.780%, all8scene positive; harm149.5→144.5.
- Remaining uncertainty: R.1 trade-off, reused development data, independent test not run.

## controls seed42
- Observation: full support joint73.031 vs fixed71.384 across8scenes.
- Hypothesis: spatial support rather than yaw smoothing alone supplies most benefit; aligned audio remains necessary.
- Intervention: XY-only, yaw-only, degree-normalized support, within-scene deranged audio at fixed parameters.
- Outcome: XY-only72.506, yaw-only71.074, degree-normalized72.721, shuffled support30.668%.
- Remaining uncertainty: seed43 controls and precision-preserving refinement running.

## H9 Train refinement
- Observation: support increased1m recall but lost0.1m precision.
- Hypothesis: basin decision and point decoding should be separated.
- Intervention: original fixed15 peak within fixed support neighborhood.
- Outcome: Train joint72.679→75.059%, R.1 26.444→27.180%, visual harm220→209.
- Remaining uncertainty: frozen rule now requires validation and second-seed replay.

## H9 final refinement confirmation
- Observation: support improved primary recall but reduced R@0.1m.
- Hypothesis: support should choose a region; the original fused peak should determine precision within it.
- Intervention: frozen local refinement, two upstream seeds, all eight validation scenes; no new parameter search.
- Outcome: joint71.014→72.601%, R1 72.566→74.224%, R.1 22.136→22.279%; all eight scene-mean joint deltas positive. Visual harms149.5→147.5 overall, with scene-level regressions retained in the report.
- Remaining uncertainty: reused development scenes, strict visual non-inferiority and independent test generalization.

## Final mechanism and independent inference checks
- Observation: final gains need an aligned-audio control and a deployable replay.
- Hypothesis: aligned acoustic evidence remains necessary, and the separated inference function reproduces research outputs.
- Intervention: within-scene acoustic derangement, visual-only final decoder, 320-query module replay and independent metrics for all8380 observations.
- Outcome: shuffled joint30.418%, final visual-only49.905%, final fused72.601%; all320 predictions exact; all8380 metrics, circular yaw and malformed-input checks passed.
- Remaining uncertainty: real-world sensing and furnished/unfurnished counterfactuals were not tested; calibrated probability claims are unsupported.

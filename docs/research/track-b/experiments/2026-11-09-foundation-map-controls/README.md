# Candidate-map evidence removal and yaw-correspondence controls

2026-11-09. Track B, Exploratory. Completed exit0: all nine joint-trained foundation checkpoints, five conditions, six reused Val scenes ×64 queries =17,280 cases. All3,456 normal cases match prior Joint/R@1m/Fine/raw20/NLL within declared tolerance; zero mismatches. Actual evaluation source `cd35096d650ee209fd483c835d7c0d193d41b65e`. No Test access, training or parameter selection.

## Hypothesis and intervention scope

The final full-field scorer should require candidate-dependent map evidence and correctly aligned yaw correspondence. If removing all candidate dependence still produced a structured field, that would expose another positional input, stale cache or implementation error. Incorrect yaw correspondence should shift orientation predictions while preserving the set of scores at each xy. These are dependency/frame controls, not evidence that attention learned a novel geometric mechanism.

Frozen RGB features, query RIR spectra, model weights, query labels and candidate pose coordinates are unchanged. Alter only the prepared FP cache: visual descriptors16, acoustic descriptors16, FP queries32 and reliability weights16. No GT enters the intervention function.

- Normal: unchanged cache.
- Candidate geometry removed: replace each of the four tensors by its mean across all candidates in that scene, broadcast to every candidate. Global scene information remains; all pose-dependent map information in this cached entrypoint is removed. This is not a physically plausible floorplan.
- Yaw90/yaw180: cyclically move all four tensors together by9/18 bins within each xy's36-bin yaw block. Candidate labels stay in their original order.
- Random yaw: independently choose a nonzero offset1..35 per xy, seed20261109, then cyclically move all four tensors together. Same offsets across queries/model seeds/modes; deterministic from bank grouping, with no GT. This is not randomizing microphone channels or changing calibration metadata.

Intervention code asserts36 ordered, equally spaced yaw cells per shared xy. It does not silently assume bank layout. Unit tests verify within-xy membership, descriptor multiset preservation, deterministic randomization, original-cache preservation and actual cached-scorer score permutation/near-uniform removal.39 tests passed before the added actual-scorer test; both map-control tests then passed. The additional test did not change the running inference code.

## Observation

Mean Joint≤1m/30° over three seeds,384 queries/seed:

| Condition | Attention % | Concat % | Separable % |
|---|---:|---:|---:|
| Normal |16.753|18.229|15.538|
| Candidate geometry removed |.174|.087|.087|
| Yaw90 |.521|.434|.608|
| Yaw180 |.781|.781|.694|
| Random yaw per xy |3.125|3.819|3.038|

Attention random-yaw minus normal: −13.628pp, six-scene95% paired bootstrap CI[−17.188,−11.198]. Yaw90: −16.233pp CI[−19.878,−13.194]; yaw180: −15.972pp CI[−19.965,−12.674]. Seeds are averaged within scene; seed/query records are not independent resampling units. These exploratory intervals are not multiplicity-corrected.

For **every one of10,368 yaw-intervened cases**, R@1m and Fine≤.1m are unchanged relative to the matched normal case. This is predicted by the within-xy score permutation: each xy retains its maximal score, so the best xy remains the same and only its yaw label changes (absent numerical ties). It prevents misreading the Joint decline as translation failure. Attention Fine remains.694%, concat1.302%, separable.955%. All per-query positional metric comparisons are recorded in position_invariance_audit.json.

Under geometry removal, the largest max−min score difference across any query is1.1444091796875e−5, consistent with floating-point roundoff on theoretically identical candidate evidence. Nearest-grid NLL is11.49622 for all three modes, approximately scene-averaged log(bank size). Residual MAP/raw20 outcomes are dominated by numerical tie-breaking and bank enumeration; **the .087–.174% values are not meaningful learned localization accuracy or a robust random baseline**. No exact-flat-field claim is made. Fine is0% for all modes in this condition.

Attention raw20:49.826% normal, .174% removed,4.253% yaw90,4.427% yaw180,28.819% randomyaw. NLL:9.21889,11.49622,13.05161,14.48386,12.72609 respectively. Complete R@1m/Fine/raw20/diverse20, strict(.5m,15°)/(.2m,10°), per-seed and paired metrics are retained in adjacent JSONs; original predictions remain external.

## Interpretation, alternatives and decision

**Supported:** candidate-map dependence is necessary for spatial variation in this cached scoring path; map-to-yaw correspondence matters for reported yaw. Normal replay excludes an obvious checkpoint/cache mismatch. All three architectures behave similarly under these destructive controls.

**Not supported:** these results do not isolate the learned FP tower from raw ray geometry, distinguish attention from ordinary matching, establish acoustic-specific geometric learning or show real floorplan-error robustness. The yaw intervention is an exact score reassignment by design; a large Joint drop is expected and cannot be claimed as a newly discovered causal mechanism. Removal simultaneously changes multiple cached representations and can be far outside training distribution. These caveats also explain why this control cannot overturn study21's negative interaction comparison.

**Decision:** retain as a passed dependency/frame check, not as positive novelty evidence. A narrower learned-FP-versus-raw-geometry ablation and matched update-boundary comparison would distinguish representation effects. Final selected regimes need their own audit; study24 frozen/partial fits remain running. Wider scale, independent evaluation, physical/material shifts and complete deployment cost remain unfinished.

## Reproduction and provenance

External root `/home/jeongeonlee/track1/experiments/track_b_interaction_20261109`; outputfoundation_map_controls_v1, registryfoundation_map_controls_registered_v1. Models come from completedfoundation_matched_v1 (training sources6285d4b/18c0e3b; full SHAs and checkpoints in study21). This evaluation does not change those models.45 condition/model/seed evaluations are registered in experiment_registry.csv with actual evaluation SHA and checkpoint hashes. External prediction/protocol hashes are preserved. Timing was not instrumented; registry marks evaluation GPU-hours NA, not zero cost. No end-to-end latency claim.

Dockeravfploc:cu124-clean, RTX4090, no network, non-root UID, code and structure/foundation input caches read-only. Evaluation shared the GPU with the already authorized frozen-regime training; no unrelated process was stopped. No performance-timing comparison is drawn from shared execution.

Executed `python -m track_b.scripts.evaluate_map_interventions --run /results/foundation_matched_v1 --cache /structure/cache --features /foundation/features_v1 --out /results/foundation_map_controls_v1 --code-sha cd35096d650ee209fd483c835d7c0d193d41b65e`. Registration source `3736eb4e95d08feda96f1ad4a54ceb333a5e3b05`: `python -m track_b.scripts.register_map_interventions --run <foundation_map_controls_v1> --training <foundation_matched_v1> --out <fresh-directory>`. Both exited0. Actual cached-scorer test source `4842e37eaefaa916cc5cf540ff9e57d23a81990f`; source trace is available in Git. Post-hoc positional audit compares r1/fine by(mode,seed,scene,query) against normal rows; it does not alter predictions or select outcomes.

Reports/indices on branchrefactor/hssd-two-branch-fusion; large tensors/logs remain external. Clean-clone reproduction requires the hashed external caches/weights. No failed run was overwritten. This experiment completes this defined control suite, not the overall research goal.

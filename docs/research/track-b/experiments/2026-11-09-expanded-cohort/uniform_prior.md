# Exact uniform-pose baseline on expanded Val19

2026-11-09. Completed; Exploratory. Question: how much full-bank localization success follows solely from candidate density and map geometry? The null is a uniformly sampled valid SE2 cell, independent of RGB/RIR. It is not a learned map-only spatial prior or a model selected by validation outcomes. GT is used only to evaluate correctness regions; it never changes the sampling distribution or bank.

## Results

19 scenes×64 queries=1,216 matched observations. The uniform distribution has exact expected Joint0.4313%, R@1m2.3879%, fine0.02539%, NLL=entropy11.30275nats. Twenty uniformly sampled distinct candidates contain a Joint-correct cell with probability8.1279% on average. This is a random20 baseline, not raw score-tie Top20 or diversity-NMS: those tie-breaking algorithms need not sample uniformly. No diversity claim is made.

The first scaling cell acoustic model has Joint4.2763%, exceeding this expectation by3.8451pp,95% paired scene bootstrap[3.1780,4.5852]. R@1m difference+6.2196pp[5.0559,7.4946]; fine+0.1391pp[0.0222,0.2803]; NLL−1.2468nats[-1.3717,-1.1209]. Model seeds are averaged inside each scene; the exact prior has no sampling-seed uncertainty. All model comparisons are archived in uniform_prior_paired.json. Secondary metrics are descriptive and uncorrected for multiplicity.

Bank oracle Joint is100%; oracle fine is50.8224%. A high Joint oracle does not establish useful learned accuracy, and half of the observations lack a fine-tolerance bank cell. This geometric limit must remain visible when judging fine results.

## Interpretation and limits

The acoustic model beats a weak uniform-pose null on reused development scenes. This alone does not establish physical acoustic reasoning, superiority to learned map-only or query-distribution priors, joint-interaction benefit, or independent generalization. Training/query sampling can favor nonuniform poses; removing those competing explanations requires other controls. Prior map-removal and shuffle studies remain separate evidence. Full factorial data-size/update comparisons are unfinished. Test is untouched.

## Implementation, validation and provenance

Each scene artifact was hashed against the structural manifest. Expected Joint is the count of correct cells divided by all valid candidates. Random20 uses1−C(N−K,20)/C(N,20), evaluated stably with logarithms; draws truncate at N for tiny banks. All495 combinations for N1…30 and K0…N match direct integer-combination probabilities to1e−12, including empty/full support. All1,216 query keys match every seed/method of the first cell. No random trials, optimization, checkpoint or baseline ablation fit is applicable to this analytical null. Individual query counts stay outside Git; aggregate values/hashes are archived.

Evaluation source `99b8cddfd70e279f6e1a011e62ba98bae2c4f633`; registration source `e8bbb00cdfe2d37cf06b9d426bc53cce9da73540`; both exit0. The manifest field was corrected from sha256 to cache_sha256 before the first execution; no failed run was discarded. Reproduce `python -m track_b.scripts.evaluate_uniform_prior --cache <expanded/cache_v1> --out <fresh-prior> --code-sha 99b8cddfd70e279f6e1a011e62ba98bae2c4f633`, then `python -m track_b.scripts.register_uniform_prior --prior <fresh-prior> --evaluation <n16_budget1_evaluation_v1> --out <fresh-registration>`. CPU Docker avfploc:cu124-clean,2 threads, read-only repository/no network. External outputs uniform_prior_v1 and uniform_prior_registered_v1 under the study scale root. No result-driven hyperparameter change follows this baseline.

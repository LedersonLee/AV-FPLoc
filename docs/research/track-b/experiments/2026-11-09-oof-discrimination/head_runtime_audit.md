# Cached-feature discriminator runtime audit

2026-11-09. Completed head-only verification of all12 existing OOF-trained discriminator fits,4,608 development inferences. No training or model selection occurred. This is not an M1 backend implementation, raw-input inference audit, candidate-feature provenance proof, or independent validation result.

## Question and observation

Does the saved backend reproduce its reported output metrics when only numeric candidate features enter inference? The audited function receives model, feature tensor, OOF normalization mean/std and the declared geometry/score-statistics arm. Scene/query identifiers, labels, GT pose, distances and yaw errors are outside its signature. Features are normalized using archived OOF training statistics; the score-only arm zeroes the same first128 normalized dimensions as training. It returns one candidate index per query.

All12 checkpoints match their original recorded hashes. For raw20/diverse20×geometry/score-only×3 seeds, all384 queries/fit match historical Joint, R@1m, fine, translation error and yaw error within1e−5: zero metric mismatches. Labels/metrics are consulted only after inference. No Python builtins.open call occurred inside the warmed head-inference boundary. Feature packs and historical output hashes are archived in head_runtime_validation.json.

These checks support faithful metric reproduction and a feature-only head interface on this cached development data. Historical selected indices/logits were not archived, so exact old-index or bitwise-logit equality is not asserted; equal errors could arise from distinct symmetric candidates. The audit does not prove stored features lack upstream leakage: source candidate_features and existing OOF/feature tests provide separate, limited evidence. Native I/O is not traced by the Python hook. The classifier's model weights and normalization are explicitly required preloaded state, not raw-input-free deployment. No GT injection was added, but proposal provenance is not newly re-executed here.

## Scope and reproducibility

Existing16 Train/6 Val,64 queries/scene,3 seeds; OOF training provenance and half-Train proposal/full-Train development mismatch remain as reported in completed_results.md. All reused-Val/upstream exposure caveats remain. No quantitative accuracy improvement, causal mechanism or cross-dataset generalization follows from this audit. Baseline and ablation outcomes are unchanged; inferential uncertainty is not applicable to zero replay mismatches beyond the evaluated cohort.

Executed source `30b735c32cacc432e9a96e5489e5c39643f59d95`, exit0; CPU Docker avfploc:cu124-clean,2 threads, repository read-only/network disabled. Reproduce `python -m track_b.scripts.audit_discriminator_runtime --root <oof-root> --out <fresh-directory> --code-sha 30b735c32cacc432e9a96e5489e5c39643f59d95`. Actual external output: /home/jeongeonlee/track1/experiments/track_b_oof_20261109/head_runtime_audit_v1. Weights/features/raw observations remain outside Git. No failed invocation in this audit; no existing artifacts overwritten. Full M1 proposal/discriminator integration and broader end-to-end runtime checks remain required work.

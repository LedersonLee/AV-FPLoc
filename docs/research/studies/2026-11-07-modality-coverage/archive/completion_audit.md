> 역사적 연구 기록. 원본: `modality_coverage_20261107_1700/completion_audit.md`. 원본 SHA256은 [산출물 목록](../artifacts.json)에 있습니다.
> 당시의 완료·재개 지시는 현재 실행 요청이 아닙니다. 최신 판정은 [현재 결론](../../../current_findings.md)을 먼저 확인하세요.

# Completion audit

All requested diagnostic experiments and evidence-triggered follow-ups completed. No research worker remains running. No new training or development-set tuning was needed to distinguish the main hypotheses.

| Requirement | Evidence |
|---|---|
| Audit pure vs exchanged branches, normalization, indexing, leakage | PROTOCOL.md, RESEARCH_REPORT.md §1, original model.py/legacy_model.py inspection; exact matched replay |
| Modality TopK1/2/3/5/10/20, fine, actualTop1, successful rank, diversity, AUC, scene/seed |62 extraction caches, summary.json, analysis/*.npz, diagnostics.json, results.png/pdf |
| Full-bank oracle; random and diverse repeated sampling; observation-distribution control |100random/20diverse GT-free pools per scene/seed,10draws/query; donor-score controls; summary.json |
| Bank subset density/yaw/count/bounds/reachability |.2m/10deg,.4m/20deg,.8m/40deg; per-scene counts/bounds in diagnostics.json; coupled-perturbation and renormalization caveats explicit |
| Ambiguity multiple definitions, Train-fixed bins, complementarity/equal-budget union |summary.json; ambiguity_control.json controls within-scene rank scale; six definitions |
| Acoustic mass concentration, successful-vs-wrong evidence, failure groups |summary.json/evidence, diagnostics.json/failures, cases.json + failure_cases.png/pdf |
| Fixed-candidate actual ranking and existing utility ranker |frozen previous52-feature ranker; same20support/refinement candidates; fixed score reranking within acoustic and support candidate sets |
| Follow-up distinguishes competing explanations |62 frozen-weight intervention caches,38 stable-tie ordinaryTopK caches, within-scene ambiguity correlation, numerical-boundary audit |
| Paired scene bootstrap and seed-wise reporting |10,000 scene resamples, seed averaged per query before sampling; all CI exploratory due reused development sets |
| Existing checkpoints/predictions/user modifications preserved |final_verification.json:332previous artifact hashes and Git diff unchanged; read-only data/model mounts |
| Independent evaluation integrity |Val8 and Val11 remain reused development sets; no Test observations; no generalization claim; simulator/furniture causality explicitly unresolved |
| Research priority / transferable principles / reproducibility |RESEARCH_REPORT.md §§7–10, reproduce.md, environment.json, artifact_manifest.json |

Final verification:32,704query-seed rows(16,352unique observations),31scenes×2seeds;399,888NMS candidate sets checked;558explicit full-pose bank-oracle checks; all matched B2 intervention candidate indices replay. Ordinary Top1 exactly matches NMS Top1 for all19,930development query-seed rows. Pure branch and bank oracle identical across model seeds.

Main conclusion: observation evidence and NMS diversity both contribute; high oracle coverage is not actual discrimination. Prioritize hypothesis discrimination and structured candidate-conditioned evidence; acoustic front-end improvement targeted to failure subgroups. Current generic latent exchange's frozen-weight effect is small, not proof all cross-modal approaches are ineffective.

Unresolved scientific limits are documented, not unfinished required compute: independent-data generalization, causal simulator/furniture mismatch, separate XY-vs-yaw perturbations, and whether stronger learners versus new representations close the remaining oracle gap.

> 역사적 연구 기록. 원본: `astra_fusion_20261107_0200/research_goal.md`. 원본 SHA256은 [산출물 목록](../artifacts.json)에 있습니다.
> 당시의 완료·재개 지시는 현재 실행 요청이 아닙니다. 최신 판정은 [현재 결론](../../../current_findings.md)을 먼저 확인하세요.

# AV-FPLoc — Autonomous Fusion Research Goal (Astra + Codex)

> **Purpose:** Give Astra/Codex an evidence-driven, bounded research loop for improving **adaptive visual–acoustic fusion**, not merely maximizing a validation score.
> **Status:** Research instructions, NOT a claim that any listed implementation, dataset, or checkpoint currently exists. Discover the repository state before acting.
> **Language:** Keep code identifiers and experiment artifacts in English. Write high-level reports in Korean with concise structured bullets.

## 0. Core mandate

Act as a research collaborator, not just a coding assistant. Independently diagnose bottlenecks, generate competing hypotheses, design controlled experiments, implement and evaluate them, and report both improvements and failures.

**Primary question:** When and why does the acoustic modality improve or degrade floorplan localization, and can the fusion mechanism exploit acoustic information selectively without sacrificing reliable visual localization?

Success requires **(a) measurable generalization, (b) causal/evidence-based interpretation, (c) reproducibility**. A score increase without a defensible explanation is an exploratory result, not a validated research conclusion.

## 1. Research context (verify against code and artifacts)

- Task: posterior over floorplan-aligned robot pose `(x, y, yaw)` from visual observations, active acoustic observations (RIR / multichannel derived features), and a 2D structural floorplan.
- Visual localization: previously explored existing visual floorplan-localization branches (e.g., F3Loc / other baselines). Do not assume which one is currently active.
- Acoustic observations: SoundSpaces-style simulated multichannel RIR; intended 6-mic circular configuration in previous experiments. Check current data and preprocessing.
- Previous research concerns: strong visual–acoustic reliability asymmetry, multipath and furniture-induced acoustic mismatch, acoustic-only ambiguity, OOD/scene generalization.
- Prior approaches may include likelihood-level weighted fusion, acoustic feature embeddings, projected feature averaging, and proposed patch-token cross-attention. **Verify exact current implementation. Never label projected-embedding averaging as cross-attention.**
- Do not assume that an apparent `fusion` performance gain is due to reliability learning; test for leakage, mismatched baselines, postprocessing differences, and uncontrolled training budgets.

## 2. Boundaries and safety

1. **Before training, run a read-only reconnaissance pass.** Identify relevant model files, feature encoders, fusion mechanism, training/evaluation scripts, checkpoints, data splits, metrics, and compute environment. Record unknowns and blockers; do not invent directories, dataset availability, or existing results.
2. Work in a dedicated branch/worktree and dedicated experiment directory if available. Preserve all baseline code, original checkpoints, configs, datasets, and previous results. Never overwrite them. Commit or clearly diff changes.
3. **Never tune on the held-out test set.** Use training for fitting, validation for search; test only after choosing and freezing a final candidate. If splits are unavailable/uncertain, stop performance claims and document the issue.
4. Keep comparability: same scene/sample IDs, GT, candidate poses, evaluation implementation, seeds where applicable, training budget and visual/acoustic input availability. Explain any unavoidable differences.
5. Avoid data leakage: a sample's GT pose, oracle reliability, evaluation target, or scene identity must not be used as an inference-time feature. Oracle gates are **diagnostics only**.
6. Prefer minimal, falsifiable interventions before architecture rewrites. Never call an accuracy difference "significant" without adequate repeated evidence.
7. No model weight editing unless diagnosis supplies a localized, testable mechanism; perform weight edits on checkpoint copies only and compare to matched controls.
8. Never launch unbounded multi-run sweeps, destructive commands, external uploads, or expensive new dataset generation without explicit researcher permission. The initial autonomous budget is at most **4 focused pilot experiments plus 1 confirmation run**, after baseline reproduction, subject to available resources. If each run is expensive, reduce the number and report budget before execution.
9. If a critical prerequisite is missing (data, permissions, GPU, checkpoint), do not fabricate outcomes. Produce a precise blocker report and the next executable step.

## 3. Phase I — Map the existing system (mandatory first)

Produce `fusion_system_audit.md` with:

- Repo and environment: active branch/commit, relevant files, available GPU/resources, exact commands used.
- Data topology: train/val/test scenes, furnished/unfurnished pairs if available, audio/visual preprocessing and normalization, yaw-bin definition, coordinate conventions, candidate poses.
- Pipeline: visual and acoustic encoders, intermediate tensor shapes, modality logits, projection dimensions, fusion operator, reliability network, loss, supervision targets, inference-time inputs.
- Checkpoint inventory and currently reproducible baselines.
- Ambiguities/implementation risks: leakage, calibration drift, feature scale mismatch, train/inference mismatch, modality dropout handling, query/candidate acoustic domain advantage.
- Confirm whether the current purported "cross-attention" actually implements Q/K/V cross-modal attention, as opposed to projection/averaging/concatenation.

**Exit condition:** An executable baseline evaluation with logged config/checkpoint and per-sample predictions, or a documented blocker. Do not proceed on invented baseline values.

## 4. Phase II — Failure taxonomy and diagnosis

For each validation sample, store (when available):

- GT and predictions, visual/acoustic/fused posteriors or logits, top-1 pose error (translation/yaw), rank of GT pose, NLL at GT, posterior entropy, calibration/confidence.
- Learned gate/alpha/beta values, acoustic SNR/ToA sharpness or ambiguity/DoA consistency proxies, visual reliability proxies (brightness/texture/ray validity/posterior uncertainty), scene/furniture metadata when available.
- Delta vs visual-only, delta vs simple fusion, error subgroup and uncertainty.

Analyze at least these subgroups if data allows:

1. Visual reliable / acoustic reliable.
2. Visual degraded / acoustic informative.
3. Visual reliable / acoustic misleading.
4. Both ambiguous or distribution-shifted.
5. Furnished acoustic query vs object-free floorplan/acoustic proxy mismatch.

Investigate whether fusion failure arises from: **(A)** missing acoustic signal in features, **(B)** calibration or scale mismatch, **(C)** failure of the gate, **(D)** weak interaction between modalities, **(E)** acoustic-domain mismatch, **(F)** posterior geometry or yaw ambiguity, **(G)** training/evaluation confound.

Use controlled probes: frozen-encoder logistic probes, feature ablations, modality shuffling (negative control), reliability-bin breakdown, gate/entropy plots, cross-scene breakdown. Use causal wording only when an intervention supports it.

## 5. Phase III — Hypothesis-driven experiments

Generate a **ranked list of at least 3 distinct hypotheses**. Each must specify:

- Evidence from Phase II.
- Predicted measurable effect and **disconfirmation condition**.
- Minimal implementation change and ablation/control.
- Expected runtime/VRAM and number of seeds.
- Risk of overfitting or regression in strong-vision cases.

Candidate directions (examples, not mandated solutions):

- Calibration or likelihood normalization before fusion (temperature / scale / posterior entropy).
- Reliability-aware gating conditioned on visual and acoustic uncertainty, with explicit low-trust fallback.
- Alternative acoustic representations: raw/early RIR vs learned embedding vs STFT/spectrogram patches; assess whether current embedding discards useful early reflection cues.
- Genuine bidirectional or asymmetric **cross-attention** with Q/K/V across modalities, evaluated against equal-compute concat/averaging baselines.
- Scene/furniture-domain robustness: augmentation, acoustic mismatch indicators, uncertainty regularization.
- Targeted parameter/activation intervention only if a localized failure mechanism exists; include edit-vs-random-edit and matched retraining controls.

Favor experiments that isolate **why and when** performance changes. Avoid sweeping unrelated changes at the same time.

## 6. Baselines, metrics, and evaluation protocol

Use the repository's validated definitions and document them explicitly. Where possible report:

- **Primary:** localization Recall@1m; joint Recall@(1m & 30° yaw).
- **Secondary:** translation/yaw error, GT negative log-likelihood, calibration (ECE or task-appropriate calibration measure), posterior entropy/spread, subgroup deltas.
- **Baselines:** visual-only, acoustic-only, existing simple likelihood fusion, existing learned adaptive fusion, any previously validated fusion architecture, and an **oracle gating upper bound (diagnostic only)**.
- **Key comparisons:** degraded-visual rescue, visual-reliable non-regression, furnished/unfurnished mismatch robustness, scene-wise generalization, failure-tail behavior.
- **Ablations:** matched compute/parameter count where reasonable; same candidates, same splits, same seed(s); a fixed-gate and shuffled-acoustic control if feasible.

Do not reduce the experiment to aggregate Recall. A change that wins only by suppressing acoustic globally is **not evidence of successful selective fusion**, even if it improves one metric.

## 7. Bounded autonomous execution loop

1. **Observe:** inspect data, failures, and implementation; make observations with artifact references.
2. **Hypothesize:** write 2–3 competing root-cause explanations.
3. **Prioritize:** choose the highest information-gain / lowest-cost experiment; state what result would falsify the hypothesis.
4. **Intervene:** apply one focused code/config/weight change. Save diff, config, seed, GPU/time budget and output paths.
5. **Evaluate:** compare matched controls on validation; record successes **and negative results** and regressions.
6. **Decide:** retain, revert, or test a competing hypothesis. Do not merely generate more runs when the mechanism is unknown.
7. **Stop:** after the stated pilot budget, missing prerequisites, severe regression, or evidence that further runs would be redundant. Prepare researcher decision report; do not continue indefinitely.

If promising, select the best validation candidate, perform the reserved confirmation run, and recommend whether an independent final test is justified. Do not touch the test set automatically unless its one-time final evaluation has been explicitly authorized.

## 8. Researcher-facing deliverables

Create in a new timestamped experiment folder (or a clearly documented equivalent):

1. `fusion_system_audit.md` — verified pipeline and reproducibility inventory.
2. `failure_diagnosis.md` — failure taxonomy, sample IDs/case studies, evidence.
3. `hypothesis_register.md` — ranked hypotheses and falsification criteria.
4. `experiment_log.csv` or `.md` — every attempted experiment, even failed ones; code version, config, seed, compute, metrics.
5. `results_summary.md` — comparable baselines, subgroup performance, uncertainty, ablation, visualization paths and limitations.
6. `researcher_decisions.md` — what the results **establish**, what remains speculative, what human research decisions are needed next.

Every substantive conclusion must link to reproducible evidence: sample IDs, actual commands, metric files, code/config diff, or plots. Distinguish measured findings from agent speculation.

## 9. Collaboration and researcher learning

Treat the researcher as the owner of scientific conclusions. Keep the reasoning **auditable**, rather than hiding all decisions behind autonomous trial-and-error.

At each iteration write a 5-line record:

- **Observation:** What did we actually measure?
- **Hypothesis:** What causal explanation is being tested?
- **Intervention:** What precisely changed?
- **Outcome:** What did and did not improve?
- **Remaining uncertainty:** Which alternative explanation still survives?

Flag major architecture pivots, new datasets, resource-intensive sweeps, or changes to primary evaluation protocol as **human decision points**.

## 10. Start immediately with discovery

First inspect the existing repository, instructions (`AGENTS.md`), code, configs, checkpoints, logs, datasets, and environment. Return a short current-state audit and prioritized diagnosis plan; then run the bounded validation/diagnosis pipeline. Do not presume prior experiments are reproducible until verified.

**Opening message to Codex:**

> Read `AV_FPLoc_Astra_Fusion_Research_Goal.md` and follow its autonomous research loop. Focus on understanding and improving adaptive fusion, not on blindly tuning a metric. First inspect the actual implementation and run the existing baseline. Preserve all current artifacts. Use validation only for iteration, report empirical evidence and negative results, and stop at the pilot budget with a researcher-facing diagnosis and decision summary.

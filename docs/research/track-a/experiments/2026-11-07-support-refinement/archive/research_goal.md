> 역사적 연구 기록. 원본: `astra_fusion_v2_20261107_0225/research_goal.md`. 원본 SHA256은 [산출물 목록](../artifacts.json)에 있습니다.
> 당시의 완료·재개 지시는 현재 실행 요청이 아닙니다. 최신 판정은 [현재 결론](../../../../current_findings.md)을 먼저 확인하세요.

# AV-FPLoc — Persistent Autonomous Fusion Research Goal v2 (Astra + Codex)

> **Purpose:** Give Astra/Codex a persistent, evidence-driven research loop that keeps diagnosing and improving **adaptive visual–acoustic fusion**, rather than ending after a brief audit or a preset number of pilot experiments.
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
8. Do not launch destructive commands, external uploads, paid resources, or expensive new dataset generation without explicit researcher permission. **The former limit of 4 pilot experiments plus 1 confirmation run is REMOVED.** Keep iterating in small, controlled experiment batches using resources already authorized. A batch is a planning unit, **not a stopping condition**. Before additional compute-intensive experiments, check the available authorized compute/time budget. If no budget is known, continue cheap diagnostics and report the proposed cost before any substantial commitment. Never silently assume unlimited authorization.
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

## 7. Persistent autonomous execution loop (mandatory)

**Non-negotiable behavioral requirement:** The task is **NOT finished** when a system audit, failure diagnosis, report, or the first experiment batch has been completed. Treat these as checkpoints. Keep attempting evidence-driven improvements as long as prerequisites and the authorized resource budget permit. Do not substitute writing a report for doing the next experiment.

Repeat this cycle:

1. **Observe:** Reproduce the strongest fair baseline and inspect the poorest or most surprising validation subgroups; record raw artifacts and sample IDs.
2. **Hypothesize:** Generate at least 2 competing, falsifiable explanations, including at least one explanation that could invalidate the current architecture/reliability assumptions.
3. **Select:** Prioritize the highest information-gain, lowest-cost discrimination experiment. State what outcome would falsify each hypothesis.
4. **Intervene:** Implement exactly one interpretable change, save code diff/config/seed/checkpoint, and run a matched comparison.
5. **Evaluate:** Compare the fixed validation split against visual-only, acoustic-only, and current best fusion; assess recall, pose/yaw, calibration, and subgroup regression. If aggregate gain reflects disabling audio, report it as a failure to validate selective fusion.
6. **Update:** Keep or revert the change. Update the hypothesis register, failure taxonomy, evidence, and next experiment.
7. **Continue:** If the run fails or is inconclusive, analyze why, select a DIFFERENT hypothesis or more diagnostic experiment, and continue. No fixed experiment count automatically completes this task. If three consecutive experiments are non-informative, perform a methodological pivot or data/label integrity audit rather than repeating the same idea.
8. **Confirm:** When a candidate improvement is found, reproduce it with matched settings and independent seeds or equivalent robust checks. Verify at least one informative subgroup and check non-regression in visual-reliable cases. Keep the one-time held-out test set sealed until explicit authorization.
9. **Report and decide:** Distinguish verified improvement, promising but unconfirmed signals, and diagnosed bottlenecks. Continue from the next experiment if the objective remains unmet and authorized resources remain.

### 7.1 What counts as success?

- **Primary success:** A reproducible and fair improvement over the current strongest baseline on **validation**, with a plausible explanation supported by controlled evidence, and no unacceptable degradation in critical subgroups. Prefer meaningful gains, not noise-level changes; show uncertainties and per-scene effects.
- **Secondary research success:** A concrete, experimentally supported bottleneck and an actionable intervention that remains to be validated. This counts as progress but **not permission to declare performance improvement or to stop early** when further trials are feasible.
- **Do NOT claim success** from a single favorable seed, inconsistent metrics, leakage, tuning against the test set, weak baselines, suppressing audio across all cases, or an untested causal story.

### 7.2 Legitimate stop / pause conditions

- The requested improvement has been reproduced and documented; report whether further search is still promising.
- All *authorized* compute, time, or execution quota is exhausted. Report exact remaining hypotheses and the concrete command for resumption; ask for extended authorization rather than spinning indefinitely.
- A missing dataset, checkpoint, permission, valid split, or compute resource prevents meaningful progress after checking the repo and reasonable alternatives.
- The remaining experiments require a new resource-expensive decision, paid service, substantial dataset generation, or modification of the evaluation protocol that the researcher has not approved.
- Multiple independent approaches have been tested and further experimentation is demonstrably redundant; substantiate this with the experiment register and evidence rather than a generic statement.

**Do not stop just because an audit, summary document, one pilot batch, or one architecture variant is complete.** If the execution/session itself is interrupted, prepare a machine-readable resume record and explicitly ask to resume. A prompt cannot guarantee infinite execution or guaranteed improvement: platform, permissions, and available compute still impose limits.

### 7.3 Required continuity artifacts

Keep `next_action.md` up to date after EVERY iteration with:
- Best reproducible baseline, best candidate, primary metrics and subgroup regressions.
- Hypotheses attempted / refuted / still open.
- Exact next experiment, command, expected cost, falsification condition.
- Code/config commit or diff, checkpoint paths, run IDs, environment, and required permissions.
- Why the process is still running or the exact legitimate reason it paused.

When resuming, read `next_action.md` and the experiment register first. Never start diagnosis from zero if valid logs already exist.

## 8. Researcher-facing deliverables

Create in a new timestamped experiment folder (or a clearly documented equivalent):

1. `fusion_system_audit.md` — verified pipeline and reproducibility inventory.
2. `failure_diagnosis.md` — failure taxonomy, sample IDs/case studies, evidence.
3. `hypothesis_register.md` — ranked hypotheses and falsification criteria.
4. `experiment_log.csv` or `.md` — every attempted experiment, even failed ones; code version, config, seed, compute, metrics.
5. `results_summary.md` — comparable baselines, subgroup performance, uncertainty, ablation, visualization paths and limitations.
6. `researcher_decisions.md` — what the results **establish**, what remains speculative, what human research decisions are needed next.
7. `next_action.md` — continuously updated resumable state with the exact next planned experiment and permission/budget status.

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

First inspect the existing repository, instructions (`AGENTS.md`), code, configs, checkpoints, logs, datasets, and environment, and read prior experiment logs to avoid repeating work. Run baseline verification, failure diagnosis, and repeated hypothesis-driven interventions. **Do not treat the initial audit, a single experiment batch, or writing a results report as the task being done.** Continue within authorized resources until a reproducible improvement is found or a legitimate stop/pause condition in §7.2 is reached.

**Opening message to Codex:**

> Read `AV_FPLoc_Astra_Fusion_Research_Goal_v2_Persistent.md`. First read all prior outputs, especially `next_action.md`, the experiment log, and the previous Codex report. Resume the work; do not restart from scratch. Do not consider the first audit, the first pilot batch, or a summary report a completion criterion. Iteratively diagnose failures, test competing hypotheses, implement controlled improvements, and repeat within already-authorized compute until you obtain a reproduced meaningful validation improvement or encounter an explicitly documented stop/pause condition in §7.2. Preserve code/checkpoints; do not access the test split for tuning. Update `next_action.md` after each run and provide a precise resumption command if interrupted.

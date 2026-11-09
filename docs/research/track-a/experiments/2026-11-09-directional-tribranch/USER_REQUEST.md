# AV-FPLoc — Autonomous Research Plan
## Directional Acoustic Structural Evidence → Floorplan-conditioned Tri-branch Joint Learning

**Target:** IJCAI 2027 research exploration  
**Priority:** A short, falsifiable acoustic-structure feasibility study, followed **regardless of A's outcome** by a controlled tri-branch pilot.  
**Research principle:** Discover which information improves floorplan pose-hypothesis discrimination, not which architecture is most complex.

---

## 1. Research questions and decision policy

Investigate two distinct but complementary hypotheses.

- **A — Directional acoustic structural evidence:** Can the 6-channel active Room Impulse Response (RIR) encode coarse *body-frame directional openness, relative boundary distance, or spatial extent*, sufficiently reliably to separate visually ambiguous pose hypotheses? Exact wall identification, calibrated Direction of Arrival (DoA), or metric depth is **not assumed**.
- **B — Floorplan (FP)-conditioned joint representation:** Can visual observations (V), acoustic observations (A), and candidate-conditioned FP structure be encoded and *jointly optimized* to produce better SE(2) pose discrimination than the strongest existing score fusion and GEOQ (geometry-conditioned utility ranker)? Is explicit cross-modal interaction necessary, or is an equally informed concatenation/conditional scoring model sufficient?

**Hard workflow rule:** Complete A's bounded feasibility and failure analysis, then execute B's pilot **even if A is negative**. If A succeeds, test its structural representation as one B input; if it fails, B uses the established acoustic embeddings/features. Do not turn A into an unlimited acoustic-depth project or make B contingent on an unproven physical cue.

Preserve existing learned models and prior findings. Critically test the researcher's hypotheses and independent alternatives; do not force cross-attention, Spherical Harmonics, or acoustic depth to succeed.

## 2. Existing evidence and contracts to audit before running

Audit code, hashes, datasets, metric definitions, and actual feature shapes; the points below are starting context, not substitutes for an implementation check.

- Existing strong system: UnLoc-based visual branch + contrastive acoustic RIR encoder + B2 score/latent processing → per-pose standardized fixed fusion (visual 0.85, acoustic 0.15) → local support / Support Non-Maximum Suppression (NMS) / local refinement → GEOQ ranking.
- GEOQ: Geometry-conditioned candidate **utility-gain** ranking, **not** a calibrated posterior. The selected GEOQ input was 156 dimensions (score/local statistics S26, candidate geometry G26, observed geometry Q26, and 78 baseline-relative features). If code provenance shows a variant, record it.
- HSSD latest reported 11-scene / 6,072-query head-held evaluation: fixed fusion 66.250%, B2 69.667%, support/refinement 71.781%, GEOQ 73.968% at joint 1 m/30°. E0+GEOQ 74.313% was exploratory and its incremental interval included zero. The visual teacher had exposure to these scenes: not fully independent full-pipeline generalization.
- Previous high coverage: B2 Support Top-20 oracle around 97%, but this is **GT-included coverage**, not achievable Top-1 performance or proof that a specific representation identifies the correct candidate.
- Previous angular pilot: 6 horizontal microphones on a ~5 cm radius ring; shared azimuth transform checked, but acoustic steering had broad response (~90° half-height in the synthetic test), rank limitations, and nontrivial information loss; do not interpret 10° yaw bins as 10° DoA resolution. The learned E/C/N CNN token indices are **not** physical direction bins.
- Previous frozen feature exchange, angular/Fourier, concat, and small visual adapter experiments did not demonstrate improvement over strong GEOQ. They do **not** test full joint optimization of both base encoders.
- Visual rays: approximately 11 horizontal rays over −50° to +50°; acoustic 6-channel active RIR at 48 kHz. Verify sensor and coordinate-frame contracts, including front/left/up, yaw rotation, source centered over the ring, 2D FP extrusion geometry and acoustic reference generation.
- HSSD: 135 scenes reportedly had RGB/RIR/FP input samples; dense candidate banks were usable for roughly 54 approved scenes, with earlier training using 12–24 scenes. Recheck actual readiness and source-family splits; do not assume all 135 have a usable full bank.

### Deliverable 0
A short `AUDIT.md`: exact repository paths, baselines/checkpoints, input/label contracts, data exposure, known limitations, stage A/B computational estimates. No blind regeneration of existing RIRs.

## 3. Phase A — Directional Acoustic Structure: low-cost observability test

### A1. Prediction targets from FP geometry (training/evaluation only)

For an observed query at known training pose/yaw, derive *body-frame* geometric targets from the approved structural FP using the existing ray caster:

- **A1-minimal:** Front / left / back / right coarse *near vs. far* or *closed vs. open* labels and pairwise ordering of relative ray depth.
- **A1-optional:** 8 broad azimuth sectors and uncertainty/unknown mask; optional ordinal distance bins. Avoid claiming exact continuous metric depth is feasible a priori.
- Distinguish physical wall, doorway, outside, and unknown/unobserved footprint where masks allow it. A black FP pixel is not automatically evidence of a wall. Document maximum ray range and floorplan height assumptions.
- Fix sectors, bins, range thresholds, GT-free feature selection, and evaluation groups **on Train**; report class balance and blind/random priors.

**Critical leakage control:** The A predictor takes **observed acoustic features only** at inference (possibly specified sensor geometry); neither current GT pose nor the candidate's oracle FP depth may enter the acoustic predictor. FP ray targets are for training labels and diagnostics. At B time, the candidate's *hypothesized* FP geometry may be supplied to the candidate scorer, but never the FP geometry extracted at the unknown true query pose.

### A2. Compare physically meaningful probes, not just large models

- No-learning or simple interpretable baseline: early-window energy, time/envelope T/N, complex inter-channel coherency C, broad temporal/sector features, and acoustic score/embedding linear/MLP probes.
- Probe 1: linear/regularized model for coarse direction/ordinal target.
- Probe 2: small MLP with identical acoustic inputs and constrained trainable capacity.
- Optional: modest directional-consistency objective only if measurable signal exists. Do not rebuild the full RIR model or insist on a full 360° metric acoustic depth map.
- Include **no-audio prior, shuffled/donor observation, channel-order or yaw-rotation controls, and capacity-matched nuisance features**. Controls should preserve physically meaningful channel synchronization when needed; corruption-induced out-of-distribution loss is sensitivity, not proof of a physical cue.

### A3. Verify actual discrimination relevance

Report separately:

1. Direction/sector target accuracy, ordinal agreement, per-sector reliability and confidence (vs. data-driven priors).
2. Pose- and scene-held-out generalization; repeated-render stability where receipts allow.
3. In *visually ambiguous pairs* defined GT-free in Train, whether predicted directional structure more often favors the GT-success hypothesis than GT-failure hypothesis, with candidate FP rays held fixed.
4. **Conditioned benefit:** with the exact same Top-20 candidates, does directional structure provide incremental pairwise or Top-1 discrimination **given existing acoustic score + visual score + GEOQ features**? Compare against predicted-label donor and matched-capacity inputs.
5. Physical failure modes: multipath, furnishings/proxy-RIR mismatch, broad azimuth response, unknown FP cells, out-of-FoV visual geometry, floor/ceiling contributions.

A successful geometry-label probe alone is not evidence of localization contribution, and a candidate FP ray oracle by itself is not a deployable audio method. Keep metrics labeled accordingly.

### A4. Phase A decision receipt

Write `A_FEASIBILITY.md` with a clear **positive / mixed / negative / inconclusive** judgment, counter-hypotheses and smallest useful next probe. **Proceed to Phase B regardless.** Use coarse directional features in B only if they pass Train-held-scene and matched/donor relevance checks; otherwise use the established acoustic query/reference embeddings.

## 4. Phase B — FP-conditioned Tri-branch joint-learning pilot

### B1. A physically meaningful interface

Three input branches:

- **V encoder:** RGB → visual intermediate tokens/ray-depth evidence, retaining the visual partial field of view (FoV) and missing-direction mask.
- **A encoder:** six-channel RIR E/C/N and/or current acoustic embeddings. If A yielded a valid coarse structural cue, include it in *a separate controlled arm*, not as a mandatory assumption.
- **FP encoder:** candidate pose-dependent structural ray distances/local map geometry (and optional FP acoustic reference embedding). FP is a *known reference map*, **not a third observed sensor**.

Define a scorer for each candidate pose `p` such as `s_theta(p | V_obs, A_obs, FP[p])`. Consider one lightweight FP-candidate query attending to V/A evidence, or a candidate-conditioned gated MLP, before implementing three pairwise cross-attentions. Specify exactly what each token contains and what space/time direction it represents. Compare to equally informed non-attention models.

The scorer outputs **candidate logits/evidence**. Do not call them calibrated likelihoods or posteriors until normalized and independently calibrated. Candidate FP inputs must be generated for each hypothesized pose, never from GT at inference.

### B2. Controlled architecture and training matrix

Start with the *same frozen V/A encoders and candidate bank*, matched data and an identical candidate-ranking objective:

| ID | Variant | What it isolates |
|---|---|---|
| B0 | Existing B2 + GEOQ | Strong full-system reference |
| B1 | Fixed standardized score fusion + support/refinement | Basic score-fusion reference |
| B2a | Existing V/A + FP candidate features → concatenation / MLP | Same-information, non-attention baseline |
| B2b | FP-conditioned candidate scoring / gating | FP conditioning beyond static geometry ranker |
| B2c | FP-query-to-V/A cross-attention or relational exchange | Explicit interaction beyond concatenation |
| B2d | B2b or B2c + validated directional acoustic cue (if available) | Incremental physical cue contribution |
| B2e | No-audio and no-FP ablations with matched capacity | Dependency on true modalities/reference |

Keep parameter count, *active* trainable capacity, epochs/steps, scene weights, feature sets and query budget as close as practical; record any difference. Avoid an obviously weakened new matched base as the only reference. Fix Top-20 proposals from B2 for the first comparison to isolate discrimination from candidate coverage.

### B3. Staged encoder unfreezing — required causal comparison

Once the frozen baseline is correct, evaluate **small targeted fine-tuning** rather than a sudden full retrain:

- **F0:** Frozen V and A; learn FP encoder + fusion scorer only.
- **F1:** Same scorer + partial Acoustic encoder fine-tuning; compare acoustic encoder fine-tuning **without** cross-modal interaction.
- **F2:** Same scorer + small trainable Visual adapter and partial A; compare a same-capacity no-audio Visual adapter and donor-audio adapter.
- **F3 (conditional):** Wider partial/full V–A joint fine-tuning only when F1/F2 has a credible train-held-scene signal and compute risk is acceptable.

For each stage keep the matched-control initialization, updates and supervision comparable, and analyze whether the gain comes from trainability, FP conditioning, or observation-specific V↔A exchange. Existing UnLoc depth/ray quality and pure visual localization must not be silently degraded.

### B4. Candidate reasoning vs. end-to-end

- **Fixed-candidate first:** Same Support-NMS Top-20 candidates, same query/candidate IDs, same joint success and fine metrics; record Top-1 conditional on candidate-oracle coverage, visually ambiguous pair AUC, repair/harm, visual-success preservation.
- **Full bank after promising evidence:** Apply the learned scorer to the complete FP candidate bank using cached query/FP embeddings or two-stage coarse-to-fine inference; measure candidate generation coverage and computational cost separately. No GT filter in deployment.
- **Decision control:** Compare naive MAP, fixed support/refinement and frozen GEOQ where interfaces permit. If new scores require adapting GEOQ, perform a matched retrain and explicitly separate scorer and back-end changes.
- **Fusion controls:** Matched acoustic vs multiple donor-acoustic observations, no-audio trained control, interaction-disabled, same-feature concatenation and scalar score fusion. Donor changes must be physically paired to recipient FP candidate references, not only permuted output slots.
- Test optional V→A→V sequential interaction **only as an independent extension** if A→V has evidence, and compare the same budget of single-pass and bidirectional alternatives.

## 5. Training goals and metrics

Use existing strict metrics consistently (position ≤1m and circular yaw ≤30°, R@1m, strict R@0.1m) plus stage-specific diagnostics:

- **A:** coarse directional label accuracy / ordinal error / pairwise GT-success discrimination / repeated rendering and donor gap.
- **V/A interaction:** audio-conditioned visual Top-1 and ambiguous-pair accuracy on fixed candidates; matched–donor improvement; extra gain over same-feature concat/score fusion.
- **Hypothesis extraction:** Top-1/5/10/20 oracle coverage, mode separation, candidate-bank reachability, diversity. Coverage is not actual Top-1 localization.
- **Hypothesis decision:** Conditional Top-1 given successful Top-20, repair/harm, scene/seed breakdown; preserve fine success and visual-only successes.
- **Final system:** Joint success, R@1m, R@0.1m, runtime, bank storage and memory. Report point score vs. calibrated posterior separately; NLL/ECE require an explicitly normalized valid distribution.

Do not optimize only coarse joint success while silently harming fine precision. Pre-register a reasonable multi-objective selection criterion or report the Pareto trade-off.

## 6. Dataset scaling, independence and budget discipline

- Before training, audit HSSD usable scenes, dense bank completeness, source family overlaps, prior teacher exposure and floorplan quality. Reuse approved existing RGB/RIR/features and hashes. If more scene banks are needed, prototype the data generation cost first.
- Train12/24 (or actually available groups) may be used for controlled exploratory scene-level training and selection, with **scene-grouped nested splits where feasible**. Use inner development for architecture and threshold selection. Preserve outer held scenes without training adapter/head on them.
- Previous Val8/Val11 and the 11 head-held HSSD scenes have been seen by the research process; some were also visible to the upstream visual teacher. They are development or teacher-exposed transfer evidence, not a pristine end-to-end Test.
- Historical acoustic encoder selected on Val3; visual teacher trained on Train116 and monitored Val19. Record any overlap of new scene families. If an upstream-teacher-unseen evaluation set is unavailable, say so explicitly and do not mislabel teacher-exposed scenes as independent generalization.
- **Never inspect Test labels or use Test for architecture, hyperparameter, or threshold selection.** Do not convert previously tested scene groups to untouched validation.
- Keep experiments small initially, with informative comparisons instead of hundreds of redundant fits. Scale only if a specific falsifiable hypothesis warrants it. No uncontrolled paid resources or external upload. Respect existing local compute limits and do not overwrite checkpoints or the user's git changes.
- For uncertainty, aggregate within query across seeds and use scene-level paired bootstrap; never count multiple pairs, renders or encoder seeds as independent scenes. CI on reused exploratory data is not selection-bias-adjusted confirmation.

## 7. Autonomous execution and stopping decisions

Run the full sequence **Audit → A feasibility → A decision receipt → B frozen controlled pilot → B partial unfreeze controls (when feasible) → frozen-candidate evaluation → targeted counterfactuals → next-action decision**.

Do not stop at the first audit or the first weak A result. If an implementation is blocked, use the verified alternative features, record the exact blocker and continue the next informative task. If a candidate improves a weak baseline but not GEOQ, report it as a negative strong-baseline result. If neither arm improves, investigate whether the limitation is observability, missing physical correspondence, optimization, sample/scene diversity, reference mismatch, or duplicate information; avoid architecture lottery.

Stop this research *stage* when the primary distinctions have strong evidence, a reproducible improvement plus ablations is found, or further experiments have little information value under the authorized compute and data constraints. Save `NEXT_ACTION.md` and exact resume commands. Do not promise infinite background computation or autonomous running beyond the live Codex session.

## 8. Deliverables

1. `AUDIT.md` — code/asset/feature/reference/GT provenance and exact baselines.
2. `A_FEASIBILITY.md` — cue labels and physical limits, oracle vs. inferred cue, simple probes, control comparisons, visually ambiguous pair contribution, repeatability, fail cases.
3. `B_EXPERIMENT_MATRIX.md` — hypotheses, exact trainable/frozen layers, params, objective, fixed candidates and matched control contracts.
4. `RESULTS.md` — Train-held-scene and exploratory validation, scene/seed effects, strong GEOQ comparison, extraction/decision/representation contributions separately.
5. `MECHANISM_AND_FALSIFICATION.md` — what information actually matters, competing explanations, donor/no-audio/calibration/geometry prior controls, failures and transferable principle.
6. `REPRODUCE.md`, experiment logs, configs, checkpoint hashes, GT-free inference interface, `NEXT_ACTION.md`.
7. A simple method figure for any promising candidate and a one-page IJCAI contribution assessment: **established fact / causal hypothesis / negative result / independent generalization still required**.

### Final decision question

**Can acoustic observations reliably provide pose-discriminative directional structural cues, and can a trainable floorplan-conditioned V/A joint representation provide extra information beyond B2 + GEOQ and equally informed score/concat baselines on scene-held-out data?**

A may be negative while B succeeds, or vice versa. Both are acceptable scientific outcomes. Optimize research understanding and falsifiability before claiming a new architecture or novelty.

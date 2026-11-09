# Candidate-conditioned interaction: matched controls

Study ID `track-b-matched-interaction-20261109`, 2026-11-09. **Status: running; first fit completed, comparison incomplete. Evidence: Exploratory.** This follows the [structure and independent fusion experiments](../2026-11-09-structure-supervision/README.md). Full Track B goal remains active.

## Hypothesis and competing explanations

Candidate-conditioned token attention may improve full-field matching over pooled-token concatenation and separable modality scores. Alternative explanations are additional capacity, different initial weights, inherited structure-matching skill, weak visual representations, and optimization differences. All three current variants use exactly2,429,659 parameters, the same backbone checkpoint per seed, same seeded new modules, same query order and1,000 update budget. Actual compute differs and must be reported; equal parameter count is not equal FLOPs.

No Track A weights/reference bank/B2 proposals enter the pipeline. The input remains RGB, six-channel observed RIR and FP. All valid0.25m/36yaw candidates are scored before Top20 extraction; GT is used only for training loss and evaluation. This is the current-cohort full-field M1/F2/F3 comparison; unlike the prior additive model it contains an actual query/key/value attention operation.

## Dataset and initialization

Reuse the frozen16Train/6Val cohort,1,024/384 queries, seeds42/43/44 and preprocessing of the preceding study. No Test observations. Cache and pretrained hashes are recorded in `protocol.json`; original input/split sources are in the preceding report. Reused development scenes prohibit independent-validation claims.

Load the archived `frozen_v1/fusion_<seed>_pretrained.pt` geometry-pretrained backbone, fresh AdamW lr0.0003, weight decay0.0001, batch8, dense soft CE plus10×geometry auxiliary loss, gradient norm cap5. Skip1,500 sampling steps to match the previous dense phase order. All three encoders and new heads train jointly. Save initial states for tensor-by-tensor comparison after all modes initialize; this comparison is **pending**, not inferred solely from equal seeds. Select only the final scheduled update, no epoch selection.

## Models and separation of effects

Shared visual/acoustic convolutional features feed both the pretrained16-ray heads and token projections. Visual tokens preserve16 image columns; acoustic tokens preserve4 frequency×16 time bins with positional features. FP CNN features and geometric rays create candidate queries of width32.

- **Attention (M1 variant):** actual scaled dot-product Q/K/V over the16visual/64acoustic tokens for every candidate. Candidate-dependent visual/acoustic retrieved tokens enter a gate and nonlinear joint head, including products and absolute differences.
- **Pooled concat (F2 control):** same projections, but use mean K/V, candidate-conditioned tanh transformation, then the identical gate/joint head. No attention weights are computed. This is a pooled-token concatenation control, not an assertion that all possible concat baselines were exhausted.
- **Separable (F3 control):** same parameterized modules; apply the nonlinear head separately to visual+FP and acoustic+FP, sum their scores, and gate audio using only acoustic+FP. Its score is mathematically separable between the modalities conditional on FP. A cross-difference unit test verifies this property. Shared parameters across the two additive terms do not create inference-time modality interaction.

The common geometry score provides0.85visual +0.15×gate×acoustic, plus0.1×new head score. The gate is twice a sigmoid (range0–2); its reliability interpretation is unverified. Query tensors have no labels/scene IDs. Candidate chunks2048 and activation checkpointing retain exact full-bank softmax gradients. The final score tensor covers the entire bank; chunking is not candidate pruning.

## Validation and interrupted run

24 functional tests pass, including original contracts, checkpointed vs unchunked score equivalence, gradients to all parameter tensors, equal parameter counts, and separable-score cross difference. The PyTorch checkpoint path emits a known CPU-autocast deprecation warning; it is not an experiment failure.

Original prototype profile (before removing key biases) used2,429,723 parameters,231,156 candidates and batch8. First-step full forward/backward/optimizer time: attention0.749s, concat0.479s, separable0.461s. Peak allocated bytes243,123,200 /242,798,592 /257,413,120 with one scene resident. These are cold first-step measurements, not steady-state deployment or full cache memory. All three encoder gradients and retrieval/head gradients were nonzero. [Original profile](prototype_profile.json). Updated-model cost must come from actual run timings.

`matched_v1` was deliberately stopped after detecting that each attention key bias cancels inside softmax. Those64 parameters did not contribute to the attention function but could contribute to the pooled controls. This undermined the intended active-capacity match. Logs and initial checkpoint are retained externally, no final checkpoint or accuracy is claimed. Last logged update index100 (at least101 updates); final exact step is unrecorded. Docker stop returned training process exit137. This was an intentional audit correction, not an OOM or convergence failure.

Remove key biases in **all** variants; corrected source `aa452ee78a75af31d8b426684a9c81f94ff00039`. Start fresh `matched_v2` with a new output directory. Tests still pass. Full nine-fit comparison is pending. No claim that attention is better is justified at this point.

## Evaluation and remaining tests

Final checkpoints evaluate normal inputs, cyclic within-scene audio shuffle, and brightness×0.2 RGB. Report Joint≤1m/30°, R@1m, Fine≤0.1m, raw20, diverse20, nearest-grid NLL and entropy. Fine uses1e-6m tolerance. Brightness is a deterministic inference perturbation, not realistic night-domain transfer. Diverse20 is computed only for normal in this runner; do not invent degraded diversity results. Paired uncertainty must resample scenes with seeds averaged, not treat all query-seed records as independent.

Decision pending results: compare attention to both capacity controls and fixed independent fusion8.941%. The latter has separate FP encoders/checkpoints and differs in capacity/compute. If matched attention does not improve, retain negative evidence and diagnose geometry/visual learning before adding complexity. OOF discrimination, robust material/furniture/FP shifts, stronger visual initialization, grid density, complete cost/calibration and unused final evaluation remain incomplete.

## Reproduction and artifacts

Code/Docs branch `refactor/hssd-two-branch-fusion`. Original prototype source `dc8e689325ccc168ab977daf292e3c71049de2bb`; interrupted runner source `65a8236103d82de5b0c5ac3a68c2ff9dd1828b8c`; corrected source above. External data/output root `/home/jeongeonlee/track1/experiments/track_b_interaction_20261109`. Raw logs, serialized inputs, checkpoints and per-query predictions stay outside Git. Docker image SHA `e51cb712100f25d0018aa65e371e63deaa51d12601c465ce698c0c446800c318`.

Executed corrected run:

```bash
docker run --rm --network none --gpus 'device=0' \
 --user "$(id -u):$(id -g)" -e OMP_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2 \
 -v /home/jeongeonlee/track1/AV-FPLoc:/work:ro \
 -v /home/jeongeonlee/track1/experiments/track_b_structure_20261109:/prior:ro \
 -v /home/jeongeonlee/track1/experiments/track_b_interaction_20261109:/out \
 -w /work --entrypoint python avfploc:cu124-clean \
 -m track_b.scripts.train_interaction --cache /prior/cache \
 --pretrained /prior/frozen_v1 --out /out/matched_v2
```

Use a fresh output directory for a deliberate reproduction. A running job must be observed, not restarted because a polling call times out. Initial aborted outputs must not be deleted or overwritten. Unrelated user renderer edits are preserved and excluded from commits.


## Interim evidence (not the completed comparison)

Attention seed42 completed1,000 updates: Joint4.6875%, R@1m7.5521%, Fine0.2604%, Raw20 32.8125%, Diverse20 59.8958%, NLL9.5015. Audio shuffle Joint3.125%; brightness×0.2 Joint1.3021%. Training275.724s; whole-cache peak allocated1,777,123,328bytes. These are one-seed results, not evidence for superiority or general inferiority. Other fits continue on the same live process. [Completed-run snapshot](interim_completed_runs.json).

The attention42 and concat42 **initial checkpoint tensors are exactly equal** across all state keys; further modes/seeds await initialization and the full post-run audit. [Partial verification](initialization_partial.json). Aggregator `track_b.scripts.summarize_interaction` refuses to publish a full result unless all nine fits and final update indices exist, every initial state matches its seed anchor, and parameter counts match.

[Selected-condition audit](selected_conditions.json) checked all1,408 already-selected Train/Val observed sidecars: furnished_native_scene, materials=false, transmission=false throughout. No paired furnished/unfurnished/material-varied observation is represented in this cache. This is not an assertion of absence across the entire dataset. A top-level legacy catalog read during discovery described clutter conditions for a Test scene; these are metadata only and its observations were not opened or used. A broad filename listing was stopped because it was unnecessarily slow; it provided no robustness evaluation. Follow-up must obtain a suitable non-Test query-condition pair or state that physical shift remains unvalidated; it must not repurpose candidate FP-RIR references as the proposed observation pipeline.


## Full-bank runtime and frame audit

At audit source `23a398b`, CPU inference of the completed attention42 checkpoint on one development query scores all99,612 valid SE(2) candidates. Raster scattering has exactly99,612 finite entries. Removing GT/auxiliary target keys from the loaded scene does not change the output bit-for-bit. Only the selected observation-cache and checkpoint files are opened during loading; forward execution opens no observation/reference files. PyTorch lazily imports its `opt_einsum` bytecode, explicitly retained in the trace. [Actual runtime trace](runtime_audit.json). This proves the traced entrypoint for one real query/map, not an exhaustive audit of every pipeline entrypoint.

First audit attempt at `b2e6995` stopped because its zero-file-open assertion classified that framework lazy import as unexpected. It did not find a reference loader; the initial failure log is retained externally as `runtime_audit.log`. Revised audit permits exactly that known framework bytecode path, records it, and still rejects all other forward file accesses; replay exits0. No performance values or training artifacts were altered.

The test suite now has25 passing tests. Added interaction-specific tests verify joint rigid translation/rotation of map coordinates (tolerance3e-5 absolute/1e-5 relative), new raster sizes, and score changes under observed-audio swapping. This is coordinate-frame equivariance, not invariance to physically rotating the camera/microphone observation independently of the map. Physical orientation and real-world generalization remain separate requirements.

Reproduce with the same container mounts, CPU-only (omit `--gpus`):

```text
python -m track_b.scripts.audit_interaction_runtime --scene /prior/cache/108736611_177263226.pt --checkpoint /out/matched_v2/attention_42.pt --out /out/runtime_audit_v2.json
```


## Independent fusion brightness control (CPU, same-device replay pending)

To match brightness×0.2 in the interaction evaluation, freeze the separately trained visual/acoustic checkpoints and evaluate the existing0.85/0.15 fixed fusion on the same six scenes/three seeds, without calibration or learning. CPU normal→dark Joint: visual2.083→0.087%, acoustic5.122→5.122%, fixed fusion9.028→1.215%. Dark fusion is worse than acoustic-only, demonstrating negative transfer under this specified photometric perturbation. This does not establish physical night-domain robustness or a benefit from the learned attention gate. [All metrics, mismatches and paired scene CIs](fusion_photometric_cpu.json).

CPU vs prior GPU normal replay checks1,152 query-model cases. There are160 NLL differences beyond atol2e-5/rtol1e-5 (maximum absolute0.00075245nats) and one Joint/R@1m discrepancy (scene108294783_176710410, seed44, query1298). Consequently CPU fixed fusion9.028% differs from original GPU8.941% by one query-seed success; no silent replacement of the GPU result. The numeric cause (near-tied scores versus a decision boundary) remains unresolved. Use within-CPU comparisons here; same-GPU photometric replay is required before final interaction comparison.

Initial `fusion_photometric_v1` at source4855c41 exited1 on the first NLL replay mismatch; no full evaluation result claimed. Its log is retained. Revised sourcebb68791 runs the complete audit and reports every discrepancy instead of hiding it; `fusion_photometric_v2` exits0. This computation used two CPU threads and no GPU while training was running; training wall times are not claimed as controlled standalone timing benchmarks.

The subsequent device-selectable runner at3370937 is prepared for a fresh `--device cuda` output after ongoing training, so its timing is not disturbed by competing GPU evaluation. This GPU replay has **not** yet been executed. Completed CPU command used the standard mounts and no `--gpus`:

```text
python -m track_b.scripts.fusion_photometric --cache /prior/cache --run /prior/supervised_v1 --fusion-reference /prior/independent_fusion_v1 --out /out/fusion_photometric_v2
```

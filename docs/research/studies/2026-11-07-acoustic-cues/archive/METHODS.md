> 역사적 연구 기록. 원본: `acoustic_cues_20261107_1830/METHODS.md`. 원본 SHA256은 [산출물 목록](../artifacts.json)에 있습니다.
> 당시의 완료·재개 지시는 현재 실행 요청이 아닙니다. 최신 판정은 [현재 결론](../../../current_findings.md)을 먼저 확인하세요.

# Acoustic structural evidence: methods and interpretation limits

## Scope and evaluation

This study follows the user’s *Understanding Acoustic Structural Evidence for Visual Ambiguity Resolution* specification. Original B2, GEOQ, encoder weights, hypothesis sets and source caches are preserved. Train12 has 6,387 queries, Val8 4,190 and Val11 (`extra`) 5,775. Val8/Val11 are reused development sets, not independent generalization. No Test observations are read. Upstream encoders have existing training exposure; scene-held-out probes/rankers do not make the entire pipeline strictly nested.

Joint success is distance ≤1 m and yaw error ≤30°. R1 ignores yaw; fine success is distance ≤0.1 m. Bank poses use 0.2 m cells and 36 world yaw bins. Acoustic and fusion coverage use original NMS rules. Fixed-set discrimination always uses the original B2 support/refinement Top-20, while re-extracted support performance is separately named. Coverage and fixed-set discrimination have different denominators/availability constraints and must not be conflated.

Train-derived entropy tertiles define low/medium/high visual ambiguity. Baseline predictions define retrospective rescue/failure groups; these GT-defined groups are not inference-time routing rules. Pure and exchanged branches are explicitly separate. The strong-wrong group uses the Train 2/3 quantile of exchanged acoustic Top-1/Top-2 score margin. Signal failure figures use the analogous **pure** margin and say so. Bootstrap draws whole scenes, keeps paired variants together, and averages the two seed contexts within scene; 10,000 draws with RNG seed 20261107. Seed counts are not independent sample counts. Group membership can differ between seeds. Intervals describe development scene variability, are not multiplicity-adjusted, and are not an independent confirmatory test.

## Acquisition and actual available signal

`data_catalog.json` maps every used query ID, raw row, channel roll, feature shard and FP reference. `signal_audit.json` checks existence of all indexed raw files/metadata and physically checks 16 deterministic records per role/scene. `raw_subset_verification.json` separately hashes every waveform used by the raw diagnostic subset against both its metadata and the feature-cache receipt. Full raw availability does not mean every raw waveform was re-encoded or hashed.

The fixed active source is at the center of a 6-microphone ring, radius 0.05 m. Sampling is 48 kHz. The direct path is approximately sample 7 (0.146 ms); most measured direct-window peaks/onsets are constant, with rare Train exceptions. A common source-ring distance cannot encode room position by direct travel time. Direct-window maxima are not guaranteed direct-path estimators in exceptional records. No peak is assigned a wall identity.

Production rendering uses six sequential mono listener positions. Materials and transmission are disabled; 20,000 indirect rays, 512 source rays and depth 50 are used. FP references come from extruded floorplans, observations from native furnished 3D scenes. Runtime Habitat 0.2.2 reports temporal coherence disabled. Neither real material identification nor field-measured RT60 is established. The SoundSpaces simulator describes stochastic ray tracing and pressure-response synthesis from energy responses; this motivates testing repeatability rather than interpreting every inter-channel phase as a physically synchronized array observation. See [official SoundSpaces2 documentation](https://github.com/facebookresearch/sound-spaces/blob/main/SoundSpaces2.md) and [SoundSpaces2 paper and supplement](https://par.nsf.gov/servlets/purl/10424655). Local code/metadata determine the actual settings used here.

## Encoder input and interpretable descriptors

| Input | Actual information | Main limitations |
|---|---|---|
| E, 6×19×95 | Normalized STFT energy, roughly 2–128 ms and 500–4000 Hz; log1p; FFT256/hop64 | Global absolute energy removed; direct arrival excluded; frequency/time resolution finite |
| C, 2×6×2×19 | Adjacent microphone complex coherency; **early <50 ms and full <128 ms** | Second window is not late-only; phase-sensitive descriptor is not established DoA or wall correspondence |
| N, 6×252 | Microphone-mean-removed, 500–1500 Hz smoothed envelope, 0.5 ms bins, per-mic maximum normalization | Absolute amplitude and carrier phase unavailable; channel-common component removed |
| Final acoustic embedding | 128 dimensions | Decodability and decision use are separate questions |
| Intermediate probe representation | Mean/std summaries of streams, 576 dimensions; fixed projection to128 also tested | Larger representation has greater effective probe capacity |

Handcrafted groups per microphone: T23 (16 energy time bins, moments and cumulative early fractions), F9 (6 spectral bands and moments), C16 (real/imaginary frequency summaries and magnitude for early/full), N27 (envelope bins/moments/three separated dominant peaks/early fraction/gradient), M1 (relative microphone energy). Total456 dimensions. These are deterministic GT-free signal descriptors, not reflection labels. Query channel rolls follow the original schedule; FP world-yaw mapping uses six base directions and cyclic ring rotation.

The raw64-query-per-scene subset uses uniformly spaced query rows, then the union of their original seed42/43 candidate rows, without GT selection. Raw22/mic descriptors include windowed peak/onset, total/pre-2ms log energy, tail energy fractions/quantiles, three broad separated peaks and adjacent GCC-PHAT lag/peak/margin. GCC uses a roughly2–32 ms window and ±8 samples; multipath makes it an ambiguous correlation lag, not a direct DoA estimate. Energy quantiles are not RT60. Raw timing windows start at sample103, i.e.2.146 ms from simulation origin; time labels are rounded. Feature summaries use the existing feature-contract2 ms origin.

Common positive waveform gains0.1/2/10 are tested by recomputing E/C/N. Within the tested gain range and normal signal operating regime, their invariance supports non-identifiability of absolute signal level from these normalized inputs; epsilon/clipping behavior at degenerate amplitudes is not tested. Correlation-based level prediction from geometry or shape remains possible and does not contradict invariance. Near-constant raw direct-timing targets make relative probe MSE ill-conditioned; use raw variance and distributions rather than interpreting a large ratio as information loss.

## Controlled experiments

**Frozen query perturbations.** Thirteen Train variants: baseline, no_E/C/N, E/C/N-only, no_C_imag, time/frequency/spatial averaging, early_EN and late_EN. Six predeclared variants are confirmed on development sets. Encoder deltas are anchored on historical cached embeddings/tokens to preserve baseline exactly; original token extraction uses batch64. Full-bank pure/exchanged acoustic and fusion Top1/5/20, GT mass at standardized score temperature0.05, support re-extraction and fixed20 GEOQ are evaluated separately. Query-only changes leave FP references unchanged. These are OOD sensitivity tests, not causal estimates of unavailable information.

**Both-side frozen controls.** Transform all query inputs and the union of original fixed20 FP references, comparing query-only/reference-only/both. Report acoustic argmax on those20, not full-bank Top-K. Early_EN/late_EN mask only E/N and retain C; they also retain the original normalization scale. They cannot establish early/late necessity. Time averaging retains C’s temporal summaries. Both-side transforms reduce domain mismatch but remain frozen-model OOD tests.

**Matched cue-only learning.** Fixed Ridge alpha10 learns query descriptors→nearest-GT FP descriptors with Train-only scaling and four scene folds. Both query and FP remove the same groups. Full-bank retrieval and fixed20 scores are measured. All arms have nominal456 input/output coordinates, zeroing removed groups; their effective active dimensionality differs. This is matched linear learner retraining, **not retraining the original acoustic encoder**. No B2/visual/embedding scores enter fitting; the same floorplan bank and fixed-set availability still apply. Ridge is deterministic: two seeds only change B2 fixed candidates and subgroup context, not independent Ridge training runs.

**Early/full/late coherency.** Recover late cross moments by subtracting early from full cross moments using E-derived window powers; normalize by late powers. This uses the original STFT frames, not a newly padded STFT starting at50 ms. Float32 algebra and cached-float16 error are separately verified. Early/full/late each use48 active features in the same456 coordinates; real/imaginary/magnitude subsets have36/36/24 active features. All six arms are evaluated without development-driven selection. No new waveform/rerender contributes to training.

**Conditional cue additions to GEOQ.** Keep original GEOQ78 features, add120 cue comparison summaries, then absolute/candidate0-relative doubling gives396 columns. All arms use150 iterations,15 leaves,min-leaf100,L2=10,learning-rate0.07 and the original joint+0.5fine utility. Fold training uses the previous72 outer-adapter training caches; held scenes use original candidates. Compare all cues, removals, donor query cues and a score-basis expansion; unused coordinates are zero. Existing GEOQ/B2 acoustic evidence remains, so this tests incremental usefulness, not removal of all acoustic information. Frozen thresholds are seed42=0 and seed43=0.01. GEOQ replay must reproduce the previous choices.

**Representation/geometry probes.** Train scene-held-out Ridge predicts signal descriptors from128D final/576D intermediate/fixed128D projected intermediate representations. Another probe predicts log1p eleven floorplan rays from512 uniform FP poses per scene. Low error means linear association/decodability, not necessity, absence of other information, or observation-to-wall correspondence. Handcrafted+GEOQ is the controlled incremental representation test above.

**Render falsification.** Two Train FP scenes, two poses each, forwardA/forwardB/reversed listener processing order: twelve fresh rings. Compare same-pose repeat variability with different poses. Pair counts are dependent and too small for a broad scene-level causal claim. Reversal includes a fresh stochastic render, so it does not isolate processing order from stochastic variability. No original data or references are overwritten.

## Provenance and known failed attempts

- Initial batch32 token replay failed; batch64 restored exact historical tokens. Failed logs retained.
- Initial catalog path construction was corrected before producing the authoritative `data_catalog.json`.
- Initial Train screen applied full-fit GEOQ provisionally. `correct_train_oof.py` replaces GEOQ predictions with held-scene models and verifies baseline against prior nested OOF. Only `oof_screen_seed*/train` enters Train screen analysis.
- During initial Train screen execution, the on-disk evaluator acquired a `--modes` option for later development runs. The actual initial Train source is archived as `evaluate_screen_initial.py`, SHA256 `c4b0548708273a7b1f129504316eee39291f1762ee14a22499e4a58948a103e7`. Some later Train receipts hash the newer on-disk file rather than the already-loaded source; receipt hashes alone are not proof of execution provenance. Default computation was unchanged.
- Initial renderer container working directory was invalid under a read-only mount; rerun used `/output`. Renderer output is separate from the dataset.
- An arbitrary0.005 late-coherency cache tolerance failed. Investigation separated near-exact float32 algebra from float16 source quantization rather than hiding the discrepancy. Verification records both.
- All completed and failed artifacts remain in this directory. Existing prior experiment directories and user edits are preserved.

## Follow-up falsification: late-coherency quantization

After early/full/late comparisons, `late_quantization_control.py` tests whether float16 moment subtraction alone explains weak late-C results. It uses the first Train scene in each fold, the existing64-query raw subset and original fixed20 sets. Both query/reference late descriptors are recomputed directly from raw STFT using exactly the same frame boundaries. The cached-feature-trained held-scene C_late Ridge/scalers are frozen. No development data or parameter selection enter this follow-up. This is a fixed-set robustness test, not full-bank evaluation or retraining on fully unquantized raw features. `late_quantization_control.json` retains all score changes and per-seed outcomes; source raw hashes were already verified in the full subset pass.

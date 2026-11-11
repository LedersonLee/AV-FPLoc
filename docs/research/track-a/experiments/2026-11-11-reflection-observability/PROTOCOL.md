# Experiment 1 protocol — frozen before cohort measurements

Date: 2026-11-11 KST. Scope: reflection observability only; no network, scorer, localization accuracy evaluation, or Experiment 2–4. Evidence: Exploratory simulation with controlled interventions, historically exposed development scenes.

## Competing explanations and falsification

H1: valid finite first-order reflected paths predict localized acoustic energy. H0: early-time decay, window duration, broad reverberation, or map closure explains an apparent advantage. A peak alone never supplies wall identity. Require removed/moved-wall contrasts against same-condition repeat variability. Energy alignment may hold while fine inter-channel phase/timing fails.

## Controlled suite

Single vertical wall, rectangular room (8×8m), corridor (2.4×18m); source x=.17m, height1.5m, front-wall distances .3/2/6m. Six 5cm-ring omni receivers. For each: baseline, independent repeat, front wall removed, front wall moved outward1m (36 rings). Finite panels extend y=−8..12m. A remote floor at y=−20m prevents a degenerate simulator navigation bounding box; its >125ms monostatic return lies outside the analyzed 1–80ms range. Report all cases including overlap and weak returns. No reflection-path GT exposed by renderer: geometric path is prediction; intervention is causal test. 48kHz,20k indirect/512 source rays,depth50,diffraction enabled,materials/transmission disabled,temporal coherence off,default maxIR4s. Default variable-length output is retained whole.

Pilot: one ring,0.756s render,751472bytes; projected36 rings≈27s/27MB (scene initialization/complexity can vary). Container limited4CPU and8GiB; host20 logical CPUs, within80% bound. Initial missing trimesh and failed PLY/zero-depth OBJ load attempts are retained in local pilot logs. GLB and remote floor fix loaded-bounds issue. No bulk HSSD rendering needed unless raw support is insufficient.

## HSSD cohort and geometry

Use all12 historical directional-structure Train scenes with saved pure-visual NMS Top20; uniformly32 of128 query rows per scene, before observing new evidence. Existing128-row ordering and query IDs preserved. No Test payload/results opened. Teacher Train116/Val19 exposure and repeated research use prevent independent-test claims. Same candidates for every control, no query truth in feature/path selection. Known acquisition height1.5m is a sensor assumption, not query XY/yaw ground truth. Apply query channel roll from schedule; candidate yaw supplies hypothesized ring orientation.

Predict from floorplan extrusion vertical cross-sections. Require finite specular point, source/receiver same side, both legs unobstructed; reject out-of-range paths. Architectural stage cross-section is an audit: FP reflection point within0.20m of parallel native vertical surface is architecture-supported; otherwise unknown/closure, analyzed separately. This is geometric support, not semantic proof of a wall (fixed built-ins may remain). FP includes closed exterior boundaries; black/outside is never automatically a physical reflector. Merge coplanar adjacent triangle cuts, deduplicate identical paths.

## Fixed analysis

c=343m/s; sample0 simulation origin; no peak-based recentering. Primary half-window0.5ms; report0.25/1ms sensitivity without selecting a winner. Analyze1–80ms; report paths outside range/captured support. Per-path peak energy, mean squared energy normalized by full early mean, peak/mean sharpness, six-channel aligned waveform correlation and observed-minus-predicted inter-channel peak spread. Equal-width controls: uniform random time, ±5ms shift, next visual candidate geometry, exact same-scene time/distance-marginal donor-window permutation across queries, and unguided full early RIR. Geometry arm includes every valid candidate path; candidate0 reported separately; no GT-selected arm. Random seeds42/43/44 for random/donor controls.

Aggregate paths within candidate, candidates within query, queries within scene; paired scene bootstrap10000 draws. Twelve scenes, not thousands of paths, are uncertainty units. Report distance/bearing/overlap strata, missing support and all scene effects. Candidate consistency is diagnostic only; no scoring-rule tuning or reranking. Full early has unequal duration and is compared using per-sample power rather than raw total energy.

Implementation support clarification (before HSSD metric execution): predicted channel centers must all lie2–79ms so the widest±1ms window stays within1–80ms. Full early reference is samples48..3840 inclusive. The exact-marginal control holds every predicted time/distance fixed and permutes the assigned query waveform by a nonzero cyclic shift within scene, three fixed seeds. It preserves the window marginal exactly rather than approximately matching time bins.

## Post-primary renderer falsification (explicit exploratory follow-up)

The36-ring run showed single-wall peaks later than c343 predictions (~14ms vs11.66ms at2m;~41ms vs34.99ms at6m). Do not re-center HSSD windows or infer wall GT from these peaks. Diagnose independent sample rates24/44.1/48/96kHz, diffraction off,depth1 with diffraction off,200k indirect rays with diffraction off; baseline included,2 distances×7settings×3fresh repeats=42rings. Compare whole early response, disappearance under wall removal, and sample-rate-invariant arrival in seconds. These tests separate sampling conversion, stochastic noise, and higher-order/diffraction explanations; they cannot identify hidden backend behavior without path exports/source.

Mesh-validity follow-up: all seven renderer settings retained the late single-wall peak. Add closed0.1m-thick wall boxes with front face unchanged atz0 and one-sided front-facing panels, each2distances×3repeats, to distinguish duplicate coincident-face artifacts from backend timing behavior. Same remote floor, source/listener positions and production settings. This adds12rings; no primary result is replaced.

Direct-path clock audit: add wall-free source→array-center separations4m/12m (matching reflection lengths),3fresh repeats each. This distinguishes globally incorrect sound speed/time scaling from reflection-specific behavior. These6rings are explicitly not collocated acquisition and do not enter HSSD or the primary36-ring suite.

Verification correction: uniform random centers were initially2–79ms but per-mic offsets could put the widest1ms sensitivity window up to0.15ms outside1–80ms. Narrow the random-center range by the largest channel offset, preserving inter-channel offsets and every other control. Rerun all random arms and retain v1 summaries/metrics locally. Matched-marginal primary result and geometry windows are unchanged. Final viewer tables are generated from the corrected summary rather than hardcoded.

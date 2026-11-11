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

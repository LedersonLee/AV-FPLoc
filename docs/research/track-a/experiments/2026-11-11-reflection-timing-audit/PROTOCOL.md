# Experiment1B — protocol, 2026-11-11 KST

Scope: renderer physics/timing audit only. Preserve Experiment1; no new encoder/fusion/ranker/localization/material research. HSSD rerun gated by controlled physical evidence. Research-partner autonomous investigation; observed/causal/consistent/unresolved distinguished.

## Frozen estimator and acceptance criteria (before new estimator execution)

Reuse all96 prior RIRs and verify hashes. Primary single-wall2m is development/diagnosis;6m and moved/removed/near0.3m are reporting conditions with no threshold selection. These are historically examined simulated conditions, not pristine independent tests. All settings reported; no winner selected after viewing evaluation. Sampling48kHz, c343, common sample0; no fitted speed, delay correction, per-channel realignment or GT-centered search.

Search1–80ms for all single-wall estimators, independent of expected return time. Separate broad waveform maximum, causal0.125ms energy-envelope onset, causal short-time energy increase, causal band-limited500–2000Hz and2000–8000Hz onset, thresholded energy centroid and envelope peak. Filtered estimates retain filter delay; never use zero-phase future leakage to claim physical first arrival. Primary power threshold1e−6 of maximum direct squared amplitude (reference from removed same-distance ring), sensitivity1e−8/1e−4; additionally report relative-response onset1%/10% with global amplitude gate so zero waveforms cannot become detections. Raw exact-nonzero onset is numerical diagnostic only. Paired removed energy-difference onset is an intervention diagnostic, not an HSSD inference method.

Primary tolerance0.10ms absolute arrival error covers2samples+1cm round-trip geometry error; movement-shift error≤0.20ms; within-condition repeat SD≤0.10ms; removed-wall false detections≤5%. All tolerances fixed here. Finite-segment image-source ground truth uses independent scalar wall-plane reflection and both leg lengths, compared to previous geometry solver. A separately generated fractional-delay image-source impulse is a labeled analytic reference only, never a SoundSpaces recording.

## A/C/H2/H3/H4 interventions

Audit installed Habitat C++/Python wrapper, RLR public header, exported symbols, library hashes/revisions. Internal symbol names alone are not proof of an internal algorithm or usable ABI. Do not guess opaque struct layouts. Public geometric first-hit ray API is geometry evidence, not an acoustic path GT export. Inspect paper's early-path clustering vs histogram/pressure conversion distinction.

Use the documented RLR C API in an isolated executable for same mesh/source/listeners, compare it with retained Habitat baseline. New controls at2m/6m: baseline,listener radius0 vs default0.1m,indirect disabled,direct disabled,temporal coherence on,six listeners in one context. Include3independent fresh contexts per arm; retain within-context same-position repeats for temporal on/off. All flags/position/normal/hit distances and API return codes logged. This is source/listener/rendering audit, not fitting a new renderer. Reflection suppression uses `indirect=false`, which suppresses all indirect components, not a claim of exact order isolation.

Pilot one ring before scale; container4CPU/8GiB, acoustic2threads (host20CPUs, under80%). Do not rerun prior sample-rate/diffraction/raycount experiments. Need not regenerate HSSD.

## D gate and completion

Only run HSSD feasibility if A–C establish physically attributable usable arrival evidence with the above tolerance/causal controls. Otherwise skip D and explicitly conclude limited observability or renderer-dependent artifact with verified vs unresolved mechanism. A–C may fail; failure is valid completion. Preserve all failed attempts/corrections. Required: research report, physical timing audit, mechanism overview, actual-result interactive HTML, code/environment/input hashes/costs, Track A canonical index, verification and scoped branch Push.

Pilot audit: initial C wrapper called defaults before initializing `thisSize`, received explicit API error; that pilot is invalid and preserved. Corrected wrapper initializes size first and checks every configuration return. Corrected one-ring pilot:0.03146s,440736bytes,unitScale1; native first-hit wall distance1.999999881m, normal+Z. This reproduces the late~14ms return without Habitat. Main run is3fresh contexts×2distances×6arms×6channels=216mono outputs, plus stationary on/off5repeats×3contexts×2distances=60mono outputs (276mono =46six-channel equivalents). Rough size20MB, few seconds predicted; no old sample-rate/ray-count/diffraction suite repeated.

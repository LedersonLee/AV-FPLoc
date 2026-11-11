# Frozen PoC protocol — 2026-11-11

Frozen before acoustic outcomes. Code config is authoritative. Reuse existing raw visual NMS20 scores (softmax T=1, uncalibrated conditional mass). Select queries by top-two score gap <= log(3), no truth-based coverage filter; report attainable oracle coverage and unconditional joint success. No candidate insertion. All seeds share visual queries; bootstrap query clusters after averaging seed outcomes. No Test observations read.

Room/source/event sampling is independent of robot GT. No detection range or audibility model: semantic-only likelihood is constant in pose. B1 must equal B0; forcing a robot-room heuristic would contradict this generative model. B2 marginalizes event with fixed activation prior. B3 integrates all valid latent source locations, never true source xyz. Silence returns visual prior exactly. Angles are geometric oracle, not measured acoustic DoA.

Use raw channels 0..5 from seven-MEMS local UMA calibration with provisional 5 cm radius. Validate mono receiver locations, common timing origin and TDOA independently. Native HSSD furnished scene plus stage-only comparison; no FP extrusion substituted for HSSD.

Existing approved branch retained because shared checkout contains user changes; new code and outputs isolated. No existing Track code changed. Native import initially aborted with free(): invalid pointer; investigate documented system libstdc++ preload, preserve failure.

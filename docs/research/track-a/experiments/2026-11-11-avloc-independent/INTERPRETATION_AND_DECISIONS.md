# Interpretation and decisions — 2026-11-17

Primary Track A, AVLoc-style6ch HSSD adaptation. Conclusions below concern the completed seed42 primary20cm comparison on reused Val19/9,965queries; the10cm uniform-router control is also complete. These are development findings, not a reproduction of the original paper's binaural dataset results or an independent Test claim.

## 1. Does the AVLoc-style framework match or exceed Track A?

There is a metric-dependent tradeoff. At the identical20cmXY/10°yaw candidate grid, A3 Joint66.92% trails B2+GEOQ73.07%, but fine-position accuracy30.92% exceeds23.20%. A3−GEOQ scene-bootstrap macro95% intervals exclude zero in opposite directions: Joint[−7.808,−4.439]pp and fine[5.969,9.157]pp. B2/GEOQ averages seeds42/43; A3 has one seed. Bootstrap is over scenes, not independent optimizer replicates.

At10cm A3 reaches76.19%Joint and49.47%fine; the same A2 checkpoint reaches74.18/49.33%. This is a pose-grid intervention. It cannot establish superiority over20cmGEOQ, and native10cmGEOQ was not produced. [Metrics](REQUESTED_METRIC_TABLE.md), [paired evidence](A3_initial_contrasts.json).

## 2. Which components explain the differences?

The data reject the simple hypothesis that acoustic ray regression alone already solves global localization in this adaptation: A1Joint2.43%, rayMAE1.4785m. Its errors are not explained solely by ambiguity of perfect floorplan profiles. This does not prove that the raw acoustic input lacks structural information; sensor adaptation, representation and optimization remain rival explanations.

Joint A2 improves over A0 by14.82ppJoint and over the completed interaction-disabled control by24.59pp. Independent equal fusion performs poorly; preserving vision inside FoV and audio outside recovers some benefit. These results support the joint design over the tested alternatives, but disabled-pass/equal-mixture controls alter multiple computation and gradient paths, so pure attention causality is unresolved.

A3 adds1.997ppJoint over separately trained A2. Replacing weights within the same frozen A3 by uniform weights reduces Joint66.92→47.82%; every scene declines. This establishes an inference weighting effect under this training regime. It does not isolate dynamic reliability from direction-specific expert preference, and it does not substitute for a separately trained fixed-router experiment. Acoustic-only weight averages0.086%inside FoV, while the joint head can still carry acoustic information. [Interaction](INTERACTION_CONTROL_RESULTS.md), [routing control](A3_UNIFORM_CONTROL.md), [weight/error analysis](A3_ROUTER_AND_CALIBRATION.md).

## 3. Does a sharp posterior mean more accurate localization?

No. A3 failures have median normalized entropy0.000777, and a preserved case is31.47m/175°wrong despite a sharp field. Peak margins associate with success within methods but cannot certify it. Score scales differ, so raw gap magnitudes across frameworks are not calibrated probabilities.

Temperature improves A3's nearest-gridNLL/ECE/Brier on the scene-grouped development procedure but leaves residual miscalibration and unchanged Top1 poses. A2 at10cm has the opposite ECE/Brier trend despite improved NLL. The event used to fit a distribution and the event used to evaluate reliability matter. Reused checkpoint/teacher exposure prevents independent reliability claims. [Margins](PEAK_MARGIN_ANALYSIS.md), [calibration](A3_ROUTER_AND_CALIBRATION.md), [A2 counterexample](GRID_AND_A2_CALIBRATION.md).

## 4. Does FP-RIR performance justify its cost?

The existing bank-backed B2+GEOQ has a6.15ppJoint advantage over A3 on the matched primary grid, while ray localization has better fine accuracy and removes the inference FP-RIR-bank dependency. Existing Val19 raw FP-RIR storage is323,380,029,888bytes; FP embeddings1,315,731,456bytes, versus20cmray-bank/XYpayload11,421,280bytes. These encode different information and are not interchangeable compression formats. [Storage audit](bank_cost_audit.json).

Whether6.15pp broad-success advantage justifies bank generation/storage is an application decision, not established by accuracy alone. A3 requires a larger trained multimodal model; its measured warm stage sum is36.52ms versus A2's28.48ms on this shared host. Baseline timings start from cached embeddings and exclude input encoders/adaptation; no end-to-end speedup is established. Historical FP-RIR rendering time is unavailable. [Measured cost scope](COMPLETED_RUNTIME_RESULTS.md). The practical storage/deployment advantage is evidenced; a universal compute-efficiency claim is not.

## 5. What should Track A keep or reconsider?

Retain B2+GEOQ as the strong broad-success baseline and preserve its candidate-discrimination evidence. Treat AVLoc-style joint rays as a complementary precision/storage alternative. Keep Joint, fine and Top20coverage together: A3Top20coverage95.32% does not guarantee good final selection. A high-coverage field provides an opportunity for future discrimination, but no untested hybrid gain is claimed.

Reconsider equal mixing of weak acoustic-only estimates, equating sharper fields with reliability, and comparing different candidate spacings as model improvements. Prioritize future controlled tests that distinguish fixed directional preference from query-dependent reliability and that evaluate a genuinely matched native10cm baseline if rendering is separately authorized. These are proposed follow-ups, not missing results silently assumed successful. Seed43 replication remains deferred under the user's seed42-first request.

## Scientific limits and remaining execution

Full112Train, cleanVal checkpoint selection, single current training seed, reused19Val scenes and6ch sensor adaptation bound generalization. Test17 remains unused, Track B unchanged, and all checkpoints/failures preserved. This document answers the five interpretation questions from verified evidence. Both uniform controls, final paired receipts, figure review and integrity audit are complete. At10cm learned versus uniform Joint is76.19%versus56.25%; the inference-weighting conclusion is consistent across grids. Seed42 benchmark scope is complete; deferred replication and unmeasured end-to-end/rendering costs remain explicit limitations, not claimed findings. Each linked report records its executed source hashes; frozen training code62c58da20ed26ba95863638bbfd4926829999eec.

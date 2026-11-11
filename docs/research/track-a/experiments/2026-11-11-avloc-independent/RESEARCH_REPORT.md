# AVLoc independent reimplementation and matched HSSD benchmark

Study `avloc_20261111`, Primary Track A, started2026-11-11. **Active, incomplete**. Evidence level Exploratory.

Objective: reproduce the paper-based architecture as an explicit6ch HSSD adaptation, train on the full eligible cohort, and complete all requested matched baseline, calibration, observability, interaction and cost comparisons. No superiority of Track A is assumed.

## Current evidence

Paper Method3/AppB,C,G and the three supplied reviews were read. The implementation reproduces the specified token/expert depths and routing formula, with deviations explicitly listed. Synthetic full-resolution architecture checks passed for A0/A1/A2/A3 and the interaction-disabled control. Unimodal predictions are invariant to changes in the other input; detached routing gradients and absent vision weights outside FoV were checked. Independent SciPy tests verify the chirp and convolution, and reflection is involutive. These are code checks, not learned localization results.

Full Train/Val asset validation is running over every scheduled query in112 Train and19 Val scenes after family exclusions. Both image decoding and RIR finite/shape/checksum/pose checks are performed, rather than one waveform per scene. Test metadata was used only to exclude source families; no Test observation evaluation. Existing checkpoints/results and Track B are unchanged.

## Hypotheses and competing explanations

- Audio36-ray predictions may localize by geometry; alternatively direct-path or sensor-layout shortcuts may produce apparent accuracy. Diagnose ray error, fine-pose success conditions and geometry uniqueness.
- Joint attention may add information beyond independent V/A fusion; alternatively complementarity alone or extra capacity explains gains. Use independent fusion and matched-parameter interaction-disabled controls with compute reported.
- Sharp posterior may correlate with correctness; alternatively scale/temperature causes overconfidence. Separate entropy/margins, GT mass, NLL/ECE and held-development temperature calibration.
- FP-RIR matching may justify its storage/render cost; alternatively ray matching achieves similar accuracy without that bank. Measure both accuracy and deployment/training cost definitions.

## Results and failures so far

No trained A0–A3 checkpoint or benchmark table exists yet. Baseline/ablation numeric rows are pending, not zero or failed methods. The alternative SSD-root audit found no observation schedules (0 valid queries) and is unusable; it is preserved as a failed asset-location attempt. The authoritative network-backed HSSD audit remains running. SSD RGB availability does not imply the entire RIR dataset is mirrored.

## Remaining completion gates

1. Finish whole-cohort asset audit, source hashes, coordinate/ray-target tests and actual-data backward smoke.
2. Implement complete training/augmentation/checkpoint pipeline; train A0/A1/A2 sufficiently on all valid queries, validate before large A3 router optimization.
3. Train A3 and required independent/disabled/router controls with paired initialization and multiple seeds.
4. Evaluate UnLoc,cosine,fixed fusion,B2,B2+GEOQ,A0–A3 on the identical cohort/grid; report all requested pose/ray/coverage/repair-harm/scene metrics, and separate10cm grid.
5. Complete posterior comparison figures, acoustic-only observability, calibration/correlation and cost analysis, with failure cases and paired scene uncertainty.
6. Update this report with actual numbers and all required deliverables. Keep the active goal until requirement-by-requirement audit proves completion.

See [method](METHOD_AT_A_GLANCE.md), [paper/implementation audit](IMPLEMENTATION_AUDIT.md), and [assumptions](implementation_assumptions.md). Reproduction currently covers only the initial code tests; full experiment reproduction must be added with checkpoint/data provenance after runs.

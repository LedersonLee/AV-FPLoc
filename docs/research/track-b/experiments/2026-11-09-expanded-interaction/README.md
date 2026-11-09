# Expanded-cohort matched interaction replication

Study29,2026-11-09,Track B. Exploratory;9 scheduled fits running, no final outcome. Actual main source `c083c376631c3f6ed8d89daf9d3a9d93c350fd3b`. The expanded independent F1 in study27 has stronger acoustic/fusion performance; this experiment asks whether the prior interaction regression persists when initialized from that stronger full-Train baseline.

## Hypothesis and competing explanation

Weak pilot acoustic evidence or limited scene diversity may have contributed to interaction failure. Alternatively the current CE/score-scale/gate/residual learning still degrades MAP despite stronger precursors, as study28's fixed-model interventions suggest. This is an expanded-cohort replication with equal-update architecture controls, not a clean isolated data-size experiment: the historical16-scene interaction fits used GPU residency and only6 Val scenes, whereas this run uses CPU scene residency and19 Val. Compare the three architectures within this run and against their verified common initialization; do not attribute all old/new differences causally to scene count.

The112/budget1 precursor is chosen after observing study27. It retains the original1,500/1,000 precursor schedule and1,000 interaction updates; this is development-informed setup, not preregistered independent model selection. Both budget cells were already exposed. No Test use or independent evidence promotion follows. Remaining settings are fixed before the9-fit result; no epoch/seed/architecture selection within the batch.

## Fixed models, data and comparison

112 audited Train scenes,19 existing Val scenes,64 observations/scene, seeds42/43/44, .25m/36yaw full candidate bank. RGB uses frozen upstream DepthAnythingV2 features plus a trainable local adapter,6ch observed RIR uses the independent acoustic encoder, and visual/acoustic learned FP paths initialize separately. Attention/concat/separable each have2,834,908 local parameters; all local components train, upstream stays frozen. Same initialized tensors across architectures per seed, same1,000 sampling/optimizer updates, batch8, AdamW3e−4/weight decay1e−4,1,500 skipped draws,1000 visual auxiliary+10 acoustic auxiliary+1000 map-anchor weights. No Top20/B2/RIR reference-bank input.

Precursor final models are n112_budget1_visual_joint_attempt1 and n112_budget1_acoustic_attempt1 from study27; independent fixed F1 is n112_budget1_evaluation_v1. All three trainable architectures receive identical additional updates, but the initial F1 reference receives none. Therefore improvement over initial F1 alone would not separate extra optimization from interaction benefit. Concat/separable are the matched update/capacity controls; measured FLOPs/time can differ. No unearned matched-compute claim.

Normal, within-scene audio shuffle and separately re-extracted brightness0.2 inputs are retained. Brightness0.2 was weak for this backbone and is not claimed as severe degradation/night; physical/material/furniture and stronger occlusion robustness remain separate work. Expected9×19×64×3=32,832 final evaluation records, with own raw/diverse Top20. Primary Joint, R@1m/fine/coverage/NLL reported together, paired scenes with seeds averaged. All9 fits must complete before architecture/regression inference; no early stop from interim metrics.

## Implementation and executed validation

CPU residency stores raw cached scenes and initial acoustic FP anchors on CPU, transferring the requested scene for inference/training. Anchor computation stores into the underlying scene mapping so retrieval through SceneSequence preserves it. The old GPU default remains supported. Explicit precursor effective-cohort equality prevents a16-Train prefix head from being silently used as112-Train. Protocol records CPU residency; this is not approximate/pruned bank scoring.

43 pytest tests passed (2 warnings). A2-step pilot CPU run passed and its initial state matched archived pilot initialization. A separate112Train/19Val2-step run completed exit0 with all scenes/anchors loaded and branch gradients recorded; it is a runtime/memory check, not a performance result or a worst-bank backprop benchmark. Source d924448fac0954011fa8e0aad25149d14d140c07 implements residency; pilot smoke uses it; expanded smoke sourcea35d7134212eedc30724e1d0888a774e83f81b6e. CPU/GPU long-run bitwise equivalence is not asserted; earlier study27 numeric residency failures remain relevant and preserved.

Brightness-feature extraction completed19 scenes/1,216 queries. Every hash/query alignment/upstream source/preprocessing contract passed; overlapping6 pilot tensors match exactly. Verifier sourcea35d7134212eedc30724e1d0888a774e83f81b6e; feature-builder content unchanged from prior source, executed at0eaf47e740499a9351ca34746bcbd50b21d1880b. Frozen features are inputs, not HSSD-trained visual heads.

Before main launch, the saved seed42 attention initial state replayed all1,216 normal Val queries against the independent fixed baseline with zero mismatches in Joint/R@1m/fine/raw20/NLL at declared tolerance. This is a one-seed initial metric replay, not bitwise score equality or completed all-seed validation. Actual sourcec083c376631c3f6ed8d89daf9d3a9d93c350fd3b. Full-run initial states and final metrics require completion checks. No failures in these new smoke/verifier invocations; results are archived separately from main evidence.

## Reproduction, source and limits

External root /home/jeongeonlee/track1/experiments/track_b_expanded_interaction_20261109; main matched_v1, session39480. Dockeravfploc:cu124-clean, RTX4090,2 CPU threads, non-root/no network, source/input read-only. Command: `python -m track_b.scripts.train_foundation_interaction --cache <scale/cache_v1> --visual-run <scale/scaling_fits_v1/n112_budget1_visual_joint_attempt1> --acoustic-run <scale/scaling_fits_v1/n112_budget1_acoustic_attempt1> --visual-cache <scale/features_v1> --lowlight-cache <scale/features_lowlight_v1> --out <fresh-output> --scene-residency cpu --steps 1000 --seeds 42 43 44 --modes attention concat separable --code-sha c083c376631c3f6ed8d89daf9d3a9d93c350fd3b`. All actual hashes/settings are in main_protocol.json. Preserve partial artifacts; use resume only after authoritative terminal-state/protocol checks. No existing checkpoint overwritten. Weights/raw predictions remain outside Git.

No physical mechanism, attention superiority, scaling causality, independent Test result or overall completion is implied by setup. Reused development selection, uncertain upstream exposure, approximate map/source-family independence and limited input perturbations remain. Full scaled OOF candidate/backend training, costs and final independent evaluation remain unfinished. These models are full-bank scorers, not a replacement ranker on historical B2 candidates.

후단 연결 준비: [실제 M1 점수장 기반 특징 계산 검증](backend_preparation.md). 아직 M1 OOF/판별기 성능 결과가 아님.

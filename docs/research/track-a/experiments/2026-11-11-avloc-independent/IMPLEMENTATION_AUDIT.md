# Paper-to-code audit

Source: user-supplied AVLoc PDF, Method§3 (pp4–5), AppendixB/C (pp13–14), AppendixG (pp15–16), and the supplied OpenReview review PDF. Formula(1) was visually checked on the rendered PDF page. Submission: https://openreview.net/forum?id=Yi9xMxlOKe .

| Paper requirement | Independent implementation / evidence | State |
|---|---|---|
| ImageNet ResNet50 RGB,640×480 | Tokens(vision), pretrained-path loader; official V2 weight hash verified | structural test passed; trained use pending |
| 4ch ResNet18 binaural input | 18ch ResNet18, explicit6ch circular-array adaptation | implemented |
| Spatial/time-frequency tokens,d128,2 intra-modal layers | learned row/column/modality embeddings,2 transformer layers | shape test passed |
| V11 rays,2layers; A36/AV36,4layers,8heads | separate Expert modules and image-column seeding | shape test passed |
| Distance and Laplace scale | positive independent heads | tested finite |
|36 ray-wise routers, uncertainty detach, FoV mask | separate3-input or2-input MLPs | normalization/detach/zero-mask tests passed |
| Moment-matched Laplace | total variance includes expert disagreement | independent analytic test passed |
| Multi-expert NLL weights1/.1/.5/.5 | objective() | forward finite; training pending |
|48kHz,50ms20Hz–20kHz sweep,100ms echo | audio.py causal FFT convolution | matches independent SciPy calculation |
| STFT512/64/16 | full257×301 field, no pooling | shape/finite test passed |
| AppendixC corruptions and reflection | reflection primitive tested; augmentation runner pending | incomplete |
|≤100epochs,AdamW5e−4→1e−6,patience20,Val-L1 selection | frozen planned protocol | not trained |
| Global floorplan ray matching,10cm grid | primary matched20cm and separate10cm planned | incomplete |

Official-code availability: the supplied submission says code/benchmarks will be released upon publication and exposes no code URL. A current exact-ID/title web search found no result; live OpenReview/API requests returned access errors (403/API, page fetch failure). Therefore public-code availability is **not positively verified**. Proceed using an independent paper-based implementation and preserve this uncertainty; do not claim official code was reproduced.

Review issues informing controls: reviewer xnLp asks about sensor/sim-to-real gaps and split inconsistency; reviewer ZToi questions acquisition practicality and missing audio-to-ray detail; reviewer riEz highlights joint-only versus router increment, missing scale/error calibration, and out-of-FoV visual contamination through the joint expert. These are questions/criticisms, not empirical facts about HSSD. Phase4 explicitly tests complementarity, interaction and reliability separately.

All numerical benchmark claims remain pending. This audit does not equate architecture shape tests with a reproduced training result.


Update2026-11-11: full asset preparation, geometry targets/banks and the training pipeline are now implemented and code-checked. See the current [research report](RESEARCH_REPORT.md) and [assumptions](implementation_assumptions.md); any earlier pending implementation notes describe the initial audit snapshot. Learned benchmark results remain pending.

## Current-state reconciliation — 2026-11-17

The table above preserves the initial implementation audit; its pending statuses are historical. Production evidence now supersedes them:

- ImageNet initialization and architecture: [initialization checks](initialization_test.json), [architecture checks](architecture_test.json), pinned initialization and training sources in [method](METHOD_AT_A_GLANCE.md).
- Full-cohort data and augmentation/training: [asset audit](asset_audit_summary.json), [optimizer/resume tests](training_state_test.json), [documented assumptions](implementation_assumptions.md). Production A0/A1/A2/disabled/A3 completed100/70/99/69/87epochs under max100/patience20. See [all-arm accounting](completed_training_cost_all_seed42.json).
- Learned A3 routing and moment matching: [full9965query replay](A3_router_diagnostics.json); architecture normalization is not reliability evidence. See [interpretation and calibration](A3_ROUTER_AND_CALIBRATION.md).
- Native20/10cm floorplan matching: [A3 integrity](A3_integrity.json), [all requested metrics](REQUESTED_METRIC_TABLE.md), [earlier expert gate](expert_gate_seed42.json). Frozen geometry/pose contracts remain unchanged.

Official-code availability remains unverified beyond the original documented search/access result; this is an independent6ch adaptation, not an official implementation replication. All results are reused-development; uniform-router localization and final integrated audit remain pending.

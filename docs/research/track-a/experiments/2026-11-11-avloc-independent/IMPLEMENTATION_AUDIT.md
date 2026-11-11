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

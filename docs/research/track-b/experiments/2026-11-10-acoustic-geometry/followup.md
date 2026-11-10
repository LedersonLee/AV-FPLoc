# Transfer and physical intervention follow-up

2026-11-10, Study31, Track B. Inner-development evidence is Exploratory. All negative outcomes retained. The full study is active pending frozen held-out evaluation.

## Hypothesis and result

Geometry pretraining should improve localization beyond equal-capacity scratch training if the representation captures usable pose-dependent structure. The competing range-scale explanation remains more consistent with these runs: depth errors improve over a train prior, but angular shape reconstruction fails and geometry pretraining does not improve localization.

[Transfer table](transfer_results.md) and [full aggregate](transfer_summary.json): the unchanged full candidate matcher with fixed .85 visual/.15 acoustic fusion reaches joint1m30°38.558% with the legacy acoustic branch. Geometry frozen, fine-tuned and joint reach26.514%,26.204%,27.197%; identical-capacity scratch reaches27.669%, and total-update-matched scratch33.366%. Shared visual predictions reach24.235% and are identical in all arms. Fine0.1m and posterior NLL are retained, including degradations.

Family-paired frozen/fine-tuned/joint minus same-local-budget scratch joint effects are −1.156pp [−2.067,−.228], −1.465pp [−2.604,−.423], and −.472pp [−1.628,.602]. Against total-update-matched scratch the effects are −6.852pp [−8.480,−5.273], −7.161pp [−9.180,−5.339], and −6.169pp [−7.503,−4.850]. These intervals average seeds and source variants before10,000 source-family bootstrap draws; multiple contrasts are uncorrected.

## Controlled implementation and budget

95 training scenes,17 inner-val scenes/16 families,64 queries each, seeds42/43/44. All arms retain FoundationStructureMatcher map descriptors, score formula, complete candidate bank and fixed fusion. Only the acoustic branch is replaced. Geometry input uses the spatial regression encoder with a newly initialized Flatten2048→128→16 readout. The original36-direction decoder is used only by joint training. Frozen means encoder frozen/eval mode; readout and map remain trainable. No GT feature or candidate reference RIR enters inference.

Both visual and legacy branches use1500 old-ray warmup updates plus1000 localization updates. New scratch uses1000; scratch_total uses2500. Pretrained arms use1500 geometry updates plus1000 localization updates. All local updates reset the same seed-specific scene/query sampler. Initial map weights and new readout weights match across relevant arms; frozen encoder hashes are unchanged. Thus same-local and same-total-update controls are distinct; FLOPs and legacy capacity are not claimed equal. Batch8, AdamW lr3e-4/weight decay1e-4; acoustic old-ray MSE weight10, joint masked geometry loss weight1. Visual old-ray and map anchor weights1000. Visual trainable heads/maps are newly fit on95 scenes; frozen DepthAnythingV2 upstream asset exposure is not excluded.

21 fit training loops total262.35s, excluding cache and evaluation. [Raw replay](transfer_runtime.json) checks two predeclared inner-val scenes, first query,seed42,all6 arms: raw decoded RGB/body-aligned RIR reproduce features, full fields and Top20 bitwise. Median latency72.4–75.5ms includes RGB backbone and STFT, excludes file decode/model loading/map setup; map setup and memory/parameter counts are reported separately. Shared GPU,20 repetitions after2 warmups. The current spatial preprocessing helper also computes unused Echo features; that overhead is included. This is not a whole-dataset latency guarantee.

## Phase and structure strata

[Matched phase control](phase_contrast.json): removing spatial phase channels raises regression MAE by.04735m [.03470,.06318] and categorical MAE by.03742m [.02631,.04904]. Slightly useful range information does not establish directional reconstruction.

[Strata](strata_summary.json) retain room polygon labels, unknown/ambiguous labels, near-wall<1m, angular span≥4m and valid-target coverage<75%. These are descriptive diagnostic cuts, not model selection. Both MAE and RMSE are query-averaged before scene/family aggregation, so they need not match primary pooled-ray metrics. Spatial regression MAE is.708m at angular span<4m versus1.654m at≥4m;1.970m at coverage<75% versus1.378m otherwise. Room-label errors include bathroom.766m,bedroom1.075m,kitchen1.508m,living1.731m,hallway1.477m. Different strata have different scenes and geometry; these are associations, not causal room-type effects. Sparse strata with fewer than3 families suppress intervals.

## Physical controls and limitations

See [paired results](pair_results.md).16 inner-val families,8 predeclared uniform poses per family, exact matched source/receiver/heading. Fresh furnished, native stage-only, and independent fresh-simulator furnished repeats total272 waveforms. Native stage-only removes **all** scene objects, including doors/windows stored outside the stage. It must not be described as a pure furniture intervention. A further128-waveform control preserves original transforms for every object whose template registry is in `objects/openings`; other objects, including possible fixed fixtures, are removed. This is a folder-classified architectural-opening control, not a guarantee of movable-furniture-only removal. No paired consistency training is introduced after model freeze.

Renderer settings:48kHz,6 omni microphones at5cm radius, centre source,20000 indirect rays,depth50,512 source rays,diffraction max order10,default materials,transmission disabled. No pose snapping. Native stage/config/helper hashes and loaded object counts are checked. Fresh repeats are not bit-identical, so waveform change alone cannot identify furniture effects. Furnished poses were originally selected under furnished navigability; results are conditional on that selection. No real-world invariance claim is justified.

## Provenance, failures and pending evaluation

Transfer training `f6e9d922df2b5df31c976a6da363fb1193ee4920`; summary `334695c1d7787e64c28fb475de9659c499dedf96`; runtime `c55c0ae93f7a72517765fbda3364fc084f0ad3c0`; strata `d992c2cdf811c4da898eed19b0579dddfac5adee`; phase contrast `b241e0a50d03f7b3aeb94e7ae30ede63aa5abafc`. Native pairs source `16ec5114d323dbd56decc5e7687da639ecaa470c`; opening renderer `8ee29be225f83f9a0786e08201fefa9379eeda98`. Raw data/checkpoints remain external under the study output root. Code base has two unrelated dirty Gibson files excluded from every commit; none is imported here.

Failed CPU smoke `pair_smoke_v1` stopped at image entrypoint user setup before Python. An earlier Habitat import without the established libstdc++ preload failed with invalid free; the corrected CPU launcher passed. `pair_smoke_v2` and `opening_smoke_v1` passed. The one-scene pair evaluator smoke had an overly broad evidence label, corrected to dynamic cohort counts before the full run. No failed artifacts were overwritten.

[Method freeze](method_freeze.json) records39 actual checkpoint hashes before held-out performance.19 historical development scenes retain previous upstream/study exposure. Official17 Test families have zero overlap with the development split, but3 prior observed directories and earlier visual Test use/upstream Hypersim asset-family exposure are not fully excluded. Fresh acoustic rendering does not erase those facts. Evaluate every fixed arm/seed, without adjusting epochs, architecture or thresholds. Current completion is not claimed.

Geometry-conditioned cross-modal matching remains a possible research direction, not a conclusion compelled by these results. The present bottleneck could be target masking, weak angular identifiability, frontend/architecture or learning budget; the failed transfer alone cannot identify which. A subsequent architecture study should require angular countercontrols and equal-budget localization gains before claiming useful structural matching.

# Completed matched-model grid sensitivity

2026-11-09. Track B, Exploratory. All nine scheduled joint foundation checkpoints completed the same six frozen grid/mask variants used by the independent-F1 study:20,736 cases, exit0. The3,456 baseline-grid cases reproduce prior Joint/R@1m/Fine/raw20/NLL within the declared tolerance, zero mismatches. No retraining, temperature fitting, grid selection or Test access.

## Protocol

Use identical hashed map banks and the same six reused Val scenes/64 queries per scene, seeds42/43/44, attention/concat/separable. Conditions: baseline.25m/36yaw; .20m and.50m with36yaw;12/72yaw with.25m; .1m clearance erosion with.25m/36yaw. Map geometry used for rays is unchanged by the erosion validity mask. All banks are exhaustive, with no GT filtering/clamping. Observation batch8 preserves the prior numeric replay contract. Model/map descriptors are recomputed for each supplied bank; source weights remain fixed.

GT validity is defined by the nearest raster cell in each declared bank. All queries are valid under that specific criterion; all-query and valid-subset metrics are both retained. This does not prove exact continuous physical inclusion or eliminate 3D/2D mismatch. The scientific competing explanation is that a grid changes candidate density and quantization, giving apparent recall gains without better evidence or MAP decisions.

## Results

| Grid | Attention Joint % | Concat Joint % | Separable Joint % | Attention Fine % | Attention raw20 % | Attention diverse20 % |
|---|---:|---:|---:|---:|---:|---:|
|.25m/36 baseline|16.753|18.229|15.538|.694|49.826|76.649|
|.20m/36|16.927|18.056|16.667|1.476|48.090|76.389|
|.50m/36|16.233|15.972|13.281|1.042|57.378|74.653|
|.25m/12|15.538|16.667|13.281|.955|52.083|76.128|
|.25m/72|17.188|17.969|16.146|.868|46.007|77.604|
|.25m/36, erosion.1m|17.014|18.316|15.538|.694|49.653|76.823|

Attention .20m−baseline Joint+.174pp,95% six-scene paired bootstrap CI[−1.215,1.128].72yaw−baseline+.434pp CI[−.694,1.302]. Erosion+.260pp CI[−.174,.694]. None establishes a reliable Joint improvement. .50m raw20 rises7.552pp CI[5.903,9.028], while Joint changes−.521pp CI[−2.257,1.215] and diverse20 falls. The stronger raw20 on a coarser bank therefore cannot be called better localization.

Compared **within the same grid** to independentF1, attention Joint deficits are7.118/7.465/6.771/7.031/6.858/7.031pp in table order. All corresponding95% scene intervals are strictly below zero. Indeed all18 mode/grid Joint comparisons have negative means and negative upper CI bounds. For example attention.20m−F1−7.465pp CI[−9.896,−4.948]; baseline−7.118pp CI[−10.590,−3.646]. Query/grid keys and grid manifest hashes are exactly matched before these comparisons. The multiple exploratory intervals are not multiplicity-corrected and the six reused scenes remain the uncertainty units.

Fine success is low across all methods and grids. The finer bank expands the quantization ceiling, but may not improve a badly ranked field. Conversely the coarse bank's Fine oracle drops as documented in the original F1 study; a small observed Fine increase from an already poor baseline does not overcome that ceiling or imply greater metric precision. These distinctions prevent choosing raw20 alone as an optimization target.

## Metrics, interpretation and decision

Full R@1m, raw5/10, strict(.5m,15°)/(.2m,10°), oracle Joint/Fine, nearest-grid NLL, log(bank-size)−NLL, normalized entropy and reliability/selective-risk summaries are in matched_summary.json and matched_reliability.json. NLL across different bank cardinalities is not directly comparable without its uniform reference. ECE is a diagnostic of the defined MAP-neighborhood mass, not a continuous-pose guarantee. Per-query rows remain external. No cost timings were instrumented in this evaluation; the registry marks GPU-hours NA.

**Observation:** the negative joint-versus-independent comparison persists across every tested grid. Raw candidate coverage is sensitive to candidate density; its changes do not track MAP success monotonically.

**Interpretation:** the baseline failure is not rescued by these resolution/valid-mask changes. This does not identify whether objective, learned representation, optimizer or architectural inductive bias is responsible. It also does not imply that all possible grid designs are equivalent.

**Decision:** retain the original.25m/36 primary protocol and all alternative results as sensitivity controls; do not select the best observed grid for an independent claim. Continue the already frozen encoder-regime control and expanded-development preparation. Physical map degradation, final selected-regime audit, deployment costs and independent evaluation remain separate work.

## Provenance and reproduction

Evaluation source `09fddf31d86c02f61551beb68678ed8721f25fbf`; registry/comparison code `0016785503fb4463f0a3e926de6da5e21724a576`. Concurrent unrelated Track A commits entered the shared checkout after these sources; evaluated Track B files were unchanged. Original training sources and checkpoint hashes are recorded in study21 and matched_protocol.json. No training or source artifact was overwritten.

External root `/home/jeongeonlee/track1/experiments/track_b_interaction_20261109`, outputfoundation_grids_v1 and registryfoundation_grids_registered_v1. Grid source `/home/jeongeonlee/track1/experiments/track_b_grid_20261109/cache_v1`; independent referenceindependent_v2.54 model/seed/grid evaluation rows are in matched_experiment_registry.csv; prediction and summary hashes are in matched_external_hashes.json. Geometry cache is FP-derived, not an RIR reference bank.

Executed `python -m track_b.scripts.evaluate_interaction_grids --run /results/foundation_matched_v1 --cache /structure/cache --grid-cache /grids/cache_v1 --features /foundation/features_v1 --out /results/foundation_grids_v1 --code-sha 09fddf31d86c02f61551beb68678ed8721f25fbf`; then `python -m track_b.scripts.register_interaction_grids --run <foundation_grids_v1> --reference <independent_v2> --out <fresh-directory>`. Both exit0. Dockeravfploc:cu124-clean, RTX4090, non-root UID, no network, read-only code/input caches; shared authorized training means no isolated latency claims. Existing source/feature checkpoints are required for reproduction.

The original F1 grid result and failed first numeric batch-size replay are preserved in this study; this follow-up adds matched-model coverage without replacing that history. Whole Track B remains unfinished.

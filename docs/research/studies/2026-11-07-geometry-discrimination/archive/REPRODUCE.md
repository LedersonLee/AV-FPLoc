> 역사적 연구 기록. 원본: `structural_evidence_20261107_1730/REPRODUCE.md`. 원본 SHA256은 [산출물 목록](../artifacts.json)에 있습니다.
> 당시의 완료·재개 지시는 현재 실행 요청이 아닙니다. 최신 판정은 [현재 결론](../../../current_findings.md)을 먼저 확인하세요.

# Reproduction and frozen inference

Workspace `${WORKSPACE}`; this directory `experiments/structural_evidence_20261107_1730`.
Docker image `avfploc:cu124-clean`, immutable image SHA256 `e51cb712100f25d0018aa65e371e63deaa51d12601c465ce698c0c446800c318`.
Python/Torch from image; sklearn1.5.2 from existing `experiments/astra_fusion_v2_20261107_0225/deps`. CPU numerical libraries use2threads. Host analysis uses Python3/numpy/matplotlib. `run_script.sh` uses read-only source/data mounts and a writable output mount; no network. GPU allocation respects existing user services.

## Frozen method

Use unchanged B2 and support/refinement to produce the original ordered20candidate poses and S26. Read each candidate's11expected floorplan rays and the adapted query11rays (repeat across20candidates). `inference.rank(model, S, candidate_rays, threshold, observed_rays)` accepts only these features, no truth or scene ID. For GEO omit observed_rays. Full definitions in inference.py and modes.py from previous hypothesis_discrimination run.

- Final seed42: `GEOQ_seed42/model.joblib`, threshold0.
- Final seed43: `GEOQ_seed43/model.joblib`, threshold.01.
- These are separately evaluated seeds, not an ensemble. Selection metadata/hashes are in final_selection_seed42/43.json.
- Research control GEO uses `GEO_seed42/43/model.joblib`, prior threshold.025/.1.
- Verify API exact prediction replay: `bash experiments/structural_evidence_20261107_1730/run_script.sh verify_selected_inference.py`.

## Recompute derived analyses

From workspace root:

```bash
python3 experiments/structural_evidence_20261107_1730/final_evaluation.py
python3 experiments/structural_evidence_20261107_1730/analyze_b1.py
bash experiments/structural_evidence_20261107_1730/run_script.sh case_attribution.py
python3 experiments/structural_evidence_20261107_1730/plot_results.py
python3 experiments/structural_evidence_20261107_1730/write_report.py
```

The bootstrap uses RNG319,10,000scene draws. Never treat repeated seeds as independent queries. CI is exploratory and unadjusted for multiple research comparisons. Strict fine radius is0.1; epsilon diagnostic is separate. Actual saved predictions and models are authoritative.

## Training/extraction dependencies

Scripts are idempotent where completed receipts/results exist. To recreate from scratch, use a separate copied experiment directory, point ASTRA_FUSION_OUTPUT_DIR there, retain required provenance/scripts, and remove only the derived artifacts to regenerate in that copy. Some sources point to prior experiments intentionally; do not delete or overwrite those. All input paths/receipt hashes are recorded. Models are trained on Train12 only. Development observations are evaluated but not passed to fit.

1. `features.py --seed 42` and `features.py --seed 43`: fixed B2 candidates/features. `audit_inputs.py` documents source availability; input_audit.json is the original audit receipt.
2. `train.py --seed 42` /43: B0,V,AC,AM,VA,VAI,INTER,NOISE,VA_abs.
3. `compact.py`, `compact_controls.py`, `observation_controls.py`, `geometry_controls.py`, `geometry_interaction.py`, `constrained_interaction.py`: each with `--seed 42` and43. These install feature/model wrappers around train.py; do not substitute train.py ARMS without the wrapper.
4. `signal_features.py`, then `train_signal.py --seed 42/43`; `verify_signal.py` checks mic-roll and encoder replay. Signal source pooling requires existing E/C/envelope inputs, not new RIR rendering.
5. `decode_geometry.py`, then `train_decoded.py --seed 42/43`: Train-scene-fold Ridge log-ray probe and ranker. Decoder targets are map geometry, not query pose GT.
6. `nested_features.py --seed 42/43`:72outer-training caches. `check_nested_features.py` must pass before `nested_fit.py --seed 42/43`.
7. `select_nested.py --seed 42/43`: selects from completed12arms/thresholds using Train OOF only, strongerB0 constraints. Saves model checksum. No validation performance is consulted by this script. Design choices still arose during exploratory research.
8. `matched_b1.py --seed 42/43`: same training/config/initialization audited B1/B2 checkpoints, fixedset rank and own support coverage. No new upstream training is needed for this matched comparison.
9. Recompute analyses and inference verification above, then final_verification.py last. It verifies previous manifests, unchanged source diff, candidate identity and current artifact receipts before generating the new manifest. Inspect report and PNG.

All script arguments can be checked in their `if __name__` sections before launching. Existing logs preserve both negative and positive attempts and runtime. Final wall times are not summed into GPU-hours: jobs ran concurrently and much work was CPU-based.

## Scope that remains untested

No Test observation/result is read; Val8/Val11 are reused development data. No new independent scene, fully nested encoder retraining, raw-waveform model, physical wall/reflection matching, real/sim matched intervention, or end-to-end modality-removal retraining is claimed. New independent confirmation should use this frozen selection protocol before inspecting its outcomes.

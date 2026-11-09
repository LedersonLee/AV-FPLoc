> 역사적 연구 기록. 원본: `hypothesis_discrimination_20261107_1538/reproduce.md`. 원본 SHA256은 [산출물 목록](../artifacts.json)에 있습니다.
> 당시의 완료·재개 지시는 현재 실행 요청이 아닙니다. 최신 판정은 [현재 결론](../../../../current_findings.md)을 먼저 확인하세요.

# Reproduction and inference

All results are development evidence; Val8 and Val11 were previously used. No generalization or sealed-Test claim.

Work in `${WORKSPACE}/experiments/hypothesis_discrimination_20261107_1538`.
`run_script.sh` uses `avfploc:cu124-clean` (sha256:e51cb712100f25d0018aa65e371e63deaa51d12601c465ce698c0c446800c318),4CPUs,8GiB RAM/container and a45%CUDA allocator ceiling. These are limits, not measured peaks. Source/data read-only, output directory writable, network disabled. Torch2.4.1+cu124, prior v2 local sklearn1.5.2/joblib1.4.2/threadpoolctl3.5.0 dependencies are preserved.

Inputs: frozen B2 adapted outputs from earlier experiments; full candidate banks and features from mbt_remaining_suite_20261106/data and fusion_vs_posterior_20261107_1300/independent_data. Three missing seed43 Train adapted outputs regenerated from original held-scene checkpoint with unchanged batch32. Existing datasets/encoders/checkpoints are never overwritten.

Executed experiment order:

```bash
bash run_script.sh prepare.py --seed 42 --split train
bash run_script.sh prepare.py --seed 42 --split val
python3 audit_extraction.py
bash run_script.sh rank.py --seed 42
bash run_script.sh joint_selection.py
bash run_script.sh pair_rank.py
bash run_script.sh utility_rank.py
bash run_script.sh prepare_v2.py --seed 43 --split train
bash run_script.sh prepare_v2.py --seed 43 --split val
python3 audit_extraction43.py
bash run_script.sh utility_rank_v2.py --seed 43
bash run_script.sh ablate.py
bash run_script.sh prepare_extra.py --seed 42 --split extra
bash run_script.sh eval_extra.py --seed 42
bash run_script.sh prepare_extra.py --seed 43 --split extra
bash run_script.sh eval_extra.py --seed 43
bash run_script.sh ablate_v2.py --seed 43
bash run_script.sh cross_extract.py --seed 42
bash run_script.sh cross_extract.py --seed 43
bash run_script.sh explain_cases.py
bash run_script.sh verify_inference.py
bash run_script.sh analyze.py
bash run_script.sh plot.py
bash run_script.sh report.py
```

Commands show dependencies, not an instruction to overwrite completed folders. Some writers require a new output directory. Use ASTRA_FUSION_OUTPUT_DIR for a new run, copy needed configs and caches, and preserve completed model outputs. No continued job is needed once the completion audit passes.

Final choice files/model assets are `utility_ranker_support_nms_seed42/` and `utility_ranker_support_nms_seed43/`. `frozen_seed42.json` and43 pin model/selection/feature-code hashes before reusedVal11 evaluation. Same utility weight.5, thresholds selected on each seed's Train OOF (.025 and.1). Do not apply those models to arbitrary unversioned feature arrays.

GT-free API:

```python
from inference import rank_hypotheses
# model is the trusted, frozen joblib regressor.
# features[Q,20,26] follow modes.NAMES; hypothesis0 is support/refinement baseline.
# indices[Q,20] index the original full candidate bank.
result = rank_hypotheses(features, indices, model, threshold)
pose_indices = result['indices']
```

Prediction values are expected utility gains, not probabilities. Input features must be produced without GT. `modes.extract` contains the three extraction paths and shared feature generation; it currently relies on local research helpers. Standalone deployment packaging is not claimed. Geometry caches pin the floorplan raster/metadata and candidate ordering; verify hashes before reusing after any dataset modification.

Timing separates precomputed-feature ranking from three-extractor/feature costs; encoder and score construction are excluded. Single-extractor optimized deployment cost was not measured. Original failed models, precision-tradeoff findings, logs, per-scene predictions, oracle diagnostics and model selection tables remain available.

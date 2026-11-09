> 역사적 연구 기록. 원본: `modality_coverage_20261107_1700/reproduce.md`. 원본 SHA256은 [산출물 목록](../artifacts.json)에 있습니다.
> 당시의 완료·재개 지시는 현재 실행 요청이 아닙니다. 최신 판정은 [현재 결론](../../../../current_findings.md)을 먼저 확인하세요.

# Reproduction

Run from `${WORKSPACE}`. Exact Docker image and source configuration hashes are in environment.json. Source checkpoints, query/FP caches, and prior experiment outputs are mounted read-only; only this experiment directory is writable. No external data download or Test inference. GPU is used for batched original/acoustic scores and frozen B2 interventions. NMS/statistics run on CPU.

```
for seed in 42 43; do
  for split in train val extra; do
    bash experiments/modality_coverage_20261107_1700/run_script.sh coverage_v2.py --seed "$seed" --split "$split"
    bash experiments/modality_coverage_20261107_1700/run_script.sh intervene.py --seed "$seed" --split "$split"
  done
done
bash experiments/modality_coverage_20261107_1700/run_script.sh analyze.py
bash experiments/modality_coverage_20261107_1700/run_script.sh diagnose.py
bash experiments/modality_coverage_20261107_1700/run_script.sh ambiguity_control.py
for seed in 42 43; do
  bash experiments/modality_coverage_20261107_1700/run_script.sh ordinary.py --seed "$seed"
done
bash experiments/modality_coverage_20261107_1700/run_script.sh ordinary_analysis.py
python3 experiments/modality_coverage_20261107_1700/boundary_audit.py
python3 experiments/modality_coverage_20261107_1700/cases.py
python3 experiments/modality_coverage_20261107_1700/plot.py
python3 experiments/modality_coverage_20261107_1700/report.py
python3 experiments/modality_coverage_20261107_1700/verify.py --final
```

Existing scene outputs are hash checked and skipped. To recompute in a fresh location, copy these scripts there and set ASTRA_FUSION_OUTPUT_DIR to that absolute directory; run_script.sh then mounts it at /output. Plot/report/verify derive paths from their script location; verify expects an experiment sibling under this repository.

coverage_v2.py changes only full-bank oracle evaluation memory use: Cartesian XY/yaw reachability is factored instead of materializing query×all-pose×XY arrays. The first training worker using v1 terminated before its second scene output; v2 resumed from verified completed outputs. All original v1 outputs remain valid; final verification compares the oracle against explicit pose enumeration on samples for every bank/scene/seed. v1 remains archived for code-hash provenance. Use v2 for new runs.

All randomness is pinned: scene numeric ID + model seed for100random/20diverse proposal pools and10draws/query; paired scene bootstrap seed101,10000 draws. Pure branch outputs are deterministic and common to model seeds; seed averaging does not turn them into independent replicates. Train12 gives ambiguity tertiles; same bins then apply to reused Val8/Val11. Full thresholds and all subgroup sample counts are saved in summary.json.

GT enters retrospective metrics, coverage, mass concentration, and subgroup descriptions only. Candidate extraction, random controls, donor choice, frozen interventions and existing utility-ranker inference are GT-free. No new model or threshold selected on development outcomes.

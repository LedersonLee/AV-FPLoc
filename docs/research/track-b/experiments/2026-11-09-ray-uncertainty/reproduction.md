# 재현

외부 root `/home/jeongeonlee/track1/experiments/track_b_uncertainty_20261109`. 입력 root는 protocol/input_validation에 명시한 Study27 root. 원본 checkpoint, 캐시, 대형 평가 파일은 Git에 넣지 않는다.

```bash
python3 -m track_b.scripts.audit_uncertainty_inputs --root <scale-root> --out <new-audit-dir> --code-sha 96db20f075ded8f5c66234496c7b2dfbfb3741cf
python -m track_b.scripts.independent_fusion --cache <scale-root>/cache_v1 --run <scale-root>/scaling_fits_v1/n112_budget1_acoustic_attempt1 --visual-run <scale-root>/scaling_fits_v1/n112_budget1_visual_joint_attempt1 --visual-cache <scale-root>/features_v1 --out <new-replay-dir> --fixed-only --extended-metrics
```

감사는 host Python3 stdlib, GPU replay는 avfploc:cu124-clean, OMP_NUM_THREADS2, PYTHONPATH=/work, 입력/코드 read-only mounts, network none. 출력 `preflight_v1` 완료, `u0_replay_v1` 진행 중. 원래6개 model 학습 source는5cf803681c2d144f3a620bc00c1ab6b841e48e7c이며 각 checksum은input_validation에 기록했다. 새 uncertainty 학습 source는 아직 없다.

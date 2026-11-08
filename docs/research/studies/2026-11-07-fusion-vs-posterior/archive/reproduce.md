> 역사적 연구 기록. 원본: `fusion_vs_posterior_20261107_1300/reproduce.md`. 원본 SHA256은 [산출물 목록](../artifacts.json)에 있습니다.
> 당시의 완료·재개 지시는 현재 실행 요청이 아닙니다. 최신 판정은 [현재 결론](../../../current_findings.md)을 먼저 확인하세요.

# Reproduction

Workspace `${WORKSPACE}`, output `experiments/fusion_vs_posterior_20261107_1300`.
Container image `avfploc:cu124-clean`, content ID `sha256:e51cb712100f25d0018aa65e371e63deaa51d12601c465ce698c0c446800c318`.
`run_script.sh` mounts source/data read-only and only this output directory writable. Four CPUs,8GiB RAM/container, CUDA allocator ceiling45%/process. These are allocation limits, not measured resource peaks. No network. Existing user servers retained.

Dependencies: Torch2.4.1+cu124 and image SciPy/NumPy; sklearn1.5.2/joblib1.4.2/threadpoolctl3.5.0 from prior v2/deps read-only. Do not replace original checkpoints with full116 acoustic models. Parent Test remains sealed.

Execution order (logs and per-scene receipts preserve the actual runs):

```bash
bash run_script.sh study.py prepare --seed 43
bash run_script.sh study.py prepare --seed 42 --split extra_val
bash run_script.sh study.py prepare --seed 43 --split extra_val
bash run_script.sh study.py fit --seed 43 --kind tree
bash run_script.sh factorial.py --seed 42
bash run_script.sh factorial.py --seed 43
bash run_script.sh analyze.py
bash run_script.sh calibration.py
bash run_script.sh decoder_labels.py --seed 42
bash run_script.sh decoder_labels.py --seed 43
bash run_script.sh weight_frontier.py --seed 42
bash run_script.sh weight_frontier.py --seed 43
bash run_script.sh independent_prepare.py --smoke
bash run_script.sh independent_prepare.py
bash run_script.sh confirm.py --seed 42 --development
bash run_script.sh confirm.py --seed 43 --development
python3 verify_confirmation.py
bash run_script.sh evaluate_confirmation.py --prefix confirmation_dev
bash run_script.sh confirm.py --seed 42
bash run_script.sh confirm.py --seed 43
bash run_script.sh evaluate_confirmation.py
bash run_script.sh supplement.py
```

Seed42 original MAP gate and Train+Val3 feature caches are reused from prior v2 with checksum verification; new cache directory began as a byte copy. Cached adapted B2 outputs preserve old inference batch32, audio feature extraction uses64, vision4. Small batch changes can change acoustic convolution numerics; the64-query smoke is exactly equal in all six tensors. Original4-query failed smoke is retained and explained in PROTOCOL.md.

These commands describe dependencies; do not run blindly over finished folders. Some writers require fresh output directories; others are resumable. For fresh reruns use ASTRA_FUSION_OUTPUT_DIR, preserving the copied seed42 caches, dependencies, frozen selection and hashes. Old source directories remain dependencies, not replaceable placeholders. `confirmation_frozen.json` pins the final confirmation algorithm, gates, scenes and calibrated temperature. Once Val11 outcomes are opened, changes cannot be called untouched confirmation.

Val11 is held out from this fusion/decoder research only. Existing visual encoder validation included all19Val scenes, including LR scheduling; see validation_scope_audit.json. No inference script uses diagnostic GT to choose a weight or a hypothesis. Diagnostic oracles and posthoc subgroups are kept separate from prediction.

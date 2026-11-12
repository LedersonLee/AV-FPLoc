# Frozen inference runtime protocol and integration checks

Date: 2026-11-13 KST (2026-11-12 UTC). Track A. Evidence: Exploratory implementation checks; **no completed latency comparison**.

The cost question is whether removing FP-RIR reference matching reduces deployment expense at comparable accuracy. Storage and completed training costs are already recorded in [training cost accounting](COMPLETED_TRAINING_COST.md). Timing must separate input processing, network computation, reference scoring and pose selection. Accuracy alone does not answer this question; implementation differences, hardware contention and differing candidate counts are competing explanations for apparent speed differences.

## Measurement contract

`benchmark_ray_runtime.py` uses naturally completed best checkpoints and their existing full-Val prediction manifests. It samples one query uniformly from each of the19 Val scenes using seed20261111, without inspecting correctness. Both20cm and10cm retain10° yaw. Each query uses batch1, three warmups and ten measured repeats by default. GPU operations are synchronized at measurement boundaries. CPU matching uses the existing float64 Laplace field and stable Top20 NMS, with one PyTorch CPU thread; the launch must also set BLAS/OMP threads to1.

Stages are: existing `prepare()` including input transfer and spectrogram computation; model forward; output transfer to CPU; complete ray score field; Top20 NMS. Disk decoding, checkpoint/bank initialization, hashes and diagnostic evaluation are outside the timing windows. Summed stages are synchronized warm latency, not pipeline throughput or full cold-start latency. `prepare()` currently computes both input modalities even for A0/A1; this overhead is retained and explicitly reported rather than presented as an optimized single-modality system.

Every repeat compares predicted means/scales against the saved float32 prediction (rtol2e-4, atol2e-4), then requires exact Top1 and Top20 candidate IDs. A mismatch fails the run rather than silently timing a different prediction. Manifests, checkpoints, inputs and executable source hashes are checked. The script records environment, CPU affinity, load average, repeated times and scene sizes; it cannot itself establish exclusive hardware use.

## Observations and failure record

One predetermined real query (scene102344115, query1587) passed CUDA integration for A0/A1/A2. Maximum mean/scale differences from stored batch predictions were1.669e-6,9.537e-7,1.431e-6 respectively. All three reproduced Top1 and every Top20 candidate ID. These checks establish integration for this query only; A3 and disabled checkpoints are still pending. [Machine-readable evidence](runtime_integration_checks.json).

The initial A2 **CPU** integration failed the unchanged prediction tolerance: maximum absolute difference0.00307512 and maximum relative difference0.0008387. This occurred before candidate validation. The source of CPU/CUDA disagreement has not been isolated; it is not evidence of a training or saved-evaluation error. The failed output is preserved locally as `runtime/ray_A2_cpu_smoke_v1/failure.json`. No tolerance was relaxed and no CPU timing result is accepted.

CUDA smoke runs overlapped existing training, so their durations are deliberately excluded from published performance claims. Formal timings remain pending until the training/evaluation queue has completed and hardware use has been inspected. Per-scene medians and the distribution across scenes should accompany pooled timing; repeated queries are not independent scene evidence.

## Reproduction and remaining work

Runtime source commit: `a4463e9ddebd1596018860a199946a7530b89521`. Training source: `62c58da20ed26ba95863638bbfd4926829999eec`. Runtime image: `sha256:e51cb712100f25d0018aa65e371e63deaa51d12601c465ce698c0c446800c318`. Use the existing source at `/source`, local SSD study at `/study`, CUDA access, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, and `OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1`.

```bash
python /source/benchmark_ray_runtime.py --study /study \
  --out /study/runtime/A2_cuda_20cm_full_v1 --arm A2 --device cuda \
  --grid-cm 20 --warmups 3 --repeats 10
```

Use a fresh output path for each arm/grid. Append `--smoke --warmups 1 --repeats 1` only for integration checks. No training, FP-RIR rendering or Test observation access occurs. The existing cached baseline field timings do not include original image/audio encoders or GEOQ selection. Those stages require their own matched benchmark; this new ray-side tool does **not** complete the end-to-end B2+GEOQ cost comparison. Cross-method speed or generalization conclusions remain unverified.

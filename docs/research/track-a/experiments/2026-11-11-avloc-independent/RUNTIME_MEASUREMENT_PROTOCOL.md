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

## Follow-up: controlled CPU/CUDA backend audit

A completed four-way intervention on the same predetermined query separates preprocessing device from model device. Both models load the identical frozen A2 checkpoint; no weights or thresholds change. Source `9f07c1085df63a6fa8753b1bcf062cbb467920e8`. [Detailed evidence](runtime_backend_audit.json).

| Preprocessing | Model | Maximum mean difference from saved GPU prediction (m) | Maximum scale difference (m) | Top1 / full Top20 IDs |
|---|---|---:|---:|---|
| CPU | CPU | 0.00307512 | 0.00369477 | Exact / exact |
| CPU | CUDA | 0.00043273 | 0.00373548 | Exact / exact |
| CUDA | CPU | 0.00308561 | 0.00281560 | Exact / exact |
| CUDA | CUDA | 0.00000143 | 0.00000036 | Exact / exact |

**Observed:** changing only the model device with CUDA-prepared inputs changes ray means by up to0.00308609m and scales by0.00281525m. Changing only preprocessing with the CUDA model changes means by0.00043201m and scales by0.00373536m. Thus the CPU discrepancy is not attributable solely to audio preprocessing. These effects need not add linearly.

Additional localization checks found CPU/CUDA chirp max/mean absolute waveform differences0.00073588/0.00001099, echo differences0.00070892/0.00001561, and spectrogram differences on an identical GPU-generated echo of max2.0/mean0.00023081. These are diagnostic numerical differences, not error tolerances or acoustic quality scores. Which individual kernels cause them, and whether low-energy phase bins explain the maximum, remain unisolated.

**Interpretation and limits:** both preprocessing and model execution can affect backend portability in this implementation. All four combinations preserve the final candidates on this one query, but no full-cohort equivalence or accuracy conclusion follows. The original failed CPU check remains a failure of the predefined prediction tolerance. This audit does not relax it or replace the established CUDA predictions. A general deployment-portability claim would require full-cohort evaluation with the exact intended backend. No retraining or new rendering is justified by this narrow diagnostic alone.

Reproduce inside the same pinned runtime with CUDA access and documented deterministic settings:

```bash
python /source/audit_runtime_backend.py --study /study --out /study/runtime/backend_A2_fresh
```

## Cached baseline stages prepared — 2026-11-13

`benchmark_cached_baseline.py` (executed code `3785879`) uses the same uniform-per-scene query selection and20cm candidate grid as the ray runtime tool. It separately times original visual, acoustic cosine, fixed fusion and adapted B2 fields, followed by score-based Top20 NMS. A separate GEOQ stage times geometry summarization and utility prediction from the **historical cached support/refinement candidate bank**. The GEOQ stage is not fed the score-NMS outputs, so adding those stages would omit work and misrepresent B2+GEOQ end-to-end latency.

The first predetermined query (scene102344115/query1587) passes exact Top1/Top20 comparisons for all four fields and GEOQ selected pose/gain replay (gain atol1e-10). [Integration evidence](cached_runtime_integration.json). This is an integration test, not the full19scene timing experiment. Initial host startup lacked joblib; the pinned model-training image then lacked sklearn. Both failures occurred before any measurement output. The serialized GEOQ model records sklearn1.5.2, so matching scikit-learn1.5.2 and threadpoolctl3.5.0 were installed into a separate local runtime dependency directory; the training container/image and model files were unchanged. The successful test uses the pinned image with that directory on PYTHONPATH and OMP/OpenBLAS/MKL threads1. Actual library versions/thread pools are recorded. No raw features or model files are published.

Reproduce in the pinned image with the repository workspace mounted read-only, study mounted at `/study`, and the isolated dependency directory on PYTHONPATH:

```bash
python <workspace>/AV-FPLoc/scripts/research/track_a/avloc_20261111/benchmark_cached_baseline.py   --workspace <workspace> --study /study --out /study/runtime/cached_baseline_fresh   --warmups 3 --repeats 10
```

Use `--smoke --warmups 1 --repeats 1` for the integration-only scope. Formal timings remain deferred until active training/evaluation finishes. Original RGB/RIR encoder, B2 feature adaptation, support/refinement extraction, FP-RIR preparation and disk/cold-start costs remain excluded and must be measured or explicitly left unmeasured in the final cost comparison. This tool does not justify an end-to-end speed claim.

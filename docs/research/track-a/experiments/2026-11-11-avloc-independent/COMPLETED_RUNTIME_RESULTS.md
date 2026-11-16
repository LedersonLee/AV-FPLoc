# Completed seed42 training and warm inference costs — 2026-11-16

Track A, Exploratory; Val19 only, no training or rendering in this audit.

| Arm | Epochs | Recorded training hours | Parameters (M) | GPU model median ms | Timed-stage sum median ms |
|---|---:|---:|---:|---:|---:|
| A0 | 100 | 10.56 | 24.572 | 4.87 | 18.87 |
| A1 | 70 | 3.16 | 12.491 | 2.63 | 30.47 |
| A2 | 99 | 12.40 | 36.663 | 6.50 | 28.48 |
| A2_disabled | 69 | 9.35 | 36.663 | 7.65 | 32.26 |
| A3 | 87 | 17.45 | 37.869 | 14.37 | 36.52 |

Training totals include validation and shared-host contention; they are not isolated GPU hours. A3 has87completed epochs and best epoch67. The aggregate sum across all five arms is52.92hours. See [training receipt](completed_training_cost_all_seed42.json).

Inference uses the same uniform one-query-per-scene selection (19queries),3warmups and10repetitions, batch1, RTX4090, pinned PyTorch2.4.1 image. All95query/arm combinations pass prediction tolerance and exact Top1/Top20 replay against saved predictions. Stage sums include preprocessing, model, transfer, CPU20cm/10° field and NMS. Median stage sums are computed per repetition, not by summing independently reported medians. See [aggregate replay and timing evidence](runtime_all_seed42.json).

The GPU was idle before launch; existing resident services were preserved. CPU A3 evaluation ran concurrently. This is a shared-host measurement, not an isolated benchmark or streaming throughput. Input file decoding, initialization, loading, hashing and posterior diagnostics are excluded. Preprocessing computes both modalities even for A0/A1. No FP-RIR baseline encoder cost is included. Consequently these numbers cannot support an end-to-end speedup over B2+GEOQ. A3's model stage is slower than A2 in this implementation; timing does not establish why.

Reproduction: `benchmark_ray_runtime.py --study <study> --out <fresh-dir> --arm <arm>` for each completed arm, GPU enabled; `audit_completed_training_cost.py --study <study> --out <fresh.json> --include-arm A2_disabled --include-arm A3`. Runtime code a4463e9; accounting366b368; frozen training62c58da20ed26ba95863638bbfd4926829999eec. Exact source hashes accompany results. No training sources or checkpoints changed.

## Cached baseline stages

Same19queries,3warmups/10repetitions. All field/candidate/GEOQ replay assertions pass. [Receipt](runtime_cached_repeated.json). Code b75215d. CPU stage medians in ms:

| Method | Stage | Median ms |
|---|---|---:|
| visual | field_ms | 4.44 |
| visual | score_nms_ms | 10.28 |
| acoustic | field_ms | 6.03 |
| acoustic | score_nms_ms | 8.64 |
| fixed | field_ms | 11.39 |
| fixed | score_nms_ms | 10.56 |
| b2 | field_ms | 11.25 |
| b2 | score_nms_ms | 10.97 |
| geoq | cached_candidate_rank_ms | 0.92 |
| rebuilt_support_geoq | historical_three_extractors_ms | 14.11 |
| rebuilt_support_geoq | rebuilt_support_rank_ms | 1.14 |

These cached baseline measurements exclude original RGB/RIR encoding and B2 adaptation. The rebuilt support path runs historical score/support/connected extraction together, not optimized support-only execution; static geometry preparation is excluded. Separate medians must not be summed into claimed end-to-end latency. Comparisons against raw-input ray stages are not equivalent-cost speedups.

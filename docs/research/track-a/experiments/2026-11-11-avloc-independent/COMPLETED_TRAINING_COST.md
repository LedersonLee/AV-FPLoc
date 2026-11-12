# Completed expert training cost — 2026-11-12

Track A; Exploratory measured accounting, not an isolated throughput benchmark. Seed42, full112Train /58,894 observations, Val19 /9,965. No new training or inference; checkpoints read on CPU only.

| Expert | Executed epochs | Best epoch | Sum epoch wall time (h) | Median epoch (min) | Parameters (M) | Best checkpoint (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| A0 | 100 | 99 | 10.56 | 6.31 | 24.572 | 94.07 |
| A1 | 70 | 50 | 3.16 | 2.65 | 12.491 | 47.77 |
| A2 | 99 | 79 | 12.40 | 7.48 | 36.663 | 140.30 |

Total recorded epoch wall time: 26.11 hours. This is the sequential experts' epoch-time sum, not the research task elapsed time or an isolated GPU-hours measure.

Epoch time includes training and per-epoch validation, excludes startup/initial validation, and overlapped other evaluations/services. Data loading, caching and shared hardware affect it. It must not be presented as pure forward/backward compute or predict future A3 time. Exact parameter counts were obtained by instantiating each architecture on CPU, strictly loading its saved best-checkpoint state and summing model.parameters(). No state/checkpoint was modified. FLOPs, isolated end-to-end inference and baseline rendering/extraction costs remain unmeasured.

A0 ran100epochs; A1 ran70 and A2 ran99. The known patience-loop completion counter reports71/100 for the latter two, but contiguous executed epoch records are authoritative. Every epoch record and checkpoint is hashed in [the audit](completed_training_cost.json). The initial local auditv1 used state-dict accounting;v2 additionally verified exact model parameter counts (same values). Original outputs preserved.

Reproduce with the pinned study Docker image on CPU: `audit_completed_training_cost.py --study <study-root> --out <new-output.json>`. Audit code `1937132`; model/training code `62c58da20ed26ba95863638bbfd4926829999eec`. These counts complement the earlier bank-storage audit; FP-RIR storage and model size are separate costs. Pending interaction-disabled/A3 runs and interrupted seed43 are excluded. No Test17 access.

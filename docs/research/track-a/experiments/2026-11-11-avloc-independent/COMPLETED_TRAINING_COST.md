# Completed expert training cost — updated 2026-11-13

Track A; Exploratory measured accounting, not an isolated throughput benchmark. Seed42, full112Train /58,894 observations, Val19 /9,965. No new training or inference; checkpoints read on CPU only.

| Expert | Executed epochs | Best epoch | Sum epoch wall time (h) | Median epoch (min) | Parameters (M) | Best checkpoint (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| A0 | 100 | 99 | 10.56 | 6.31 | 24.572 | 94.07 |
| A1 | 70 | 50 | 3.16 | 2.65 | 12.491 | 47.77 |
| A2 | 99 | 79 | 12.40 | 7.48 | 36.663 | 140.30 |
| A2_disabled | 69 | 49 | 9.35 | 8.12 | 36.663 | 140.30 |

Total recorded epoch wall time including the completed control: 35.46 hours (original A0/A1/A2 subtotal26.11 hours). This is the sequential experts' epoch-time sum, not the research task elapsed time or an isolated GPU-hours measure.

Epoch time includes training and per-epoch validation, excludes startup/initial validation, and overlapped other evaluations/services. Data loading, caching and shared hardware affect it. It must not be presented as pure forward/backward compute or predict future A3 time. Exact parameter counts were obtained by instantiating each architecture on CPU, strictly loading its saved best-checkpoint state and summing model.parameters(). No state/checkpoint was modified. FLOPs, isolated end-to-end inference and baseline rendering/extraction costs remain unmeasured.

A0 ran100epochs; A1 ran70 and A2 ran99. The known patience-loop completion counter reports71/100 for the latter two, but contiguous executed epoch records are authoritative. Every epoch record and checkpoint is hashed in [the audit](completed_training_cost.json). The initial local auditv1 used state-dict accounting;v2 additionally verified exact model parameter counts (same values). Original outputs preserved.

Reproduce with the pinned study Docker image on CPU: `audit_completed_training_cost.py --study <study-root> --out <new-output.json>`. Audit code `1937132`; model/training code `62c58da20ed26ba95863638bbfd4926829999eec`. These counts complement the earlier bank-storage audit; FP-RIR storage and model size are separate costs. A3 remains running and interrupted seed43 is excluded. No Test17 access.

## Completed-control follow-up

The control audit strictly loaded its best checkpoint into the actual architecture on CPU and verified all69 contiguous full-cohort epoch records. A2 and disabled each have36,663,106 parameters and the same model tensor payload. The disabled control records33,665.89seconds over69epochs (9.35h), median487.21seconds/epoch, versus A2 median448.66seconds/epoch. Its shorter total reflects fewer executed epochs, not demonstrated faster computation. Shared-host timing and different validation/optimization paths prevent a controlled speed conclusion.

This supports the parameter-matching portion of the [control contract](INTERACTION_CONTROL_CONTRACT.md), but equal parameter counts do not mean equal compute or a pure attention intervention. Control best epoch49 and patience-loop counter70 are preserved;69 actual epochs are authoritative. [Extended cost evidence](completed_training_cost_with_disabled.json), [original three-expert audit](completed_training_cost.json).

Executed audit source `366b36862d6cc4af0b5e9d35b77584372f155a9d`; model/training source remains `62c58da20ed26ba95863638bbfd4926829999eec`. Reproduce with `audit_completed_training_cost.py --study <study-root> --out <fresh-json> --include-arm A2_disabled`. No raw checkpoint, logs or input arrays are committed. End-to-end isolated inference cost remains pending.

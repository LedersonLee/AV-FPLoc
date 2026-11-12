# Seed 42 completion before replication — 2026-11-12

Status: seed 43 deferred for resource prioritization, not a failed scientific experiment. Evidence level: Exploratory. Primary Track: A.

The user requested prioritizing the complete seed 42 experiment set before repeating A0/A1/A2 with seed 43. The original sequential queue was stopped; A0 seed 43 was interrupted during epoch 9. Eight completed epoch records, best checkpoints, optimizer/RNG resume checkpoint, protocol and logs remain local and unchanged. No partial seed 43 localization result was used for model selection. A1/A2/A2-disabled seed 43 have not started and will not automatically restart.

The remaining seed 42 interaction-disabled control now runs with the original full training protocol (112 scenes / 58,894 queries, clean Val19 / 9,965 queries, maximum 100 epochs, patience 20, minimum validation ray MAE). A3 seed 42 remains required after the actual A0/A1/A2 evaluation gate. A2 seed 42 evaluation continues independently. Test17 remains unused; no Track B changes.

## Interpretation and limits

Observed: A0/A1/A2 seed 42 training completed; A0 seed 43 has only eight completed epochs. No observation establishes that seed variance is small. The resource decision prioritizes comparisons between mechanisms over replication at this stage; it is not evidence of reproducibility. Existing baselines retain their available two-seed results, with asymmetric seed counts explicitly labeled. Scene bootstrap does not estimate training-seed variability. Small gains and final publication claims should receive targeted additional-seed validation. Report writing does not complete the remaining benchmark.

## Preservation and reproduction

Training code: `62c58da20ed26ba95863638bbfd4926829999eec`, unchanged source closure. Pinned image: `sha256:e51cb712100f25d0018aa65e371e63deaa51d12601c465ce698c0c446800c318`.

Local scheduling receipts: `experiments/avloc_20261111/seed42_priority_20261112.json` and `seed42_priority_launch.json` contain the interruption metadata, exact detached control launch command and source hashes. Original queue plan/log remain preserved. Do not restart that queue, which assumes fresh output folders. To resume A0 seed 43 later, use the same pinned image, mounts and original A0 seed 43 command with `--resume`; it restores the epoch-8 optimizer, scheduler and RNG state and recomputes epoch 9. No automatic seed 43 resumption is scheduled.

Saved A0 seed 43 `latest.pt` SHA256: `2dee7b34ac1ac9e69e7496991124c1428bf0d55138b29a0d64d218580a13c08d`.

No completed localization metric, baseline comparison or causal conclusion is reported for this interrupted arm. Those analyses are inapplicable until its protocol is completed. The interruption is administrative, not early stopping selected by validation performance.

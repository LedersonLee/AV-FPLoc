> 역사적 연구 기록. 원본: `fusion_vs_posterior_20261107_1300/PROTOCOL.md`. 원본 SHA256은 [산출물 목록](../artifacts.json)에 있습니다.
> 당시의 완료·재개 지시는 현재 실행 요청이 아닙니다. 최신 판정은 [현재 결론](../../../../current_findings.md)을 먼저 확인하세요.

# Front-end / posterior decision study

Local resources authorized freely; existing servers preserved. Test outcomes remain sealed.
Prior Val8 is development. No claim of independent generalization from this set.
Baseline: B2 epoch8, fixed acoustic weight .15, original full SE2 bank; strict <=1m/30deg and <=.1m evaluation.

## Frozen factorial
Use existing Train12 OOF-fitted HistGradientBoosting risk selector (22 GT-free features, eight candidate acoustic weights). Seed42 model is preserved from v2; seed43 fits identical procedure. Threshold is selected only on Train scene OOF. The same selected weight is supplied to MAP, local-support, and support+refinement. No decoder-specific gate fitting. Upstream and gate CV are not fully nested; Train scores are selection diagnostics, not unbiased generalization.
Local support uses T=.05, XY radius .4m, yaw radius10deg from v2. Refine selects original-score maximum in that neighborhood. Compare raw support and refined variants separately.

## Diagnostics
Oracle weights are the pre-existing eight-weight grid, not a continuous optimum. Oracle hypothesis recall is full bank Top-K joint success coverage at K1,5,20,100. These are optimistic diagnostic bounds, never inference inputs. Adjacent candidates are correlated; follow with spatially separated hypotheses if evidence warrants. GT-region mass uses normalized exp(f/.05). Confidence is posterior mass in the MAP-centered success region. It is not automatically a calibrated correctness probability.

## Independent data audit
11 remaining parent Val scenes have raw FP/Observed directories and all19 feature-readiness records. These are candidates for untouched validation of this fusion/decoder research, not necessarily untouched by unrelated acoustic experiments. Use original frozen small-pilot acoustic and visual encoders; do not substitute full116-trained model checkpoints. Verify schedule/candidate and feature protocol before preparing new inputs. Freeze final method before accessing their outcomes.

## Provenance incident
Initial host `python` command failed (only python3 exists), so copied wrapper briefly launched seed43 preparation with old default output directory. Stopped only that newly created container; moved the newly created cache_seed43 into this study. No pre-existing artifact was intentionally edited. Verify old v2 artifact hashes before relying on preservation. Wrapper corrected to new output and prior dependencies read-only.

## Input replay finding
The four-query smoke changed acoustic convolution batch shape versus original64, producing ~9e-5 embedding differences. Repeating the existing extra-Val scene with the original64-query batch reproduced mu, scale, observed embeddings, audio tokens, visual tokens and truth exactly (all max errors0). `independent_data/smoke64_103997478_171030528/receipt.json`. No tolerance was relaxed. Full preparation preserves original batch sizes64 acoustic and4 vision; tail batch retains natural scene size, as before.

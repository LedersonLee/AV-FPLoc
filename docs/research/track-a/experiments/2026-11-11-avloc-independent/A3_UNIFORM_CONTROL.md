# Same-head uniform routing control — 2026-11-17

Track A, Exploratory; fullVal19/9965queries, seed42,20cmXY/10°yaw. Native means reaggregated from all query rows and verified against summary. No new training or Test access.

| Same A3 heads | R@0.1m | R@1m | Joint | Top5 joint | Top20 joint |
|---|---:|---:|---:|---:|---:|
|Uniform available-expert weights|17.90%|48.87%|47.82%|75.26%|89.57%|
|Learned routing|30.92%|67.75%|66.92%|87.01%|95.32%|

Learned−uniform Joint+19.107pp micro; equal-scene macro+19.182pp,95%bootstrapCI[16.351,21.977]pp. Every19scene improves. There are2385repairs and481harms. Fine+13.016pp, macroCI[11.201,14.790]pp,1672repairs/375harms. [Paired evidence](A3_uniform20_contrasts.json), [uniform aggregate](A3_uniform20_summary.json).

## What this establishes

Replacing learned weights by equal weights on the same frozen heads substantially harms localization on this development cohort. The intervention changes both fused mean and moment-matched uncertainty; it does not isolate either component. Uniform weights give the comparatively inaccurate standalone acoustic head much more influence inside FoV. Downweighting that head is a plausible explanation supported by the recorded weights and head errors.

## What it does not establish

This is an inference intervention in a model trained with learned routing, not a separately trained fixed-router model. It cannot establish that the benefit requires query-adaptive reliability rather than a simpler direction-dependent prior. No independent calibration or domain robustness claim is made. A3 still trails B2+GEOQ on matched20cm Joint despite its superior fine-position accuracy. Single-seed and reused-development limitations remain.

Uniform rayMAE0.632119 versus learned0.555558. Uniform native nearest-gridNLL70.692 is lower than learned125.723 despite worse localization; the broader uniform field illustrates that distribution scoring and Top1 success are distinct objectives. Calibration results, not native sharpness alone, must accompany claims.

10cm uniform evaluation and both final paired comparisons are now complete. Original9,965-query model outputs and all prior artifacts are preserved.

Reproduce `compare_router.py --study <study> --out <fresh> --grid-cm 20 --include-uniform` at f20d22f. Uniform export and field evaluation used the original immutable evaluation snapshot; executable/input hashes in receipts. Frozen training62c58da20ed26ba95863638bbfd4926829999eec.

## Completed10cm result

Uniform: fine28.289%, R@1m57.301%, Joint56.247%, Top5joint81.555%, Top20joint92.313%. Learned A3: fine49.473%, R@1m76.829%, Joint76.187%. Learned−uniform Joint+19.940pp micro; macro+20.093pp95%CI[17.171,22.850]pp,2288repairs/301harms. Fine+21.184pp; macroCI[18.008,23.849]pp. [10cm aggregate](A3_uniform10_summary.json), [both-grid row verification](A3_uniform_final_integrity.json).

A3−A2 at10cm: Joint+2.007pp, macroCI[1.228,2.787]pp,765repairs/565harms. Fine+0.140pp, macroCI[−0.766,1.629]pp includes zero. Therefore the small fine improvement is not robust across the sampled development scenes under this bootstrap. [Final20cm contrasts](A3_final_paired20.json), [final10cm contrasts](A3_final_paired10.json). Paired code f20d22f, exact sources in receipts. Same-head uniform limitations stated above remain unchanged.

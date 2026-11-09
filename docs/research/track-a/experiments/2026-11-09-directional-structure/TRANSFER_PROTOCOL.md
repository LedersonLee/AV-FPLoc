# Exploratory transfer freeze — after New12 analysis, before Transfer11 scoring

Preserve visual-mode poses: extract20purevisual NMS hypotheses, add .15 within-candidate standardized robust structural score to .85 visual score; never move a pose within its visual mode. This is already the fixed C arm, not a new coefficient tuned after D. On12reused head-held scenes it improves joint+3.125pp CI[2.018,4.297],fine+1.367pp CI[.716,2.018] vsvisual; remains belowvisual+cosine and strongGEOQ. Fullbank addition harmsfine−5.273pp.

Fit a finalcombinedECN+embedding Ridge(alpha100, existing normalization) on allNew12; freezebefore scoring existing Transfer11. Reuse the same128query/scene, pureV NMS20, fixedcoefficient .15 and all controls. No transfer architecture/threshold/epoch selection. Compare V,cosine,V+cosine,matched/donor/prior geometry,V+geometry and originalB2/GEOQ. Transfer11 was already used by previousresearch and exposedto visualteacher, so this is exploratory head-transfer, never independent validation. Do not expandmodelcapacity if transfer fails.

Separately compute a true-query-geometry oracle over the originalNew12 fullbank solely to distinguish target representation/quantization limits from acoustic prediction error; never use oracle values for model input or methodselection. Report it explicitly as nondeployable diagnostic.

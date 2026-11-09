# Ray representation and joint acoustic–visual learning

Track A · ray_joint_20261109 · 2026-11-09 · Exploratory · **Completed controlled study**.

[Research report](RESEARCH_REPORT.md) · [Method and pipeline](METHOD_AT_A_GLANCE.md) · [Reproduction](REPRODUCE.md)

228checkpoint fits and all saved prediction replay checks are complete. Fullbank chirp improves raw by2.897pp; attention improves concat by3.060pp but not clearly interaction-disabled. Residual weighting improves fullbankattention by2.181pp. None beats B2+GEOQ. FOV-limited late fusion recovers losses from unsupported visual rays but does not improve acoustic-only. All primary and conditioned full-bank evaluations are complete, with independent uniform-score consistency checks. Geometry-difference anchor selection harms all four models. All failures/corrections are preserved in the report.

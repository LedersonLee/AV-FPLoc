# Corrections

- Initial FP preprojection extraction in64-example batches disagreed with the frozen256-batch cache by0.00013525. Attempt stopped before any fit. Restore original256 cell-block batch shapes, preserving final short blocks; no tolerance relaxation. Initial error log retained.

Post-run documentation correction: raw neural output is bounded +/-0.5, but candidate-relative subtraction bounds the correction at +/-1. Code was unchanged. Resource plan text corrected from stale10arms/60fits to implemented11arms/66fits.

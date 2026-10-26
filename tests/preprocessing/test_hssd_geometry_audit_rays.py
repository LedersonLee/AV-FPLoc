"""Geometry-audit camera sampling contract (host-side, no Habitat import)."""

from pathlib import Path
import sys

import numpy as np
import pytest

from avfploc.preprocessing.hssd.hssd_geometry_audit_rays import feature_column_depths


def test_samples_all_40_half_pixel_centers_in_rgb_order():
    row = np.arange(640, dtype=np.float32) / 2 + 1
    depths, ambiguous = feature_column_depths(row)
    np.testing.assert_allclose(depths, np.arange(40) * 8 + 4.75)
    assert not ambiguous.any()


def test_no_hit_is_nan_and_not_a_zero_distance_wall():
    row = np.full(640, 5, dtype=np.float32)
    row[7:9] = 0
    row[23:25] = np.nan
    depths, ambiguous = feature_column_depths(row)
    assert np.isnan(depths[:2]).all()
    assert not ambiguous[:2].any()
    assert np.isfinite(depths[2:]).all()


def test_one_hit_one_no_hit_or_jump_is_ambiguous():
    row = np.full(640, 5, dtype=np.float32)
    row[7], row[8] = 0, 5
    row[23], row[24] = 2, 8
    depths, ambiguous = feature_column_depths(row)
    assert ambiguous[:2].tolist() == [True, True]
    assert np.isnan(depths[:2]).all()


def test_rejects_wrong_row_width():
    with pytest.raises(ValueError):
        feature_column_depths(np.ones(639))

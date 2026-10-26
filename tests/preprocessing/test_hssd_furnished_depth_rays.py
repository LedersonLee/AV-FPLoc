"""Furnished camera-depth QA contract, independent of Habitat for unit cases."""

from __future__ import annotations

import importlib
from pathlib import Path
import sys

import numpy as np



def _module():
    return importlib.import_module("avfploc.preprocessing.hssd.hssd_furnished_depth_rays")


def test_narrow_sensor_is_used_only_when_stable_rays_match_full_depth():
    narrow = np.full(40, 3.0, dtype=np.float32)
    full = narrow.copy()
    full[0] = 3.4
    assert _module().choose_sensor_mode(narrow, full, np.ones(40, dtype=bool)) == "narrow"
    full[1] = 3.4
    full[2] = 3.4
    assert _module().choose_sensor_mode(narrow, full, np.ones(40, dtype=bool)) == "full"


def test_saved_rgb_control_rejects_materially_different_rendering():
    saved = np.full((10, 10, 3), 100, dtype=np.uint8)
    same = saved.copy()
    same[0, 0, 0] = 103
    different = saved.copy()
    different[0:5] = 130
    assert _module().rgb_control_matches(saved, same)
    assert not _module().rgb_control_matches(saved, different)


def test_depth_feature_sampler_preserves_rgb_left_to_right_and_no_hit():
    row = np.full(640, 4.0, dtype=np.float32)
    row[7:9] = 1.0
    row[23:25] = 0.0
    depth, ambiguous = _module().feature_column_depths(row)
    assert depth.shape == (40,)
    assert depth[0] == 1.0
    assert np.isnan(depth[1])
    assert not ambiguous[1]

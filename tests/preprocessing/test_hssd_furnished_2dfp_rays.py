"""Independent exact-map rays and legacy DeSDF target parity."""

from __future__ import annotations

import importlib
from pathlib import Path
import sys

import numpy as np
import pytest

from avfploc.preprocessing.hssd.hssd_unloc_dataset import WorldRaster


def _module():
    return importlib.import_module("avfploc.preprocessing.hssd.hssd_furnished_2dfp_rays")


def _fixture():
    raster = WorldRaster(0.0, 0.99, 0.0, 0.99, 0.01)
    geometry = np.full((100, 100), 255, dtype=np.uint8)
    geometry[:, 60] = 0
    record = {"grid_row": 50, "grid_col": 20, "pose_xyz_m": [0.20, 1.5, 0.50],
              "logical_yaw_deg": 0.0, "camera_width_px": 640}
    return record, geometry, raster


def test_target_uses_legacy_ten_centimetre_origin_and_pinhole_bearing():
    record, geometry, raster = _fixture()
    depth = _module().v2_compatible_target40(record, geometry, raster)
    # Feature column 19 is 8 px left of the principal point, so its yaw bin
    # rounds to 0 and its forward depth is 0.40*cos(atan(-8/240)).
    assert depth[19] == pytest.approx(0.4 * 240.0 / np.hypot(240.0, 8.0), abs=0.01)
    assert depth.shape == (40,)


def test_exact_map_ray_is_camera_forward_and_detects_wall():
    record, geometry, raster = _fixture()
    depth = _module().exact_map_depth40(record, geometry, raster)
    assert 0.38 < depth[19] < 0.42
    assert depth[19] == pytest.approx(depth[20], abs=0.03)


def test_parity_check_blocks_mismatched_target_rows():
    record, geometry, raster = _fixture()
    target = _module().v2_compatible_target40(record, geometry, raster)[None, :]
    report = _module().verify_scene_target_parity([record], target, geometry, raster)
    assert report["passed"]
    wrong = target.copy()
    wrong[0, 19] += 0.1
    assert not _module().verify_scene_target_parity([record], wrong, geometry, raster)["passed"]


def test_legacy_target_uses_explicit_desdf_occluders_not_all_black_map_cells():
    record, geometry, raster = _fixture()
    occluder = geometry == 0
    occluder[:, 60] = False  # Portal mask is black in occupancy but transparent to DeSDF.
    occluder[:, 90] = True
    expected = _module().v2_compatible_target40(record, geometry, raster, occluder_mask=occluder)
    assert expected[19] > 0.65
    assert _module().verify_scene_target_parity(
        [record], expected[None, :], geometry, raster, occluder_mask=occluder,
    )["passed"]
    assert not _module().verify_scene_target_parity(
        [record], expected[None, :], geometry, raster,
    )["passed"]

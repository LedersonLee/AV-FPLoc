#!/usr/bin/env python3
"""Exact map rays and compact, legacy-compatible DeSDF rays for HSSD."""

from __future__ import annotations

import math

import numpy as np

from .export_hssd_unloc_frames import _nearest_yaw_bins, unloc_feature_column_bearings_deg
from .hssd_unloc_dataset import WorldRaster


MAX_RANGE_M = 10.0
DESDF_SPACING_M = 0.10


def _checked_geometry(geometry: np.ndarray, raster: WorldRaster) -> np.ndarray:
    grid = np.asarray(geometry, dtype=np.uint8)
    if grid.shape != (raster.height, raster.width):
        raise ValueError(f"geometry shape {grid.shape} disagrees with raster {(raster.height, raster.width)}")
    return grid


def _legacy_targets_many(
    records: list[dict], geometry: np.ndarray, raster: WorldRaster,
    *, occluder_mask: np.ndarray | None = None,
) -> np.ndarray:
    grid = _checked_geometry(geometry, raster)
    occluder = grid == 0 if occluder_mask is None else np.asarray(occluder_mask, dtype=bool)
    if occluder.shape != grid.shape:
        raise ValueError("DeSDF occluder mask shape disagrees with geometry")
    if not records:
        return np.empty((0, 40), dtype=np.float32)
    ratio = int(round(DESDF_SPACING_M / raster.spacing_m))
    if ratio < 1 or not math.isclose(ratio * raster.spacing_m, DESDF_SPACING_M, abs_tol=1e-9):
        raise ValueError("map spacing must divide 0.10 m DeSDF spacing")
    bearings = unloc_feature_column_bearings_deg()
    positions = np.asarray([(int(r["grid_row"]), int(r["grid_col"])) for r in records], dtype=np.int64)
    if np.any(positions < 0) or np.any(positions[:, 0] >= grid.shape[0]) or np.any(positions[:, 1] >= grid.shape[1]):
        raise ValueError("capture position is outside geometry")
    if any(int(r["camera_width_px"]) != 640 for r in records):
        raise ValueError("legacy target requires 640-pixel RGB")
    origins = (positions // ratio) * ratio
    yaw = np.asarray([float(r["logical_yaw_deg"]) for r in records], dtype=np.float64)
    bins = _nearest_yaw_bins(yaw[:, None] + bearings[None, :])
    flat_origins = np.repeat(origins, 40, axis=0)
    flat_bins = bins.reshape(-1)
    keys = np.column_stack((flat_origins, flat_bins))
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    source_row, source_col, yaw_bin = unique.T
    if np.any(grid[source_row, source_col] == 0):
        raise ValueError("legacy DeSDF source is blocked in map")
    angles = yaw_bin.astype(np.float64) * (2.0 * np.pi / 36.0)
    sin_angle, cos_angle = np.sin(angles), np.cos(angles)
    ranges = np.full(len(unique), MAX_RANGE_M, dtype=np.float32)
    unresolved = np.ones(len(unique), dtype=bool)
    max_steps = int(np.ceil(MAX_RANGE_M / raster.spacing_m))
    for step in range(1, max_steps + 1):
        active = np.flatnonzero(unresolved)
        if not len(active):
            break
        delta_row = np.rint(step * sin_angle[active]).astype(np.int64)
        delta_col = np.rint(step * cos_angle[active]).astype(np.int64)
        probe_row = source_row[active] + delta_row
        probe_col = source_col[active] + delta_col
        inside = ((probe_row >= 0) & (probe_row < grid.shape[0]) &
                  (probe_col >= 0) & (probe_col < grid.shape[1]))
        hit = np.zeros(len(active), dtype=bool)
        hit[inside] = occluder[probe_row[inside], probe_col[inside]]
        if hit.any():
            ids = active[hit]
            ranges[ids] = np.minimum(
                np.hypot(delta_row[hit], delta_col[hit]) * raster.spacing_m, MAX_RANGE_M,
            )
            unresolved[ids] = False
    radial = ranges[inverse].reshape(len(records), 40)
    return (radial * np.cos(np.deg2rad(bearings))[None, :]).astype(np.float32)


def v2_compatible_target40(
    record: dict, geometry: np.ndarray, raster: WorldRaster,
    *, occluder_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Regenerate only the 40 needed legacy yaw-bin rays, not the whole DeSDF."""

    return _legacy_targets_many([record], geometry, raster, occluder_mask=occluder_mask)[0]


def verify_scene_target_parity(
    records: list[dict], target40: np.ndarray, geometry: np.ndarray, raster: WorldRaster,
    *, occluder_mask: np.ndarray | None = None,
) -> dict[str, object]:
    """Fail closed unless regenerated targets match every supplied v2 row."""

    expected = np.asarray(target40, dtype=np.float32)
    if expected.shape != (len(records), 40) or not np.isfinite(expected).all():
        raise ValueError("reference targets have wrong shape or nonfinite values")
    derived = _legacy_targets_many(records, geometry, raster, occluder_mask=occluder_mask)
    difference = np.abs(derived - expected)
    return {
        "passed": bool(np.all(difference <= 0.02)),
        "rows": len(records),
        "rays": int(difference.size),
        "mismatched_rays": int(np.count_nonzero(difference > 0.02)),
        "max_abs_error_m": float(difference.max()) if difference.size else 0.0,
    }


def exact_map_depth40(record: dict, geometry: np.ndarray, raster: WorldRaster) -> np.ndarray:
    """Raycast pinhole bearings from the unsnapped world pose on map.png."""

    grid = _checked_geometry(geometry, raster)
    pose = np.asarray(record["pose_xyz_m"], dtype=np.float64)
    if pose.shape != (3,) or not np.isfinite(pose).all():
        raise ValueError("invalid pose")
    source_row, source_col = raster.world_to_rc(float(pose[0]), float(pose[2]))
    if grid[source_row, source_col] == 0:
        raise ValueError("camera starts in blocked map cell")
    relative_bearings = unloc_feature_column_bearings_deg()
    angles = np.deg2rad(float(record["logical_yaw_deg"]) + relative_bearings)
    directions_x, directions_z = np.cos(angles), np.sin(angles)
    ranges = np.full(40, MAX_RANGE_M, dtype=np.float32)
    unresolved = np.ones(40, dtype=bool)
    stride = raster.spacing_m / 2.0
    for distance in np.arange(stride, MAX_RANGE_M + 1e-9, stride):
        active = np.flatnonzero(unresolved)
        if not len(active):
            break
        probe_col = np.rint((pose[0] + distance * directions_x[active] - raster.x_min_m) / raster.spacing_m).astype(int)
        probe_row = np.rint((pose[2] + distance * directions_z[active] - raster.z_min_m) / raster.spacing_m).astype(int)
        inside = ((probe_row >= 0) & (probe_row < grid.shape[0]) &
                  (probe_col >= 0) & (probe_col < grid.shape[1]))
        hit = np.zeros(len(active), dtype=bool)
        hit[inside] = grid[probe_row[inside], probe_col[inside]] == 0
        hit |= ~inside
        if hit.any():
            ids = active[hit]
            ranges[ids] = min(float(distance), MAX_RANGE_M)
            unresolved[ids] = False
    return (ranges * np.cos(np.deg2rad(relative_bearings))).astype(np.float32)

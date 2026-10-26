#!/usr/bin/env python3
"""Sample stage-only Habitat depth at UnLoc's 40 RGB feature columns."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np


WIDTH_PX = 640
FEATURE_COLUMNS = 40
PIXELS_PER_COLUMN = WIDTH_PX // FEATURE_COLUMNS


def feature_column_depths(depth_row: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return camera-forward depths and edge ambiguities, RGB left to right.

    A feature center falls between two sensor pixels. A mixed hit/no-hit or
    strong depth discontinuity cannot be called a reliable 3D measurement.
    Both no-hit pixels instead produce NaN with ``ambiguous=False``.
    """

    row = np.asarray(depth_row, dtype=np.float32)
    if row.shape == (1, WIDTH_PX):
        row = row[0]
    if row.shape != (WIDTH_PX,):
        raise ValueError(f"expected one {WIDTH_PX}-pixel depth row, got {row.shape}")
    left = row[7::PIXELS_PER_COLUMN]
    right = row[8::PIXELS_PER_COLUMN]
    left_hit = np.isfinite(left) & (left > 0)
    right_hit = np.isfinite(right) & (right > 0)
    both_hit = left_hit & right_hit
    discontinuity = both_hit & (np.abs(left - right) > np.maximum(0.30, 0.15 * np.minimum(left, right)))
    ambiguous = (left_hit != right_hit) | discontinuity
    depth = np.full(FEATURE_COLUMNS, np.nan, dtype=np.float32)
    stable = both_hit & ~ambiguous
    depth[stable] = (left[stable] + right[stable]) / 2
    return depth, ambiguous


def render_scene_depths(
    scene_id: str, records: list[dict], hssd_root: Path, gpu_device_id: int = 0,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Render each recorded pose once from the structural stage GLB only.

    Habitat is deliberately imported inside this function so host-side
    classifier/unit tests run without the Habitat container.
    """

    import habitat_sim
    import quaternion

    hssd_root = Path(hssd_root)
    stage = hssd_root / "stages" / f"{scene_id}.glb"
    if not stage.is_file():
        raise FileNotFoundError(stage)
    if not records:
        return {}
    width = {int(r["camera_width_px"]) for r in records}
    fovs = {float(r["camera_hfov_deg"]) for r in records}
    if width != {WIDTH_PX} or len(fovs) != 1:
        raise ValueError(f"{scene_id}: mixed/unsupported camera contract")
    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = str(stage)
    backend.gpu_device_id = int(gpu_device_id)
    backend.enable_physics = False
    sensor = habitat_sim.CameraSensorSpec()
    sensor.uuid = "audit_depth"
    sensor.sensor_type = habitat_sim.SensorType.DEPTH
    sensor.resolution = [1, WIDTH_PX]
    sensor.hfov = fovs.pop()
    sensor.position = [0.0, 0.0, 0.0]
    agent = habitat_sim.agent.AgentConfiguration()
    agent.sensor_specifications = [sensor]
    result: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    with habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent])) as sim:
        for record in records:
            index = int(record["sample_index"])
            if index in result:
                raise ValueError(f"{scene_id}: duplicate sample_index {index}")
            pose = np.asarray(record["pose_xyz_m"], dtype=np.float32)
            if pose.shape != (3,) or not np.isfinite(pose).all():
                raise ValueError(f"{scene_id}/{index}: invalid pose")
            yaw_half = math.radians(float(record["habitat_yaw_deg"])) / 2
            state = sim.get_agent(0).get_state()
            state.position = pose
            state.rotation = quaternion.quaternion(math.cos(yaw_half), 0.0, math.sin(yaw_half), 0.0)
            sim.get_agent(0).set_state(state, reset_sensors=True)
            result[index] = feature_column_depths(sim.get_sensor_observations()["audit_depth"])
    return result

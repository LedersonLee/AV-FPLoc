#!/usr/bin/env python3
"""Measure UnLoc's horizontal depth rays in the furnished HSSD scene."""

from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
from PIL import Image

from .hssd_geometry_audit_rays import feature_column_depths
from .render_hssd_unloc_rgb_pilot import (
    KNOWN_CRASHING_HSSD_OBJECT_TEMPLATES,
    _yaw_quaternion,
    configure_simulator_backend,
    write_sanitized_scene_instance,
)


def choose_sensor_mode(narrow40: np.ndarray, full40: np.ndarray, valid40: np.ndarray) -> str:
    """Use the cheap centre-row sensor only after a 95%-ray calibration."""

    narrow = np.asarray(narrow40, dtype=np.float32)
    full = np.asarray(full40, dtype=np.float32)
    valid = np.asarray(valid40, dtype=bool)
    if narrow.shape != (40,) or full.shape != (40,) or valid.shape != (40,):
        raise ValueError("calibration expects three aligned 40-ray arrays")
    comparable = valid & np.isfinite(narrow) & np.isfinite(full) & (narrow > 0) & (full > 0)
    if not comparable.any():
        return "full"
    tolerance = np.maximum(0.10, 0.05 * np.minimum(narrow[comparable], full[comparable]))
    agreement = np.abs(narrow[comparable] - full[comparable]) <= tolerance
    return "narrow" if float(agreement.mean()) >= 0.95 else "full"


def rgb_control_matches(saved_rgb: np.ndarray, rerendered_rgb: np.ndarray) -> bool:
    """Detect scene-instance or pose drift before trusting rerendered depth."""

    saved = np.asarray(saved_rgb)
    current = np.asarray(rerendered_rgb)
    if saved.shape != current.shape or saved.ndim != 3 or saved.shape[2] != 3:
        return False
    delta = np.abs(saved.astype(np.int16) - current.astype(np.int16))
    return bool(float(delta.mean()) < 2.0 and float(np.mean(delta <= 5)) >= 0.99)


def _depth_features(depth_image: np.ndarray, sensor_mode: str) -> tuple[np.ndarray, np.ndarray]:
    depth = np.asarray(depth_image, dtype=np.float32)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if sensor_mode == "narrow":
        return feature_column_depths(depth)
    if sensor_mode != "full" or depth.shape != (480, 640):
        raise ValueError(f"unsupported depth image {sensor_mode}: {depth.shape}")
    upper, upper_ambiguous = feature_column_depths(depth[239])
    lower, lower_ambiguous = feature_column_depths(depth[240])
    mixed = np.isfinite(upper) != np.isfinite(lower)
    both = np.isfinite(upper) & np.isfinite(lower)
    jump = np.zeros(40, dtype=bool)
    jump[both] = np.abs(upper[both] - lower[both]) > np.maximum(
        0.30, 0.15 * np.minimum(upper[both], lower[both])
    )
    ambiguous = upper_ambiguous | lower_ambiguous | mixed | jump
    values = np.full(40, np.nan, dtype=np.float32)
    stable = both & ~ambiguous
    values[stable] = (upper[stable] + lower[stable]) / 2.0
    return values, ambiguous


def render_furnished_scene_depths(
    scene_id: str, records: list[dict], hssd_root: Path, gpu_device_id: int = 0,
    sensor_mode: str = "narrow",
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Render one scene once; records may include ``saved_rgb_path`` controls."""

    import habitat_sim

    if sensor_mode not in {"narrow", "full"}:
        raise ValueError("sensor_mode must be narrow or full")
    if not records:
        return {}
    widths = {int(row["camera_width_px"]) for row in records}
    heights = {int(row["camera_height_px"]) for row in records}
    fovs = {float(row["camera_hfov_deg"]) for row in records}
    if widths != {640} or heights != {480} or len(fovs) != 1:
        raise ValueError(f"{scene_id}: mixed or unsupported camera contract")
    source = Path(hssd_root) / "scenes" / f"{scene_id}.scene_instance.json"
    results: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    with tempfile.TemporaryDirectory(prefix="hssd_furnished_qa_") as scratch:
        derived, removed = write_sanitized_scene_instance(
            source, Path(scratch), excluded_templates=KNOWN_CRASHING_HSSD_OBJECT_TEMPLATES,
        )
        backend = configure_simulator_backend(
            habitat_sim, root=Path(hssd_root), scene_id=scene_id, gpu_device_id=gpu_device_id,
            scene_instance_path=derived if removed else source,
        )
        rgb = habitat_sim.CameraSensorSpec()
        rgb.uuid, rgb.resolution, rgb.hfov = "rgb", [480, 640], fovs.pop()
        rgb.position = [0.0, 0.0, 0.0]
        depth = habitat_sim.CameraSensorSpec()
        depth.uuid = "qa_depth"
        depth.sensor_type = habitat_sim.SensorType.DEPTH
        depth.resolution = [1 if sensor_mode == "narrow" else 480, 640]
        depth.hfov = rgb.hfov
        depth.position = [0.0, 0.0, 0.0]
        agent = habitat_sim.agent.AgentConfiguration()
        agent.sensor_specifications = [rgb, depth]
        with habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent])) as simulator:
            for row in records:
                sample_index = int(row["sample_index"])
                if sample_index in results:
                    raise ValueError(f"{scene_id}: duplicate sample index {sample_index}")
                pose = np.asarray(row["pose_xyz_m"], dtype=np.float32)
                if pose.shape != (3,) or not np.isfinite(pose).all():
                    raise ValueError(f"{scene_id}/{sample_index}: invalid pose")
                state = simulator.get_agent(0).get_state()
                state.position = pose
                state.rotation = _yaw_quaternion(float(row["habitat_yaw_deg"]))
                simulator.get_agent(0).set_state(state, reset_sensors=True)
                observations = simulator.get_sensor_observations()
                saved_path = row.get("saved_rgb_path")
                if saved_path is not None:
                    saved = np.asarray(Image.open(Path(saved_path)).convert("RGB"))
                    current = np.asarray(observations["rgb"])[..., :3]
                    if not rgb_control_matches(saved, current):
                        raise ValueError(f"{scene_id}/{sample_index}: saved RGB does not match furnished rerender")
                results[sample_index] = _depth_features(observations["qa_depth"], sensor_mode)
    return results

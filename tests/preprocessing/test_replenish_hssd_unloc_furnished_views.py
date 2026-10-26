"""Replacements stay at new indoor poses and never cross scene splits."""

from __future__ import annotations

import importlib
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from avfploc.preprocessing.hssd.hssd_unloc_dataset import WorldRaster


def _module():
    return importlib.import_module("avfploc.preprocessing.hssd.replenish_hssd_unloc_furnished_views")


def test_replacement_pose_cells_exclude_originals_and_respect_spacing():
    raster = WorldRaster(0.0, 1.99, 0.0, 1.99, 0.01)
    grid = np.full((200, 200), 255, dtype=np.uint8)
    original = np.asarray([[20, 20], [100, 100]], dtype=np.int32)
    picked = _module().replacement_pose_cells(grid, raster, original, limit=8, seed=7)
    assert picked.shape == (8, 2)
    assert not set(map(tuple, picked)).intersection(map(tuple, original))
    distances = np.linalg.norm((picked[:, None] - picked[None, :]) * 0.01, axis=2)
    distances += np.eye(len(picked)) * 100
    assert distances.min() >= 0.25 - 1e-9
    from_original = np.linalg.norm((picked[:, None] - original[None, :]) * 0.01, axis=2)
    assert from_original.min(axis=1).max() <= 1.0 + 1e-9


def test_rgb_and_qa_depth_are_captured_from_one_simulator_observation_call():
    class FakeAgent:
        def get_state(self):
            return type("State", (), {"position": None, "rotation": None})()

        def set_state(self, state, reset_sensors):
            assert reset_sensors

    class FakeSimulator:
        def __init__(self):
            self.calls = 0

        def get_agent(self, index):
            assert index == 0
            return FakeAgent()

        def get_sensor_observations(self):
            self.calls += 1
            return {"rgb": np.zeros((480, 640, 4), dtype=np.uint8),
                    "qa_depth": np.ones((1, 640), dtype=np.float32)}

    simulator = FakeSimulator()
    image, depth, ambiguous = _module().capture_rgb_depth_once(
        simulator, {"pose_xyz_m": [0, 1.5, 0], "habitat_yaw_deg": -90},
        rotation_factory=lambda yaw: yaw,
    )
    assert simulator.calls == 1
    assert image.shape == (480, 640, 3)
    assert depth.shape == (40,) and not ambiguous.any()


def test_replacement_target_count_never_uses_another_scene_or_split():
    assert _module().missing_view_count(original_count=560, same_scene_pass_count=217) == 343
    try:
        _module().missing_view_count(original_count=560, same_scene_pass_count=561)
    except ValueError:
        pass
    else:
        raise AssertionError("a cross-scene overcount must not be accepted")


def test_navmesh_floor_snap_accepts_nominal_20cm_float_roundoff():
    class Pathfinder:
        @staticmethod
        def snap_point(_pose):
            return np.asarray([1.0, 0.200000003, 2.0], dtype=np.float32)

    sim = type("Sim", (), {"pathfinder": Pathfinder()})()
    assert _module()._navmesh_pose_ok(sim, {"floor_xyz_m": [1.0, 0.0, 2.0]})


def test_replacement_loads_original_desdf_occluder_mask(tmp_path):
    scene = "scene_a"
    source = tmp_path / "UnLoc" / "structural_targets" / scene
    source.mkdir(parents=True)
    mask = np.zeros((4, 5), dtype=np.uint8)
    mask[1, 2] = 255
    Image.fromarray(mask).save(source / "desdf_occluder_mask.png")
    loaded = _module()._load_occluder_mask(tmp_path, scene, mask.shape)
    assert loaded.shape == mask.shape and loaded[1, 2] and loaded.sum() == 1


def test_replacement_completion_is_invalidated_by_audit_or_budget_change():
    current = {"complete": True, "split": "train", "target_count": 520,
               "audit_ledger_sha256": "abc", "max_new_poses": 400,
               "sampler_version": _module().SAMPLER_VERSION}
    assert _module().replacement_manifest_is_current(
        current, split="train", target_count=520, audit_ledger_sha256="abc", max_new_poses=400,
    )
    assert not _module().replacement_manifest_is_current(
        current, split="train", target_count=520, audit_ledger_sha256="new", max_new_poses=400,
    )
    assert not _module().replacement_manifest_is_current(
        current, split="train", target_count=520, audit_ledger_sha256="abc", max_new_poses=50,
    )

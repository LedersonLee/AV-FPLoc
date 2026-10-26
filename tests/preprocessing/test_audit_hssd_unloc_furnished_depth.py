"""Historical furnished-depth audit keeps rows and split identities intact."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from avfploc.preprocessing.hssd.hssd_furnished_2dfp_rays import exact_map_depth40
from avfploc.preprocessing.hssd.hssd_unloc_dataset import WorldRaster


def _module():
    return importlib.import_module("avfploc.preprocessing.hssd.audit_hssd_unloc_furnished_depth")


def _fixture(tmp_path):
    scene = "test-scene"
    rgb = tmp_path / "UnLoc" / "RGB" / scene
    frame = tmp_path / "UnLoc" / "frame_adapter_v2" / "frames" / scene
    fp = tmp_path / "2DFP" / scene
    semantic = tmp_path / "2DFP_w_semantic" / scene
    for path in (rgb / "rgb", frame, fp, semantic):
        path.mkdir(parents=True)
    grid = np.full((100, 100), 255, dtype=np.uint8)
    grid[:, 80] = 0
    Image.fromarray(grid).save(fp / "map.png")
    (fp / "floorplan_metadata.json").write_text(json.dumps({
        "world_bounds_m": {"x_min": 0.0, "x_max": 0.99, "z_min": 0.0, "z_max": 0.99},
        "map_spacing_m": 0.01,
    }))
    np.save(semantic / "opening_scope.npy", np.zeros((100, 100), dtype=np.uint8))
    records = []
    targets = []
    raster = WorldRaster(0.0, 0.99, 0.0, 0.99, 0.01)
    for index, yaw in ((1, 0), (2, 0)):
        image = f"rgb/{index:05d}.png"
        Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(rgb / image)
        row = {"sample_index": index, "pose_index": index, "accepted": True,
               "image": image, "grid_row": 50, "grid_col": 20,
               "pose_xyz_m": [0.2, 1.5, 0.5], "logical_yaw_deg": yaw,
               "habitat_yaw_deg": -yaw - 90, "camera_width_px": 640,
               "camera_height_px": 480, "camera_hfov_deg": 106.2602}
        records.append(row)
        targets.append(exact_map_depth40(row, grid, raster))
    (rgb / "pilot_manifest.json").write_text(json.dumps({"records": records}))
    rows = [{"scene_id": scene, "sample_index": r["sample_index"], "target_index": i,
             "image_path": str(rgb / r["image"]), "logical_yaw_deg": r["logical_yaw_deg"],
             "pose_xzyaw": [0.2, 0.5, r["logical_yaw_deg"]]}
            for i, r in enumerate(records)]
    (frame / "sample_metadata.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    np.save(frame / "depth40.npy", np.asarray(targets, dtype=np.float32))
    return scene, records, np.asarray(targets, dtype=np.float32)


def test_audit_keeps_every_original_id_and_rejects_only_bad_observation(tmp_path):
    scene, records, target = _fixture(tmp_path)

    def furnished(_scene, _records, _root, gpu_device_id=0, sensor_mode="narrow"):
        good = target[0].copy()
        bad = target[1].copy()
        bad[10:13] = np.nan
        return {1: (good, np.zeros(40, dtype=bool)), 2: (bad, np.zeros(40, dtype=bool))}

    rows = _module().audit_scene(scene, "val", tmp_path, render=furnished, stage_render=lambda *a, **k: {})
    assert [r["sample_index"] for r in rows] == [1, 2]
    assert [r["split"] for r in rows] == ["val", "val"]
    assert [r["decision"] for r in rows] == ["pass", "fail"]
    assert "furnished_no_hit" in rows[1]["reasons"]


def test_stage_diagnostic_is_triggered_by_exact_map_not_quantized_target(tmp_path, monkeypatch):
    scene, records, target = _fixture(tmp_path)
    module = _module()
    monkeypatch.setattr(module, "exact_map_depth40", lambda *args: np.full(40, 8.0, dtype=np.float32))
    calls = []

    def furnished(_scene, _records, _root, gpu_device_id=0, sensor_mode="narrow"):
        return {r["sample_index"]: (np.full(40, 2.0, dtype=np.float32), np.zeros(40, dtype=bool))
                for r in _records}

    def stage(_scene, suspect, _root, gpu_device_id=0):
        calls.extend(r["sample_index"] for r in suspect)
        return {r["sample_index"]: (np.full(40, 2.0, dtype=np.float32), np.zeros(40, dtype=bool))
                for r in suspect}

    rows = module.audit_scene(scene, "val", tmp_path, render=furnished, stage_render=stage)
    assert calls == [1, 2]
    assert all(r["decision"] == "pass" and "false_open_sector" in r["diagnostic_sectors"] for r in rows)


def test_incomplete_scene_cannot_be_skipped(tmp_path):
    scene_dir = tmp_path / "scenes"
    scene_dir.mkdir()
    ledger = scene_dir / "test-scene.jsonl"
    ledger.write_text('{"sample_index":1}\n')
    assert not _module().scene_is_complete(ledger, expected_count=2, input_hashes={"source": "abc"})


def test_stale_policy_marker_cannot_skip_rerun(tmp_path):
    import hashlib

    ledger = tmp_path / "scene.jsonl"
    ledger.write_text('{"sample_index":1,"decision":"pass"}\n')
    ledger.with_suffix(".complete.json").write_text(json.dumps({
        "expected_count": 1, "input_hashes": {"source": "abc"},
        "ledger_sha256": hashlib.sha256(ledger.read_bytes()).hexdigest(),
        "policy": {"version": "old"}, "sensor_mode": "narrow",
    }))
    assert not _module().scene_is_complete(
        ledger, expected_count=1, input_hashes={"source": "abc"},
        policy={"version": "new"}, sensor_mode="narrow",
    )


def test_review_html_has_query_selector_pose_and_aligned_depth_curves(tmp_path):
    review = importlib.import_module("avfploc.preprocessing.hssd.build_hssd_furnished_depth_review")
    scene, records, target = _fixture(tmp_path)
    row = {"scene_id": scene, "split": "val", "sample_index": 1, "decision": "pass",
           "reasons": ["consistent_or_furniture_occluded"], "pose_xyz_m": [0.2, 1.5, 0.5],
           "logical_yaw_deg": 0.0, "image_path": f"UnLoc/RGB/{scene}/rgb/00001.png",
           "furnished_depth40_m": target[0].tolist(), "target_depth40_m": target[0].tolist(),
           "exact_map_depth40_m": target[0].tolist(), "stage_depth40_m": None}
    path = tmp_path / "UnLoc" / "furnished_depth_qa_v1" / "review" / "index.html"
    review.write_review(path, [row], tmp_path)
    html = path.read_text()
    assert "querySelect" in html
    assert "poseMarker" in html
    assert "furnished" in html and "2DFP" in html
    assert "UnLoc/RGB" in html

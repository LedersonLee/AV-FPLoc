from pathlib import Path
import json
import sys

import numpy as np

from avfploc.preprocessing.hssd.audit_hssd_unloc_geometry import _aggregate, audit_scene, load_scene_inputs
from avfploc.preprocessing.hssd.hssd_geometry_audit_policy import AuditPolicy
from avfploc.preprocessing.hssd.build_hssd_geometry_audit_review import build_html


def fixture(tmp_path):
    scene = "test_scene"
    rgb = tmp_path / "UnLoc" / "RGB" / scene
    frames = tmp_path / "UnLoc" / "frame_adapter_v2" / "frames" / scene
    (rgb / "rgb").mkdir(parents=True)
    frames.mkdir(parents=True)
    records = []
    metadata = []
    for index, yaw in [(0, 0), (1, 90)]:
        (rgb / "rgb" / f"{index:05d}.png").write_bytes(b"PNG")
        records.append({"accepted": True, "sample_index": index, "pose_xyz_m": [0, 1.5, 0],
                        "habitat_yaw_deg": -yaw-90, "logical_yaw_deg": yaw,
                        "camera_width_px": 640, "camera_hfov_deg": 106.2602,
                        "image": f"rgb/{index:05d}.png"})
        metadata.append({"scene_id": scene, "sample_index": index, "target_index": index,
                         "image_path": f"UnLoc/RGB/{scene}/rgb/{index:05d}.png",
                         "logical_yaw_deg": yaw, "pose_xzyaw": [0, 0, yaw]})
    (rgb / "pilot_manifest.json").write_text(json.dumps({"records": records}))
    (frames / "sample_metadata.jsonl").write_text("".join(json.dumps(r)+"\n" for r in metadata))
    np.save(frames / "depth40.npy", np.full((2, 40), 5, dtype=np.float32))
    return scene, records, metadata


def test_same_pose_two_yaws_only_bad_view_excluded(tmp_path):
    scene, _, _ = fixture(tmp_path)
    records, targets, issues = load_scene_inputs(scene, tmp_path)
    assert not issues
    assert len(records) == 2
    def render(*args, **kwargs):
        return {0: (np.full(40, 5.), np.zeros(40, bool)),
                1: (np.full(40, np.nan), np.zeros(40, bool))}
    rows = audit_scene(scene, "train", tmp_path, AuditPolicy(), render=render)
    assert [(r["sample_index"], r["decision"]) for r in rows] == [(0, "pass"), (1, "fail")]
    assert rows[0]["target_index"] == 0
    assert rows[1]["target_index"] == 1


def test_missing_v2_record_fails_closed_not_shifted(tmp_path):
    scene, _, _ = fixture(tmp_path)
    path = tmp_path / "UnLoc" / "frame_adapter_v2" / "frames" / scene / "sample_metadata.jsonl"
    path.write_text(path.read_text().splitlines()[0] + "\n")
    rows = audit_scene(scene, "test", tmp_path, AuditPolicy(),
                       render=lambda *a, **k: {0: (np.full(40, 5.), np.zeros(40, bool))})
    assert len(rows) == 2
    assert rows[1]["decision"] == "fail"
    assert "missing_adapter_row" in rows[1]["reasons"]


def test_duplicate_v2_index_marks_affected_source_row_failed(tmp_path):
    scene, _, _ = fixture(tmp_path)
    path = tmp_path / "UnLoc" / "frame_adapter_v2" / "frames" / scene / "sample_metadata.jsonl"
    first = path.read_text().splitlines()[0]
    path.write_text(path.read_text() + first + "\n")
    rows = audit_scene(scene, "val", tmp_path, AuditPolicy(), render=lambda *a, **k: {})
    assert len(rows) == 2
    assert rows[0]["decision"] == "fail"
    assert "duplicate_adapter_row" in rows[0]["reasons"]


def test_review_html_contains_query_toggle_map_pose_and_both_curves():
    row = {"scene_id": "scene_a", "sample_index": 7, "split": "test", "decision": "fail",
           "reasons": ["no_hit_sector"], "image_path": "UnLoc/RGB/scene_a/rgb/00007.png",
           "pose_xyz_m": [1.0, 1.5, 2.0], "logical_yaw_deg": 90,
           "stage_depth40_m": [None] * 40, "target_depth40_m": [1.0] * 40,
           "counts": {"no_hit": 40}}
    html = build_html([row], {"scene_a": {"x_min": 0, "x_max": 3, "z_min": 0, "z_max": 3}})
    assert 'id="query"' in html
    assert "00007.png" in html and "2DFP/" in html
    assert "GT pose" in html and "stage_depth40_m" in html and "target_depth40_m" in html


def test_summary_records_remaining_pose_and_yaw_diversity(tmp_path):
    root = tmp_path / "scenes"
    root.mkdir()
    rows = [
        {"sample_index": 0, "decision": "pass", "logical_yaw_deg": 0, "pose_xyz_m": [1, 1.5, 2]},
        {"sample_index": 1, "decision": "fail", "logical_yaw_deg": 90, "pose_xyz_m": [1, 1.5, 2]},
        {"sample_index": 2, "decision": "pass", "logical_yaw_deg": 180, "pose_xyz_m": [3, 1.5, 4]},
    ]
    (root / "scene_a.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    summary = _aggregate(tmp_path, {"train": ["scene_a"]})
    assert summary["by_scene"]["scene_a"]["source_poses"] == 2
    assert summary["by_scene"]["scene_a"]["passing_poses"] == 2
    assert summary["by_split"]["train"]["passing_yaws_per_pose"] == {"1": 2}

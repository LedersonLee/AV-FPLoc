#!/usr/bin/env python3
"""Audit saved furnished RGB views against their UnLoc structural 2DFP rays."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .audit_hssd_unloc_geometry import _atomic_text, _hash_file, _split_scenes, load_scene_inputs
from .export_hssd_unloc_frames import unloc_feature_column_bearings_deg
from .hssd_furnished_2dfp_rays import exact_map_depth40
from .hssd_furnished_depth_policy import AUDIT_DIRECTORY, FurnishedAuditPolicy, classify_furnished_view
from .hssd_furnished_depth_rays import render_furnished_scene_depths
from .hssd_geometry_audit_rays import render_scene_depths
from .render_hssd_unloc_rgb_pilot import _raster_from_metadata


def _scope_near_map_hits(record: dict, map_depth40: np.ndarray, scope: np.ndarray, raster) -> np.ndarray:
    """Annotate exact pinhole rays with exterior-opening scope near their map hit."""

    result = np.zeros(40, dtype=np.uint8)
    bearings = np.deg2rad(unloc_feature_column_bearings_deg())
    world_angles = np.deg2rad(float(record["logical_yaw_deg"])) + bearings
    pose = np.asarray(record["pose_xyz_m"], dtype=np.float64)
    radial = map_depth40 / np.cos(bearings)
    for offset in np.arange(-0.15, 0.301, raster.spacing_m):
        distance = np.maximum(0.0, radial + offset)
        cols = np.rint((pose[0] + distance * np.cos(world_angles) - raster.x_min_m) / raster.spacing_m).astype(int)
        rows = np.rint((pose[2] + distance * np.sin(world_angles) - raster.z_min_m) / raster.spacing_m).astype(int)
        valid = (rows >= 0) & (rows < scope.shape[0]) & (cols >= 0) & (cols < scope.shape[1])
        result[valid] = np.maximum(result[valid], scope[rows[valid], cols[valid]])
    return result


def _json_depth(values: np.ndarray | None) -> list[float | None] | None:
    if values is None:
        return None
    return [float(value) if np.isfinite(value) else None for value in values]


def audit_scene(
    scene_id: str, split: str, hssd_root: Path, policy: FurnishedAuditPolicy | None = None,
    *, gpu_device_id: int = 0, sensor_mode: str = "narrow",
    render=render_furnished_scene_depths, stage_render=render_scene_depths,
) -> list[dict]:
    """Return one decision per original accepted view; renderer drift aborts scene."""

    policy = policy or FurnishedAuditPolicy()
    root = Path(hssd_root)
    accepted, joined, issues = load_scene_inputs(scene_id, root)
    if len({int(row["sample_index"]) for row in accepted}) != len(accepted):
        raise ValueError(f"{scene_id}: duplicate original accepted ID")
    rgb_root = root / "UnLoc" / "RGB" / scene_id
    fp_root = root / "2DFP" / scene_id
    geometry = np.asarray(Image.open(fp_root / "map.png").convert("L"), dtype=np.uint8)
    scope = np.load(root / "2DFP_w_semantic" / scene_id / "opening_scope.npy")
    raster = _raster_from_metadata(fp_root / "floorplan_metadata.json")
    if scope.shape != geometry.shape:
        raise ValueError(f"{scene_id}: semantic and geometry grids disagree")
    candidates = [dict(row) for row in accepted if int(row["sample_index"]) in joined and int(row["sample_index"]) not in issues]
    for position in sorted({0, len(candidates) // 2, len(candidates) - 1}):
        if candidates:
            row = candidates[position]
            row["saved_rgb_path"] = str(rgb_root / row["image"])
    source_root = root / "source" / "hssd-hab_local_20261012"
    furnished = render(scene_id, candidates, source_root, gpu_device_id=gpu_device_id, sensor_mode=sensor_mode)
    exact: dict[int, np.ndarray] = {}
    suspects: list[dict] = []
    for capture in candidates:
        index = int(capture["sample_index"])
        if index not in furnished:
            continue
        exact[index] = exact_map_depth40(capture, geometry, raster)
        observed = furnished[index][0]
        valid = np.isfinite(observed) & (observed > 0)
        tolerance = np.maximum(policy.absolute_tolerance_m, policy.relative_tolerance * np.minimum(observed, exact[index]))
        if np.any(valid & ((exact[index] - observed) > tolerance)):
            suspects.append(capture)
    try:
        stage = stage_render(scene_id, suspects, source_root, gpu_device_id=gpu_device_id) if suspects else {}
    except Exception:
        stage = {}
    rows: list[dict] = []
    for capture in accepted:
        index = int(capture["sample_index"])
        joined_row = joined.get(index)
        row: dict[str, object] = {
            "scene_id": scene_id, "split": split, "sample_index": index,
            "target_index": None if joined_row is None else int(joined_row[0]["target_index"]),
            "image_path": None if joined_row is None else joined_row[0]["image_path"],
            "pose_xyz_m": capture.get("pose_xyz_m"), "grid_row": capture.get("grid_row"),
            "grid_col": capture.get("grid_col"), "logical_yaw_deg": capture.get("logical_yaw_deg"),
            "habitat_yaw_deg": capture.get("habitat_yaw_deg"), "camera_hfov_deg": capture.get("camera_hfov_deg"),
            "furnished_depth40_m": None, "target_depth40_m": None, "exact_map_depth40_m": None,
            "stage_depth40_m": None, "ambiguous40": None, "scope40": None,
        }
        if index in issues:
            verdict = {"decision": "fail", "reasons": sorted(issues[index]), "counts": {}, "ray_classes": []}
        elif index not in furnished:
            verdict = {"decision": "fail", "reasons": ["missing_furnished_render"], "counts": {}, "ray_classes": []}
        else:
            observed, ambiguous = furnished[index]
            target = joined_row[1]
            map_depth = exact[index]
            stage_result = stage.get(index)
            stage_depth = None if stage_result is None else stage_result[0]
            stage_ambiguous = None if stage_result is None else stage_result[1]
            ray_scope = _scope_near_map_hits(capture, map_depth, scope, raster)
            verdict = classify_furnished_view(
                observed, ambiguous, target, map_depth, stage_depth, ray_scope, policy,
                stage_ambiguous40=stage_ambiguous,
            )
            row.update({
                "furnished_depth40_m": _json_depth(observed), "target_depth40_m": _json_depth(target),
                "exact_map_depth40_m": _json_depth(map_depth), "stage_depth40_m": _json_depth(stage_depth),
                "ambiguous40": np.asarray(ambiguous, dtype=bool).tolist(), "scope40": ray_scope.tolist(),
            })
        row.update(verdict)
        rows.append(row)
    return rows


def _input_hashes(scene_id: str, hssd_root: Path) -> dict[str, str]:
    root = Path(hssd_root)
    paths = {
        "capture_manifest": root / "UnLoc" / "RGB" / scene_id / "pilot_manifest.json",
        "adapter_metadata": root / "UnLoc" / "frame_adapter_v2" / "frames" / scene_id / "sample_metadata.jsonl",
        "adapter_targets": root / "UnLoc" / "frame_adapter_v2" / "frames" / scene_id / "depth40.npy",
        "geometry": root / "2DFP" / scene_id / "map.png",
        "scope": root / "2DFP_w_semantic" / scene_id / "opening_scope.npy",
        "source_scene": root / "source" / "hssd-hab_local_20261012" / "scenes" / f"{scene_id}.scene_instance.json",
    }
    return {name: _hash_file(path) for name, path in paths.items()}


def scene_is_complete(
    ledger: Path, *, expected_count: int, input_hashes: dict[str, str],
    policy: dict[str, object] | None = None, sensor_mode: str | None = None,
) -> bool:
    marker = ledger.with_suffix(".complete.json")
    if not ledger.is_file() or not marker.is_file():
        return False
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
        rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
        ids = [int(row["sample_index"]) for row in rows]
        return bool(
            payload.get("expected_count") == expected_count
            and payload.get("input_hashes") == input_hashes
            and (policy is None or payload.get("policy") == policy)
            and (sensor_mode is None or payload.get("sensor_mode") == sensor_mode)
            and payload.get("ledger_sha256") == _hash_file(ledger)
            and len(ids) == expected_count and len(ids) == len(set(ids))
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def audit_and_save_scene(
    scene_id: str, split: str, hssd_root: Path, output_root: Path,
    *, gpu_device_id: int = 0, sensor_mode: str = "narrow",
) -> dict[str, object]:
    hashes = _input_hashes(scene_id, hssd_root)
    capture = json.loads((Path(hssd_root) / "UnLoc" / "RGB" / scene_id / "pilot_manifest.json").read_text())
    expected_count = sum(bool(row.get("accepted")) for row in capture["records"])
    ledger = Path(output_root) / "scenes" / f"{scene_id}.jsonl"
    if scene_is_complete(
        ledger, expected_count=expected_count, input_hashes=hashes,
        policy=FurnishedAuditPolicy().to_dict(), sensor_mode=sensor_mode,
    ):
        rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
        return {"scene_id": scene_id, "split": split, "skipped": True, **dict(Counter(r["decision"] for r in rows))}
    rows = audit_scene(scene_id, split, hssd_root, gpu_device_id=gpu_device_id, sensor_mode=sensor_mode)
    ids = [int(row["sample_index"]) for row in rows]
    if len(ids) != expected_count or len(ids) != len(set(ids)):
        raise ValueError(f"{scene_id}: audit ledger is incomplete or duplicated")
    _atomic_text(ledger, "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    _atomic_text(ledger.with_suffix(".complete.json"), json.dumps({
        "scene_id": scene_id, "split": split, "expected_count": expected_count,
        "input_hashes": hashes, "ledger_sha256": _hash_file(ledger),
        "policy": FurnishedAuditPolicy().to_dict(), "sensor_mode": sensor_mode,
    }, indent=2, sort_keys=True) + "\n")
    return {"scene_id": scene_id, "split": split, "skipped": False, **dict(Counter(r["decision"] for r in rows))}


def _work(payload: tuple) -> dict[str, object]:
    scene, split, root, output, device, sensor_mode = payload
    return audit_and_save_scene(scene, split, root, output, gpu_device_id=device, sensor_mode=sensor_mode)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hssd-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--scene", action="append", default=[])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--gpu-device-id", type=int, default=0)
    parser.add_argument("--sensor-mode", choices=("narrow", "full"), default="narrow")
    args = parser.parse_args()
    if args.workers not in (1, 2):
        parser.error("--workers must be 1 or 2 on this single-GPU host")
    output = args.output_root or args.hssd_root / "UnLoc" / AUDIT_DIRECTORY
    splits = _split_scenes(args.hssd_root)
    selected = set(args.scene)
    if selected - set(sum(splits.values(), [])):
        parser.error("requested scene is not in the fixed v2 split")
    work = [(scene, split, args.hssd_root, output, args.gpu_device_id, args.sensor_mode)
            for split, scenes in splits.items() for scene in scenes if not selected or scene in selected]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(_work, work))
    summary = {"scenes": len(results), "totals": dict(Counter({
        status: sum(int(result.get(status, 0)) for result in results)
        for status in ("pass", "review", "fail")
    })), "scene_results": results, "policy": FurnishedAuditPolicy().to_dict()}
    _atomic_text(output / "summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"scenes": summary["scenes"], "totals": summary["totals"]}, sort_keys=True))


if __name__ == "__main__":
    main()

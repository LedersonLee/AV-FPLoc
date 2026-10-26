#!/usr/bin/env python3
"""Replace rejected HSSD views using new indoor poses and the furnished QA gate."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree

from .audit_hssd_unloc_furnished_depth import _scope_near_map_hits
from .audit_hssd_unloc_geometry import _atomic_text, _split_scenes
from .hssd_furnished_2dfp_rays import exact_map_depth40, v2_compatible_target40, verify_scene_target_parity
from .hssd_furnished_depth_policy import AUDIT_DIRECTORY, FurnishedAuditPolicy, classify_furnished_view
from .hssd_furnished_depth_rays import _depth_features
from .hssd_unloc_dataset import WorldRaster, build_capture_schedule
from .render_hssd_unloc_rgb_pilot import (
    KNOWN_CRASHING_HSSD_OBJECT_TEMPLATES,
    _furnished_vertical_clearance,
    _raster_from_metadata,
    _yaw_quaternion,
    configure_simulator_backend,
    write_sanitized_scene_instance,
)


SAMPLER_VERSION = "near_original_025_to_100cm_nohit_v3"


def missing_view_count(*, original_count: int, same_scene_pass_count: int) -> int:
    if original_count < 0 or not 0 <= same_scene_pass_count <= original_count:
        raise ValueError("pass count must be from the same scene and within original count")
    return original_count - same_scene_pass_count


def replacement_pose_cells(
    geometry: np.ndarray, raster: WorldRaster, original_cells: np.ndarray,
    *, limit: int = 400, seed: int = 31415,
) -> np.ndarray:
    """Deterministic farthest-point candidates, at least 0.25 m apart."""

    grid = np.asarray(geometry, dtype=np.uint8)
    original = np.asarray(original_cells, dtype=np.int32).reshape(-1, 2)
    if grid.shape != (raster.height, raster.width) or limit < 0:
        raise ValueError("invalid geometry shape or pose limit")
    free = grid == 255
    clearance = ndimage.distance_transform_edt(free) * raster.spacing_m
    stride = max(1, int(round(0.10 / raster.spacing_m)))
    candidate = np.argwhere((clearance >= 0.10) &
                            (np.indices(grid.shape)[0] % stride == 0) &
                            (np.indices(grid.shape)[1] % stride == 0)).astype(np.int32)
    original_set = {tuple(row) for row in original.tolist()}
    candidate = np.asarray([row for row in candidate if tuple(row) not in original_set], dtype=np.int32).reshape(-1, 2)
    if not len(candidate) or limit == 0:
        return np.empty((0, 2), dtype=np.int32)
    rng = np.random.default_rng(seed)
    if len(original):
        distance_px = cKDTree(original).query(candidate, workers=1)[0]
        nearby = (distance_px >= 0.25 / raster.spacing_m - 1e-6) & (
            distance_px <= 1.0 / raster.spacing_m + 1e-6
        )
        candidate, distance_px = candidate[nearby], distance_px[nearby]
        if not len(candidate):
            return np.empty((0, 2), dtype=np.int32)
        priority = np.abs(distance_px * raster.spacing_m - 0.5)
    else:
        priority = -clearance[candidate[:, 0], candidate[:, 1]]
    shuffled = rng.permutation(len(candidate))
    candidate, priority = candidate[shuffled], priority[shuffled]
    selected: list[np.ndarray] = []
    minimum_sq = (0.25 / raster.spacing_m) ** 2
    for choice in np.argsort(priority, kind="stable"):
        row = candidate[choice]
        if selected and np.min(np.sum((np.asarray(selected) - row) ** 2, axis=1)) < minimum_sq:
            continue
        selected.append(row.copy())
        if len(selected) == limit:
            break
    return np.asarray(selected, dtype=np.int32).reshape(-1, 2)


def capture_rgb_depth_once(sim, record: dict, *, rotation_factory=_yaw_quaternion) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Acquire RGB and the 40 depth features from the identical agent state."""

    state = sim.get_agent(0).get_state()
    state.position = np.asarray(record["pose_xyz_m"], dtype=np.float32)
    state.rotation = rotation_factory(float(record["habitat_yaw_deg"]))
    sim.get_agent(0).set_state(state, reset_sensors=True)
    observation = sim.get_sensor_observations()
    image = np.asarray(observation["rgb"])[..., :3].copy()
    depth, ambiguous = _depth_features(observation["qa_depth"], "narrow")
    return image, depth, ambiguous


def _make_simulator(hssd_root: Path, scene_id: str, gpu_device_id: int, scratch: Path):
    import habitat_sim

    source = hssd_root / "scenes" / f"{scene_id}.scene_instance.json"
    derived, removed = write_sanitized_scene_instance(
        source, scratch, excluded_templates=KNOWN_CRASHING_HSSD_OBJECT_TEMPLATES,
    )
    backend = configure_simulator_backend(
        habitat_sim, root=hssd_root, scene_id=scene_id, gpu_device_id=gpu_device_id,
        scene_instance_path=derived if removed else source,
    )
    specs = []
    for name, kind, shape, fov in (
        ("rgb", habitat_sim.SensorType.COLOR, [480, 640], 106.2602),
        ("clearance_depth", habitat_sim.SensorType.DEPTH, [48, 48], 90.0),
        ("qa_depth", habitat_sim.SensorType.DEPTH, [1, 640], 106.2602),
    ):
        sensor = habitat_sim.CameraSensorSpec()
        sensor.uuid, sensor.sensor_type = name, kind
        sensor.resolution, sensor.hfov = shape, fov
        sensor.position = [0.0, 0.0, 0.0]
        specs.append(sensor)
    agent = habitat_sim.agent.AgentConfiguration()
    agent.sensor_specifications = specs
    return habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent]))


def _navmesh_pose_ok(sim, record: dict) -> bool:
    pose = np.asarray(record["floor_xyz_m"], dtype=np.float32)
    snapped = np.asarray(sim.pathfinder.snap_point(pose), dtype=np.float32)
    return bool(snapped.shape == (3,) and np.isfinite(snapped).all() and
                np.linalg.norm((snapped - pose)[[0, 2]]) <= 0.15 + 1e-3 and
                abs(float(snapped[1] - pose[1])) <= 0.20 + 1e-3)


def _source_rows(dataset_root: Path, scene_id: str) -> tuple[list[dict], np.ndarray]:
    rgb = dataset_root / "UnLoc" / "RGB" / scene_id
    frame = dataset_root / "UnLoc" / "frame_adapter_v2" / "frames" / scene_id
    capture = json.loads((rgb / "pilot_manifest.json").read_text(encoding="utf-8"))["records"]
    accepted = sorted((row for row in capture if row.get("accepted")), key=lambda row: int(row["sample_index"]))
    metadata = [json.loads(line) for line in (frame / "sample_metadata.jsonl").read_text(encoding="utf-8").splitlines()]
    if [int(row["sample_index"]) for row in accepted] != [int(row["sample_index"]) for row in metadata]:
        raise ValueError(f"{scene_id}: source adapter/capture rows differ")
    return capture, np.load(frame / "depth40.npy")


def _load_occluder_mask(dataset_root: Path, scene_id: str, shape: tuple[int, int]) -> np.ndarray:
    path = Path(dataset_root) / "UnLoc" / "structural_targets" / scene_id / "desdf_occluder_mask.png"
    if not path.is_file():
        raise FileNotFoundError(f"{scene_id}: original DeSDF occluder mask is not published: {path}")
    mask = np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > 0
    if mask.shape != shape:
        raise ValueError(f"{scene_id}: original DeSDF occluder mask shape mismatch")
    return mask


def replacement_manifest_is_current(
    manifest: dict, *, split: str, target_count: int,
    audit_ledger_sha256: str, max_new_poses: int,
) -> bool:
    return bool(manifest.get("complete") and manifest.get("split") == split and
                manifest.get("target_count") == target_count and
                manifest.get("audit_ledger_sha256") == audit_ledger_sha256 and
                manifest.get("max_new_poses") == max_new_poses and
                manifest.get("sampler_version") == SAMPLER_VERSION)


def replenish_scene(
    scene_id: str, split: str, target_count: int, hssd_root: Path, dataset_root: Path,
    output_root: Path, seed: int, gpu_device_id: int, *, max_new_poses: int = 400,
) -> dict[str, object]:
    """Capture only passing views from previously unused poses in one scene."""

    dataset_root, hssd_root, output_root = map(Path, (dataset_root, hssd_root, output_root))
    splits = _split_scenes(dataset_root)
    if scene_id not in splits.get(split, []) or sum(scene_id in values for values in splits.values()) != 1:
        raise ValueError(f"{scene_id}: split membership mismatch")
    audit_file = dataset_root / "UnLoc" / AUDIT_DIRECTORY / "scenes" / f"{scene_id}.jsonl"
    audit_marker = audit_file.with_suffix(".complete.json")
    if not audit_marker.is_file():
        raise ValueError(f"{scene_id}: historical furnished audit is incomplete")
    audit_provenance = json.loads(audit_marker.read_text(encoding="utf-8"))
    if audit_provenance.get("policy", {}).get("version") != FurnishedAuditPolicy().version:
        raise ValueError(f"{scene_id}: furnished audit has not been upgraded to the final policy")
    audit_hash = str(audit_provenance.get("ledger_sha256", ""))
    from .audit_hssd_unloc_geometry import _hash_file
    if _hash_file(audit_file) != audit_hash:
        raise ValueError(f"{scene_id}: historical audit ledger hash mismatch")
    audit_rows = [json.loads(line) for line in audit_file.read_text(encoding="utf-8").splitlines()]
    if any(row.get("scene_id") != scene_id or row.get("split") != split for row in audit_rows):
        raise ValueError(f"{scene_id}: audit ledger crossed a split")
    pass_count = sum(row["decision"] == "pass" for row in audit_rows)
    needed = missing_view_count(original_count=target_count, same_scene_pass_count=pass_count)
    scene_output = output_root / scene_id
    manifest_file = scene_output / "replacement_manifest.json"
    if manifest_file.is_file():
        existing = json.loads(manifest_file.read_text(encoding="utf-8"))
        if replacement_manifest_is_current(
            existing, split=split, target_count=target_count,
            audit_ledger_sha256=audit_hash, max_new_poses=max_new_poses,
        ):
            return existing
    source_rows, targets = _source_rows(dataset_root, scene_id)
    accepted_source = sorted((row for row in source_rows if row.get("accepted")), key=lambda row: int(row["sample_index"]))
    if len(accepted_source) != target_count or len(audit_rows) != target_count:
        raise ValueError(f"{scene_id}: original count mismatch")
    fp = dataset_root / "2DFP" / scene_id
    geometry = np.asarray(Image.open(fp / "map.png").convert("L"), dtype=np.uint8)
    occluder = _load_occluder_mask(dataset_root, scene_id, geometry.shape)
    raster = _raster_from_metadata(fp / "floorplan_metadata.json")
    scope = np.load(dataset_root / "2DFP_w_semantic" / scene_id / "opening_scope.npy")
    parity = verify_scene_target_parity(accepted_source, targets, geometry, raster,
                                        occluder_mask=occluder)
    summary: dict[str, object] = {
        "scene_id": scene_id, "split": split, "target_count": target_count,
        "original_qa_pass_count": pass_count, "needed_replacements": needed,
        "target_parity": parity, "replacement_accepted_count": 0, "shortfall": needed,
        "records": [], "complete": True, "audit_ledger_sha256": audit_hash,
        "max_new_poses": max_new_poses, "sampler_version": SAMPLER_VERSION,
    }
    if not parity["passed"]:
        summary["status"] = "target_parity_failed"
    elif needed == 0:
        summary["status"] = "complete"
    else:
        original_cells = np.asarray(sorted({(int(r["grid_row"]), int(r["grid_col"])) for r in source_rows}), dtype=np.int32)
        pose_cells = replacement_pose_cells(geometry, raster, original_cells, limit=max_new_poses, seed=seed)
        floor_y = float(accepted_source[0]["floor_xyz_m"][1])
        policy = FurnishedAuditPolicy()
        rgb_dir = dataset_root / "UnLoc" / "RGB" / scene_id / "rgb_nohit_replacements"
        rgb_dir.mkdir(parents=True, exist_ok=True)
        results: list[dict] = []
        with tempfile.TemporaryDirectory(prefix="hssd_replenish_") as scratch:
            with _make_simulator(hssd_root, scene_id, gpu_device_id, Path(scratch)) as sim:
                for batch_start in range(0, len(pose_cells), 10):
                    if summary["replacement_accepted_count"] >= needed:
                        break
                    candidate_views: list[tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
                    for local_index, cell in enumerate(pose_cells[batch_start:batch_start + 10]):
                        pose_index = batch_start + local_index
                        schedule = build_capture_schedule(cell[None, :], raster, floor_y_m=floor_y)
                        for yaw_index, row in enumerate(schedule):
                            row["pose_index"] = 200 + pose_index
                            row["sample_index"] = 1600 + pose_index * 8 + yaw_index
                        if not _navmesh_pose_ok(sim, schedule[0]):
                            results.extend({**row, "accepted": False, "reasons": ["navmesh_mismatch"]} for row in schedule)
                            continue
                        clearance = _furnished_vertical_clearance(
                            sim, pose_record=schedule[0], heights_m=(0.5, 1.0, 1.5), min_clearance_m=0.30,
                        )
                        if not clearance["accepted"]:
                            results.extend({**row, "accepted": False, "reasons": ["insufficient_vertical_clearance"]} for row in schedule)
                            continue
                        for row in schedule:
                            image, furnished, ambiguous = capture_rgb_depth_once(sim, row)
                            target = v2_compatible_target40(row, geometry, raster,
                                                            occluder_mask=occluder)
                            exact = exact_map_depth40(row, geometry, raster)
                            candidate_views.append((row, image, furnished, ambiguous, target, exact))
                    for row, image, furnished, ambiguous, target, exact in candidate_views:
                        index = int(row["sample_index"])
                        scope40 = _scope_near_map_hits(row, exact, scope, raster)
                        verdict = classify_furnished_view(
                            furnished, ambiguous, target, exact, None, scope40, policy,
                        )
                        admitted = bool(verdict["decision"] == "pass" and summary["replacement_accepted_count"] < needed)
                        result = {**row, **verdict, "accepted": admitted,
                                  "furnished_depth40_m": [None if not np.isfinite(v) else float(v) for v in furnished],
                                  "target_depth40_m": target.astype(float).tolist(),
                                  "exact_map_depth40_m": exact.astype(float).tolist(),
                                  "scene_id": scene_id, "split": split,
                                  "ambiguous40": ambiguous.tolist(),
                                  "scope40": scope40.tolist()}
                        if admitted:
                            image_name = f"{index:05d}.png"
                            Image.fromarray(image).save(rgb_dir / image_name)
                            result["image_path"] = str((rgb_dir / image_name).relative_to(dataset_root))
                            summary["replacement_accepted_count"] += 1
                        results.append(result)
                    summary["records"] = results
                    summary["shortfall"] = needed - int(summary["replacement_accepted_count"])
                    summary["status"] = "in_progress"
                    summary["complete"] = False
                    _atomic_text(manifest_file, json.dumps(summary, sort_keys=True) + "\n")
        summary["records"] = results
        summary["shortfall"] = needed - int(summary["replacement_accepted_count"])
        summary["status"] = "complete" if summary["shortfall"] == 0 else "candidate_budget_exhausted"
        summary["complete"] = True
    _atomic_text(manifest_file, json.dumps(summary, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--hssd-root", type=Path, required=True)
    parser.add_argument("--scene", action="append", default=[])
    parser.add_argument("--gpu-device-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=31415)
    parser.add_argument("--max-new-poses", type=int, default=400)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--workers", type=int, choices=range(1, 17), default=1)
    parser.add_argument("--only-final-audit", action="store_true",
                        help="Process scenes whose v3 QA ledger is ready while others are still being audited.")
    args = parser.parse_args()
    splits = _split_scenes(args.dataset_root)
    selected = set(args.scene)
    if selected - set(sum(splits.values(), [])):
        parser.error("scene not in fixed v2 split")
    output = args.output_root or args.dataset_root / "UnLoc" / AUDIT_DIRECTORY / "replacements"
    work = []
    for split, scenes in splits.items():
        for scene_id in scenes:
            if selected and scene_id not in selected:
                continue
            if args.only_final_audit:
                marker = args.dataset_root / "UnLoc" / AUDIT_DIRECTORY / "scenes" / f"{scene_id}.complete.json"
                if not marker.is_file() or json.loads(marker.read_text(encoding="utf-8")).get(
                    "policy", {},
                ).get("version") != FurnishedAuditPolicy().version:
                    continue
            frame = args.dataset_root / "UnLoc" / "RGB" / scene_id / "pilot_manifest.json"
            target = sum(bool(row.get("accepted")) for row in json.loads(frame.read_text())["records"])
            work.append((scene_id, split, target, args.hssd_root, args.dataset_root,
                         output, args.seed, args.gpu_device_id, args.max_new_poses))
    if args.workers == 1:
        results = map(_work, work)
        for result in results:
            print(json.dumps(result, sort_keys=True), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for result in pool.map(_work, work):
                print(json.dumps(result, sort_keys=True), flush=True)


def _work(payload: tuple) -> dict[str, object]:
    scene_id, split, target, hssd_root, dataset_root, output, seed, device, max_new_poses = payload
    try:
        result = replenish_scene(scene_id, split, target, hssd_root, dataset_root,
                                 output, seed, device, max_new_poses=max_new_poses)
        return {key: result[key] for key in (
            "scene_id", "split", "status", "needed_replacements", "replacement_accepted_count", "shortfall",
        )}
    except Exception as error:
        return {"scene_id": scene_id, "split": split, "status": "error",
                "error": f"{type(error).__name__}: {error}"}


if __name__ == "__main__":
    main()

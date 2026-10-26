#!/usr/bin/env python3
"""Render an inspectable furnished-HSSD RGB pilot with exterior-opening rejection."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from .hssd_unloc_dataset import WorldRaster


KNOWN_CRASHING_HSSD_OBJECT_TEMPLATES = frozenset({
    # This mesh's PBR texture transform aborts Habitat-Sim 0.2.x during the
    # 108736611_177263226 scene import (exit 134).  Keep this explicit rather
    # than silently dropping arbitrary objects.
    "xxxx21d2f3c8xedd4x4619x90b0x62bad36c41cb",
})


def fov_ray_bearings(*, logical_yaw_deg: float, hfov_deg: float, ray_count: int) -> np.ndarray:
    """Return world-X/Z bearings for evenly spaced horizontal camera rays."""

    if not 0 < hfov_deg < 180 or ray_count <= 0:
        raise ValueError("hfov_deg must be in (0, 180) and ray_count must be positive")
    return float(logical_yaw_deg) + np.linspace(-hfov_deg / 2.0, hfov_deg / 2.0, ray_count)


def logical_camera_forward_xz(logical_yaw_deg: float) -> np.ndarray:
    """Camera-forward unit vector in the stored +X-right, +Z-down map frame."""

    angle = math.radians(float(logical_yaw_deg))
    return np.asarray([math.cos(angle), math.sin(angle)], dtype=np.float64)


def habitat_camera_forward_xz(habitat_yaw_deg: float) -> np.ndarray:
    """Habitat camera forward after a Y-axis agent rotation.

    Habitat's identity camera looks along local ``-Z``.  This is kept next to
    the renderer so the schedule's ``logical - 90°`` conversion is testable.
    """

    angle = math.radians(float(habitat_yaw_deg))
    return np.asarray([-math.sin(angle), -math.cos(angle)], dtype=np.float64)


def exclude_for_exterior_opening(first_hit_scopes: np.ndarray, *, max_fraction: float) -> bool:
    """Reject only a view with a material fraction of first hits at an exterior opening."""

    hits = np.asarray(first_hit_scopes, dtype=np.uint8)
    if hits.ndim != 1 or not 0 <= max_fraction <= 1:
        raise ValueError("first-hit scopes must be 1-D and max_fraction must be in [0, 1]")
    return bool(np.mean(hits == 2) > max_fraction)


def even_pose_subset(records: list[dict[str, object]], *, pose_count: int) -> list[dict[str, object]]:
    """Select a deterministic, room-spanning subset from the 200-pose schedule."""

    if pose_count <= 0:
        raise ValueError("pose_count must be positive")
    by_pose = {int(record["pose_index"]): record for record in records}
    ordered = [by_pose[index] for index in sorted(by_pose)]
    if pose_count >= len(ordered):
        return ordered
    indices = np.rint(np.linspace(0, len(ordered) - 1, pose_count)).astype(int)
    return [ordered[int(index)] for index in indices]


def scene_render_is_complete(scene_output: Path, *, pose_count: int) -> bool:
    """Return whether a scene has a complete, auditable RGB render.

    A manifest alone is not enough: an interrupted renderer can leave it or
    its image directory partially written.  Parallel workers may therefore
    skip a scene only after every accepted record has its persisted RGB file.
    """

    manifest_path = Path(scene_output) / "pilot_manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if int(manifest.get("pose_count_requested", -1)) != int(pose_count):
        return False
    records = manifest.get("records")
    if not isinstance(records, list):
        return False
    return all(
        not bool(record.get("accepted"))
        or isinstance(record.get("image"), str)
        and (Path(scene_output) / record["image"]).is_file()
        for record in records
        if isinstance(record, dict)
    )


def horizontal_clearance_from_depth(depth: np.ndarray) -> float:
    """Robust nearest horizontal clearance from one level camera depth image.

    The central third of the image removes floor/ceiling hits.  A low quantile
    rather than a single pixel keeps the decision robust to small rasterisation
    artifacts while still rejecting a shelf, cabinet, or wall occupying the
    camera body.
    """

    image = np.asarray(depth, dtype=np.float32)
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[..., 0]
    if image.ndim != 2:
        raise ValueError("depth must be a 2-D image or H×W×1 array")
    height = image.shape[0]
    band = image[height // 3:(2 * height + 2) // 3, :]
    valid = band[np.isfinite(band) & (band > 0.0)]
    return 0.0 if not len(valid) else float(np.quantile(valid, 0.05))


def vertical_clearance_report(
    depth_sweeps: np.ndarray, *, heights_m: tuple[float, ...], min_clearance_m: float,
) -> dict[str, object]:
    """Require body clearance at every requested height and horizontal bearing.

    ``depth_sweeps`` has shape ``[height, cardinal_yaw, image_row, image_col]``.
    The returned plain values are intentionally persisted in the manifest so a
    reviewer can audit rejected poses instead of treating the sampler as a
    black box.
    """

    sweeps = np.asarray(depth_sweeps, dtype=np.float32)
    if sweeps.ndim != 4 or sweeps.shape[0] != len(heights_m) or sweeps.shape[1] <= 0:
        raise ValueError("depth_sweeps must have shape [len(heights_m), yaw, row, col]")
    if min_clearance_m <= 0:
        raise ValueError("min_clearance_m must be positive")
    per_height = [min(horizontal_clearance_from_depth(frame) for frame in sweeps[index])
                  for index in range(len(heights_m))]
    failing = [float(height) for height, clearance in zip(heights_m, per_height)
               if clearance < min_clearance_m]
    return {
        "accepted": not failing,
        "minimum_clearance_m": float(min(per_height)),
        "per_height_minimum_clearance_m": {
            f"{float(height):.2f}": float(clearance) for height, clearance in zip(heights_m, per_height)
        },
        "failing_heights_m": failing,
    }


def _raster_from_metadata(path: Path) -> WorldRaster:
    metadata = json.loads(Path(path).read_text(encoding="utf-8"))
    bounds = metadata["world_bounds_m"]
    return WorldRaster(
        x_min_m=float(bounds["x_min"]), x_max_m=float(bounds["x_max"]),
        z_min_m=float(bounds["z_min"]), z_max_m=float(bounds["z_max"]),
        spacing_m=float(metadata["map_spacing_m"]),
    )


def _first_hit_scope(
    *, x_m: float, z_m: float, bearing_deg: float, geometry: np.ndarray, scope: np.ndarray,
    raster: WorldRaster, max_range_m: float = 10.0,
) -> int:
    """Return the opening scope at the first blocked geometry cell along one ray."""

    theta = math.radians(float(bearing_deg))
    dx, dz = math.cos(theta), math.sin(theta)
    for distance in np.arange(raster.spacing_m / 2.0, max_range_m, raster.spacing_m / 2.0):
        try:
            row, col = raster.world_to_rc(x_m + distance * dx, z_m + distance * dz)
        except ValueError:
            return 0
        # Geometry deliberately represents portals as free.  Therefore an
        # exterior portal must be observed before testing the first black wall
        # cell; otherwise the ray walks through it and falls off the map.
        if scope[row, col] == 2:
            return 2
        if geometry[row, col] == 0:
            nearby_scope = scope[max(0, row - 1):row + 2, max(0, col - 1):col + 2]
            return int(nearby_scope.max())
    return 0


def exterior_opening_hits(record: dict[str, object], geometry: np.ndarray, scope: np.ndarray, raster: WorldRaster, *, ray_count: int) -> np.ndarray:
    """Evaluate the exact scheduled FoV in the map's logical (+X, +Z) frame."""

    x_m, _, z_m = [float(value) for value in record["pose_xyz_m"]]
    bearings = fov_ray_bearings(
        logical_yaw_deg=float(record["logical_yaw_deg"]), hfov_deg=float(record["camera_hfov_deg"]), ray_count=ray_count,
    )
    return np.asarray([
        _first_hit_scope(x_m=x_m, z_m=z_m, bearing_deg=float(bearing), geometry=geometry, scope=scope, raster=raster)
        for bearing in bearings
    ], dtype=np.uint8)


def _yaw_quaternion(yaw_deg: float):
    import quaternion

    half = math.radians(yaw_deg) / 2.0
    return quaternion.quaternion(math.cos(half), 0.0, math.sin(half), 0.0)


def write_sanitized_scene_instance(
    source: Path, output_dir: Path, *, excluded_templates: frozenset[str] | set[str],
) -> tuple[Path, list[str]]:
    """Write an auditable derived scene instance without known crashing assets."""

    source = Path(source)
    payload = json.loads(source.read_text(encoding="utf-8"))
    excluded = set(excluded_templates)
    removed = [
        str(entry.get("template_name")) for entry in payload.get("object_instances", [])
        if str(entry.get("template_name")) in excluded
    ]
    payload["object_instances"] = [
        entry for entry in payload.get("object_instances", [])
        if str(entry.get("template_name")) not in excluded
    ]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{source.stem}.sanitized.scene_instance.json"
    output.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return output, removed


def configure_simulator_backend(
    habitat_sim, *, root: Path, scene_id: str, gpu_device_id: int, scene_instance_path: Path | None = None,
):
    """Build an HSSD renderer backend with an explicit non-empty light key.

    HSSD's dataset config intentionally declares ``default_lighting: ""``.
    Habitat-Sim 0.2.x accepts that when loading some older assets but the
    current renderer aborts while resolving the empty lighting layout.  The
    built-in default key is deterministic and preserves the scene geometry and
    camera contract; it only supplies the missing renderer configuration.
    """

    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = str(scene_instance_path or root / "scenes" / f"{scene_id}.scene_instance.json")
    backend.scene_dataset_config_file = str(root / "hssd-hab.scene_dataset_config.json")
    backend.gpu_device_id, backend.enable_physics = gpu_device_id, False
    backend.scene_light_setup = habitat_sim.gfx.DEFAULT_LIGHTING_KEY
    backend.override_scene_light_defaults = True
    return backend


def _create_simulator(
    habitat_sim, *, root: Path, scene_id: str, gpu_device_id: int, resolution: tuple[int, int], hfov_deg: float,
    clearance_resolution: tuple[int, int] = (48, 48), clearance_hfov_deg: float = 90.0,
    scene_instance_path: Path | None = None,
):
    backend = configure_simulator_backend(
        habitat_sim, root=root, scene_id=scene_id, gpu_device_id=gpu_device_id, scene_instance_path=scene_instance_path,
    )
    rgb = habitat_sim.CameraSensorSpec()
    rgb.uuid, rgb.resolution, rgb.hfov = "rgb", list(resolution), float(hfov_deg)
    rgb.position = [0.0, 0.0, 0.0]
    depth = habitat_sim.CameraSensorSpec()
    depth.uuid, depth.resolution, depth.hfov = "clearance_depth", list(clearance_resolution), float(clearance_hfov_deg)
    depth.position, depth.sensor_type = [0.0, 0.0, 0.0], habitat_sim.SensorType.DEPTH
    agent = habitat_sim.agent.AgentConfiguration()
    agent.sensor_specifications = [rgb, depth]
    return habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent]))


def _furnished_vertical_clearance(
    sim, *, pose_record: dict[str, object], heights_m: tuple[float, ...], min_clearance_m: float,
) -> dict[str, object]:
    """Probe the furnished scene at 0.5/1.0/1.5 m before allowing a query pose."""

    x_m, _, z_m = [float(value) for value in pose_record["pose_xyz_m"]]
    floor_y_m = float(pose_record["floor_xyz_m"][1])
    frames: list[list[np.ndarray]] = []
    for height_m in heights_m:
        sweeps: list[np.ndarray] = []
        for logical_yaw_deg in (0.0, 90.0, 180.0, 270.0):
            state = sim.get_agent(0).get_state()
            state.position = np.asarray([x_m, floor_y_m + height_m, z_m], dtype=np.float32)
            state.rotation = _yaw_quaternion(-logical_yaw_deg - 90.0)
            sim.get_agent(0).set_state(state, reset_sensors=True)
            sweeps.append(np.asarray(sim.get_sensor_observations()["clearance_depth"], dtype=np.float32))
        frames.append(sweeps)
    return vertical_clearance_report(np.asarray(frames), heights_m=heights_m, min_clearance_m=min_clearance_m)


def render_scene(
    *, scene_id: str, hssd_root: Path, floorplans_root: Path, semantic_root: Path, output_root: Path,
    pose_count: int, ray_count: int, exterior_max_fraction: float, gpu_device_id: int,
    clearance_heights_m: tuple[float, ...] = (0.5, 1.0, 1.5), min_clearance_m: float = 0.30,
) -> dict[str, object]:
    """Render one scene's filtered pilot views and record every acceptance decision."""

    import habitat_sim

    floorplan = Path(floorplans_root) / scene_id
    semantic = Path(semantic_root) / "semantic_maps" / scene_id
    records = [json.loads(line) for line in (floorplan / "capture_schedule.jsonl").read_text(encoding="utf-8").splitlines()]
    chosen_poses = {int(record["pose_index"]) for record in even_pose_subset(records, pose_count=pose_count)}
    candidates = [record for record in records if int(record["pose_index"]) in chosen_poses]
    geometry = np.asarray(Image.open(floorplan / "map.png").convert("L"), dtype=np.uint8)
    scope = np.load(semantic / "opening_scope.npy")
    raster = _raster_from_metadata(floorplan / "floorplan_metadata.json")
    hfov = float(candidates[0]["camera_hfov_deg"])
    resolution = (int(candidates[0]["camera_height_px"]), int(candidates[0]["camera_width_px"]))
    scene_output = Path(output_root) / scene_id
    rgb_output = scene_output / "rgb"
    rgb_output.mkdir(parents=True, exist_ok=True)
    source_scene = Path(hssd_root) / "scenes" / f"{scene_id}.scene_instance.json"
    sanitized_scene, removed_templates = write_sanitized_scene_instance(
        source_scene, scene_output / "renderer_scene_instance",
        excluded_templates=KNOWN_CRASHING_HSSD_OBJECT_TEMPLATES,
    )
    scene_instance_path = sanitized_scene if removed_templates else source_scene
    sim = _create_simulator(
        habitat_sim, root=Path(hssd_root), scene_id=scene_id, gpu_device_id=gpu_device_id,
        resolution=resolution, hfov_deg=hfov, scene_instance_path=scene_instance_path,
    )
    try:
        pose_records = {int(record["pose_index"]): record for record in candidates}
        clearance_by_pose = {
            pose_index: _furnished_vertical_clearance(
                sim, pose_record=record, heights_m=clearance_heights_m, min_clearance_m=min_clearance_m,
            ) for pose_index, record in pose_records.items()
        }
        decisions: list[dict[str, object]] = []
        accepted: list[dict[str, object]] = []
        for record in candidates:
            hits = exterior_opening_hits(record, geometry, scope, raster, ray_count=ray_count)
            exterior_fraction = float(np.mean(hits == 2))
            clearance = clearance_by_pose[int(record["pose_index"])]
            decision = {
                **record, "first_hit_exterior_fraction": exterior_fraction,
                "first_hit_exterior_rays": int((hits == 2).sum()),
                "vertical_clearance": clearance,
            }
            if not bool(clearance["accepted"]):
                decision["accepted"], decision["exclusion_reason"] = False, "insufficient_vertical_clearance"
            elif exclude_for_exterior_opening(hits, max_fraction=exterior_max_fraction):
                decision["accepted"], decision["exclusion_reason"] = False, "exterior_opening_in_fov"
            else:
                decision["accepted"] = True
                accepted.append(decision)
            decisions.append(decision)
        for record in accepted:
            state = sim.get_agent(0).get_state()
            state.position = np.asarray(record["pose_xyz_m"], dtype=np.float32)
            state.rotation = _yaw_quaternion(float(record["habitat_yaw_deg"]))
            sim.get_agent(0).set_state(state, reset_sensors=True)
            image = np.asarray(sim.get_sensor_observations()["rgb"])[..., :3]
            image_name = f"{int(record['sample_index']):05d}.png"
            Image.fromarray(image).save(rgb_output / image_name)
            record["image"] = f"rgb/{image_name}"
            record["rgb_shape"] = list(image.shape)
    finally:
        sim.close()
    summary = {
        "scene_id": scene_id, "variant": "furnished", "renderer": "Habitat-Sim", "pose_count_requested": pose_count,
        "scene_instance_source": str(source_scene),
        "scene_instance_rendered": str(scene_instance_path),
        "excluded_crashing_object_templates": removed_templates,
        "candidate_view_count": len(candidates), "accepted_view_count": len(accepted),
        "excluded_view_count": len(candidates) - len(accepted),
        "excluded_clearance_view_count": sum(r.get("exclusion_reason") == "insufficient_vertical_clearance" for r in decisions),
        "excluded_exterior_view_count": sum(r.get("exclusion_reason") == "exterior_opening_in_fov" for r in decisions), "ray_count": ray_count,
        "exterior_max_fraction": exterior_max_fraction, "camera_contract": {
            "hfov_deg": hfov, "resolution_px": [resolution[1], resolution[0]], "camera_height_m": 1.5,
        }, "furnished_clearance_contract": {
            "heights_above_floor_m": list(clearance_heights_m), "cardinal_logical_yaws_deg": [0, 90, 180, 270],
            "depth_resolution_px": [48, 48], "depth_hfov_deg": 90.0, "minimum_required_clearance_m": min_clearance_m,
            "metric": "minimum 5th-percentile depth in central image band across height/yaw probes",
        }, "records": decisions,
    }
    (scene_output / "pilot_manifest.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def ensure_legacy_capture_destination(output_root: Path, *, allow_legacy_ungated: bool) -> None:
    """Keep the old RGB pilot from silently overwriting canonical QA-gated data."""

    canonical = Path(output_root).resolve().parts[-3:] == ("HSSD", "UnLoc", "RGB")
    if canonical and not allow_legacy_ungated:
        raise ValueError(
            "canonical HSSD/UnLoc/RGB requires the furnished-depth QA gate; "
            "use replenish_hssd_unloc_furnished_views.py or explicitly opt into legacy ungated capture"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hssd-root", type=Path, required=True)
    parser.add_argument("--floorplans-root", type=Path, required=True)
    parser.add_argument("--semantic-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scene", action="append", required=True)
    parser.add_argument("--pose-count", type=int, default=20)
    parser.add_argument("--ray-count", type=int, default=40)
    parser.add_argument("--exterior-max-fraction", type=float, default=0.25)
    parser.add_argument("--min-clearance-m", type=float, default=0.30)
    parser.add_argument("--gpu-device-id", type=int, default=0)
    parser.add_argument("--skip-completed", action="store_true",
                        help="Do not rerender a scene whose manifest and accepted RGB files are complete.")
    parser.add_argument("--allow-legacy-ungated", action="store_true",
                        help="Explicitly allow this pre-QA pilot to write the canonical RGB folder.")
    args = parser.parse_args()
    try:
        ensure_legacy_capture_destination(args.output_root,
                                          allow_legacy_ungated=args.allow_legacy_ungated)
    except ValueError as error:
        parser.error(str(error))
    requested_scenes = list(args.scene)
    pending_scenes = [scene_id for scene_id in requested_scenes if not args.skip_completed or not scene_render_is_complete(
        args.output_root / scene_id, pose_count=args.pose_count,
    )]
    summaries = [render_scene(
        scene_id=scene_id, hssd_root=args.hssd_root, floorplans_root=args.floorplans_root,
        semantic_root=args.semantic_root, output_root=args.output_root, pose_count=args.pose_count,
        ray_count=args.ray_count, exterior_max_fraction=args.exterior_max_fraction,
        gpu_device_id=args.gpu_device_id, min_clearance_m=args.min_clearance_m,
    ) for scene_id in pending_scenes]
    payload = {"schema_version": "hssd-unloc-rgb-pilot-v1", "scenes": summaries}
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "pilot_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"scenes": len(summaries), "accepted_views": sum(s["accepted_view_count"] for s in summaries)}, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Reconcile a canonical HSSD map.png with Habitat structural connectivity.

The semantic polygons define the indoor candidate domain.  The stage wall
slice supplies the initial obstacles.  Habitat's navmesh is used only as an
independent connectivity oracle: when it connects two indoor regions that the
map splits, its path may carve a narrow *interior* corridor.  The semantic
outer contour is never carved.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from .hssd_unloc_dataset import (
    EXTERIOR_REGION_LABELS,
    WorldRaster,
    _polygon_mask,
    audit_map_connectivity,
    build_capture_schedule,
    carve_internal_navmesh_path,
    indoor_region_annotations,
    sample_poses,
    write_floorplan_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-instance", type=Path, required=True)
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--floorplan-dir", type=Path, required=True)
    parser.add_argument("--agent-height-m", type=float, default=1.5)
    parser.add_argument("--agent-radius-m", type=float, default=0.10)
    parser.add_argument("--navmesh-cell-size-m", type=float, default=0.01)
    parser.add_argument("--horizontal-tolerance-m", type=float, default=0.10)
    parser.add_argument("--repair-radius-m", type=float, default=0.10)
    parser.add_argument("--pose-count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=31415)
    parser.add_argument("--gpu-device-id", type=int, default=0)
    return parser.parse_args()


def load_raster(metadata: dict[str, object]) -> WorldRaster:
    bounds = metadata["world_bounds_m"]
    assert isinstance(bounds, dict)
    return WorldRaster(
        x_min_m=float(bounds["x_min"]), x_max_m=float(bounds["x_max"]),
        z_min_m=float(bounds["z_min"]), z_max_m=float(bounds["z_max"]),
        spacing_m=float(metadata["map_spacing_m"]),
    )


def region_names(regions: list[dict[str, object]]) -> list[str]:
    """Return deterministic, unique names even when HSSD labels repeat."""

    return [f"{index:02d}:{str(region.get('label', 'unlabelled'))}" for index, region in enumerate(regions)]


def representative_region_cells(
    regions: list[dict[str, object]], raster: WorldRaster, free_mask: np.ndarray,
) -> dict[str, tuple[int, int]]:
    """Choose the highest-clearance map cell inside every semantic region."""

    free = np.asarray(free_mask, dtype=bool)
    clearance = ndimage.distance_transform_edt(free)
    representatives: dict[str, tuple[int, int]] = {}
    for name, region in zip(region_names(regions), regions):
        loop = np.asarray(region.get("poly_loop", []), dtype=np.float64)
        mask = _polygon_mask(loop[:, [0, 2]], raster)
        candidates = free & mask
        if not candidates.any():
            raise ValueError(f"semantic region {name} has no free map cell")
        row, col = np.unravel_index(np.argmax(np.where(candidates, clearance, -1)), free.shape)
        representatives[name] = (int(row), int(col))
    return representatives


def exterior_region_mask(raw_regions: list[dict[str, object]], raster: WorldRaster) -> np.ndarray:
    """Rasterize explicitly labelled HSSD outdoor support with a small seam buffer."""

    mask = np.zeros((raster.height, raster.width), dtype=bool)
    for region in raw_regions:
        if str(region.get("label", "")).strip().lower() not in EXTERIOR_REGION_LABELS:
            continue
        loop = np.asarray(region.get("poly_loop", []), dtype=np.float64)
        if loop.ndim == 2 and loop.shape[1] >= 3 and len(loop) >= 3:
            mask |= _polygon_mask(loop[:, [0, 2]], raster)
    # Region annotations often leave a centimetre-scale seam at the outdoor
    # edge.  Treat it as exterior only for route classification, never as a
    # source of new free map cells.
    return ndimage.binary_dilation(mask, iterations=int(np.ceil(0.20 / raster.spacing_m))) if mask.any() else mask


def path_indoor_summary(
    path_xz_m: np.ndarray, indoor_mask: np.ndarray, raster: WorldRaster, *, exterior_mask: np.ndarray | None = None,
) -> dict[str, object]:
    """Describe whether a navmesh route remains inside semantic indoor support."""

    points = np.asarray(path_xz_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        return {"stays_inside": False, "sample_count": 0, "outside_samples": [], "external_only": False}
    exterior = np.zeros_like(indoor_mask, dtype=bool) if exterior_mask is None else np.asarray(exterior_mask, dtype=bool)
    if exterior.shape != indoor_mask.shape:
        raise ValueError("exterior_mask must match indoor_mask")
    sample_count = 0
    outside_samples: list[list[float]] = []
    outside_in_exterior_count = 0
    for start, end in zip(points, points[1:]):
        length_m = float(np.linalg.norm(end - start))
        count = max(2, int(np.ceil(length_m / (raster.spacing_m / 2))) + 1)
        for x_m, z_m in np.linspace(start, end, count):
            sample_count += 1
            try:
                row, col = raster.world_to_rc(float(x_m), float(z_m))
            except ValueError:
                outside_samples.append([float(x_m), float(z_m)])
                continue
            if not indoor_mask[row, col]:
                outside_samples.append([float(x_m), float(z_m)])
                outside_in_exterior_count += int(exterior[row, col])
    return {
        "stays_inside": not outside_samples,
        "sample_count": sample_count,
        "outside_sample_count": len(outside_samples),
        "outside_in_exterior_count": outside_in_exterior_count,
        "outside_samples": outside_samples[:5],
        "external_only": bool(outside_samples) and outside_in_exterior_count == len(outside_samples),
    }


def indoor_navmesh_components(
    names: list[str], navmesh_reachable: dict[tuple[str, str], bool], *,
    exterior_only_pairs: set[tuple[str, str]] | None = None,
) -> list[list[str]]:
    """Return indoor-only connected components of semantic region anchors.

    A Habitat path that reaches another region only by crossing labelled
    exterior support does not merge buildings in the training domain.
    """

    exterior_only = exterior_only_pairs or set()
    adjacency = {name: set() for name in names}
    for left, right in combinations(names, 2):
        key = (left, right)
        if navmesh_reachable.get(key, False) and key not in exterior_only:
            adjacency[left].add(right)
            adjacency[right].add(left)
    components: list[list[str]] = []
    unseen = set(names)
    while unseen:
        pending = [min(unseen)]
        component: list[str] = []
        while pending:
            current = pending.pop()
            if current not in unseen:
                continue
            unseen.remove(current)
            component.append(current)
            pending.extend(sorted(adjacency[current] & unseen, reverse=True))
        components.append(sorted(component))
    return sorted(components, key=lambda component: component[0])


def bulk_approved(
    *,
    unresolved_false_closed_count: int,
    false_open_count: int,
    skipped_pairs: list[dict[str, object]],
    indoor_navmesh_component_count: int,
) -> bool:
    """Whether a map is valid for independent-frame vision training.

    A scene can contain multiple disconnected buildings on one floor.  These
    remain valid visual contexts because samples do not require paths between
    buildings; the component count is retained solely for review metadata.
    """
    _ = indoor_navmesh_component_count
    return unresolved_false_closed_count == 0 and false_open_count == 0 and not skipped_pairs


def _save_schedule(
    floorplan_dir: Path, regions: list[dict[str, object]], raster: WorldRaster, *, count: int, seed: int,
) -> int:
    free = np.asarray(Image.open(floorplan_dir / "map.png").convert("L"), dtype=np.uint8) == 255
    poses = sample_poses(free, raster, count=count, min_clearance_m=0.10, seed=seed)
    floor_y = float(regions[0].get("floor_height", 0.0))
    schedule = build_capture_schedule(poses, raster, floor_y_m=floor_y)
    np.save(floorplan_dir / "pose_grid_rc.npy", poses)
    with (floorplan_dir / "capture_schedule.jsonl").open("w", encoding="utf-8") as stream:
        for record in schedule:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    return int(len(poses))


def reconcile(args: argparse.Namespace) -> dict[str, object]:
    floorplan_dir = args.floorplan_dir.resolve()
    required = [
        args.scene_instance, args.dataset_config, floorplan_dir / "floorplan_metadata.json",
        floorplan_dir / "map.png", floorplan_dir / "indoor_mask.png",
        floorplan_dir / "outer_boundary_mask.png", floorplan_dir / "portal_mask.png",
        floorplan_dir / "stage_wall_slice.npy",
    ]
    if not all(path.is_file() for path in required):
        raise FileNotFoundError("scene, config, and initial floorplan artifacts must exist")
    metadata_path = floorplan_dir / "floorplan_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    semantic_path = Path(str(metadata["semantic_path"]))
    if not semantic_path.is_file():
        raise FileNotFoundError(f"missing semantic config: {semantic_path}")
    raw_regions = json.loads(semantic_path.read_text(encoding="utf-8")).get("region_annotations", [])
    regions = indoor_region_annotations(raw_regions)
    if len(regions) < 2:
        raise ValueError("connectivity reconciliation requires at least two semantic regions")
    raster = load_raster(metadata)
    free_before = np.asarray(Image.open(floorplan_dir / "map.png").convert("L"), dtype=np.uint8) == 255
    indoor = np.asarray(Image.open(floorplan_dir / "indoor_mask.png").convert("L"), dtype=np.uint8) == 255
    outer = np.asarray(Image.open(floorplan_dir / "outer_boundary_mask.png").convert("L"), dtype=np.uint8) == 255
    labelled_exterior = exterior_region_mask(raw_regions, raster)
    representatives = representative_region_cells(regions, raster, free_before)

    import habitat_sim
    from magnum import Vector3

    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = str(args.scene_instance)
    backend.scene_dataset_config_file = str(args.dataset_config)
    backend.gpu_device_id = int(args.gpu_device_id)
    backend.enable_physics = False
    agent = habitat_sim.agent.AgentConfiguration()
    agent.height = float(args.agent_height_m)
    agent.radius = float(args.agent_radius_m)
    simulator = habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent]))
    try:
        settings = habitat_sim.NavMeshSettings()
        settings.set_defaults()
        settings.agent_height = float(args.agent_height_m)
        settings.agent_radius = float(args.agent_radius_m)
        settings.cell_size = float(args.navmesh_cell_size_m)
        if not simulator.recompute_navmesh(simulator.pathfinder, settings):
            raise RuntimeError("Habitat failed to recompute the structural navmesh")
        floor_y = float(regions[0].get("floor_height", 0.0))
        anchors: dict[str, np.ndarray] = {}
        anchor_snap_error_m: dict[str, float] = {}
        for name, (row, col) in representatives.items():
            x_m, z_m = raster.rc_to_world(row, col)
            requested = np.asarray([x_m, floor_y + 0.20, z_m], dtype=np.float64)
            snapped = np.asarray(simulator.pathfinder.snap_point(Vector3(*requested)), dtype=np.float64)
            error_m = float(np.linalg.norm((snapped - requested)[[0, 2]]))
            if not np.isfinite(snapped).all() or error_m > args.horizontal_tolerance_m:
                raise RuntimeError(f"region anchor {name} does not agree with navmesh (snap {error_m:.3f} m)")
            anchors[name] = snapped
            anchor_snap_error_m[name] = error_m
        paths: dict[tuple[str, str], np.ndarray] = {}
        navmesh_reachable: dict[tuple[str, str], bool] = {}
        for left, right in combinations(region_names(regions), 2):
            path = habitat_sim.ShortestPath()
            path.requested_start = Vector3(*anchors[left])
            path.requested_end = Vector3(*anchors[right])
            found = bool(simulator.pathfinder.find_path(path))
            navmesh_reachable[(left, right)] = found
            if found:
                paths[(left, right)] = np.asarray([[point[0], point[2]] for point in path.points], dtype=np.float64)
    finally:
        simulator.close()

    before = audit_map_connectivity(free_before, representatives, navmesh_reachable)
    path_summaries = {
        key: path_indoor_summary(path, indoor, raster, exterior_mask=labelled_exterior)
        for key, path in paths.items()
    }
    exterior_only_pair_keys = {key for key, summary in path_summaries.items() if summary["external_only"]}
    navmesh_components = indoor_navmesh_components(
        region_names(regions), navmesh_reachable, exterior_only_pairs=exterior_only_pair_keys,
    )
    raw_wall = np.load(floorplan_dir / "stage_wall_slice.npy").astype(bool)
    repaired_wall = raw_wall.copy()
    repaired_pairs: list[dict[str, object]] = []
    skipped_pairs: list[dict[str, object]] = []
    exterior_transit_pairs: list[dict[str, object]] = []
    for pair in before["pairs"]:
        if pair["status"] != "false_closed":
            continue
        key = (str(pair["left_region"]), str(pair["right_region"]))
        path = paths[key]
        inside = path_summaries[key]
        if not inside["stays_inside"]:
            destination = exterior_transit_pairs if inside["external_only"] else skipped_pairs
            destination.append({**pair, "reason": "navmesh path leaves semantic indoor support", "path_indoor": inside})
            continue
        repaired_wall = carve_internal_navmesh_path(
            repaired_wall, indoor_mask=indoor, outer_boundary_mask=outer, raster=raster,
            path_xz_m=path, corridor_radius_m=float(args.repair_radius_m),
        )
        repaired_pairs.append({**pair, "path_point_count": int(len(path))})

    portal = np.asarray(Image.open(floorplan_dir / "portal_mask.png").convert("L"), dtype=np.uint8) == 255
    write_floorplan_artifacts(
        floorplan_dir, regions, raster,
        scene_id=str(metadata["scene_id"]), semantic_path=semantic_path,
        wall_thickness_m=float(metadata["wall_thickness_m"]),
        mesh_wall_mask=repaired_wall, portal_mask=portal,
    )
    free_after = np.asarray(Image.open(floorplan_dir / "map.png").convert("L"), dtype=np.uint8) == 255
    after = audit_map_connectivity(free_after, representatives, navmesh_reachable)
    repair_mask = raw_wall & ~repaired_wall
    Image.fromarray(np.where(repair_mask, 255, 0).astype(np.uint8)).save(floorplan_dir / "connectivity_repair_mask.png")
    pose_count = _save_schedule(floorplan_dir, regions, raster, count=int(args.pose_count), seed=int(args.seed))
    exterior_pair_keys = {
        (str(pair["left_region"]), str(pair["right_region"])) for pair in exterior_transit_pairs
    }
    unresolved_false_closed = [
        pair for pair in after["pairs"]
        if pair["status"] == "false_closed"
        and (str(pair["left_region"]), str(pair["right_region"])) not in exterior_pair_keys
    ]
    report: dict[str, object] = {
        "scene_instance": str(args.scene_instance),
        "dataset_config": str(args.dataset_config),
        "floorplan_dir": str(floorplan_dir),
        "agent_height_m": float(args.agent_height_m),
        "agent_radius_m": float(args.agent_radius_m),
        "navmesh_cell_size_m": float(args.navmesh_cell_size_m),
        "repair_radius_m": float(args.repair_radius_m),
        "representative_cells_rc": {name: list(point) for name, point in representatives.items()},
        "anchor_snap_error_m": anchor_snap_error_m,
        "before": before,
        "repaired_pairs": repaired_pairs,
        "skipped_pairs": skipped_pairs,
        "exterior_transit_pairs": exterior_transit_pairs,
        "indoor_navmesh_components": navmesh_components,
        "indoor_navmesh_component_count": len(navmesh_components),
        "single_indoor_navmesh_component": len(navmesh_components) == 1,
        "after": after,
        "unresolved_false_closed_pair_count": len(unresolved_false_closed),
        "repair_pixel_count": int(repair_mask.sum()),
        "pose_count": pose_count,
        "approved_for_bulk": bulk_approved(
            unresolved_false_closed_count=len(unresolved_false_closed),
            false_open_count=int(after["false_open_pair_count"]),
            skipped_pairs=skipped_pairs,
            indoor_navmesh_component_count=len(navmesh_components),
        ),
    }
    (floorplan_dir / "connectivity_validation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    updated_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    updated_metadata["connectivity_validation"] = {
        "method": "semantic-region representatives + Habitat structural navmesh pairwise paths",
        "report": "connectivity_validation.json",
        "approved_for_bulk": report["approved_for_bulk"],
    }
    metadata_path.write_text(json.dumps(updated_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    report = reconcile(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["approved_for_bulk"]:
        raise SystemExit("floorplan connectivity validation did not pass")


if __name__ == "__main__":
    main()

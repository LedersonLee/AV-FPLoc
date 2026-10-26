#!/usr/bin/env python3
"""Export separate binary geometry maps and door/window semantic rasters for HSSD.

The canonical geometry map stays binary (white=indoor free, black=wall or
exterior).  This tool deliberately writes semantic labels beside it instead
of changing its training/deSDF contract.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
from scipy import ndimage

from .hssd_unloc_dataset import WorldRaster, wall_mask_from_vertical_triangles


LABEL_FREE = np.uint8(0)
LABEL_WALL = np.uint8(1)
LABEL_DOOR = np.uint8(2)
LABEL_WINDOW = np.uint8(3)
LABEL_EXTERIOR_DOOR = np.uint8(4)
LABEL_EXTERIOR_WINDOW = np.uint8(5)

OPENING_SCOPE_NONE = np.uint8(0)
OPENING_SCOPE_INTERNAL = np.uint8(1)
OPENING_SCOPE_EXTERIOR = np.uint8(2)

PALETTE = np.array([
    [255, 255, 255],  # free
    [0, 0, 0],        # wall / exterior
    [220, 38, 38],    # door
    [37, 99, 235],    # window / glass
    [245, 158, 11],   # exterior door
    [6, 182, 212],    # exterior window / glass
], dtype=np.uint8)


def opening_template_id(instance_template_name: str) -> str:
    """Collapse Habitat decomposed opening parts to their asset template ID."""

    return str(instance_template_name).split("_part_", 1)[0]


def load_opening_categories(metadata_csv: Path) -> dict[str, np.uint8]:
    """Read the authoritative floorplanner door/window category for openings."""

    categories: dict[str, np.uint8] = {}
    with Path(metadata_csv).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            category = str(row.get("main_category", "")).strip().lower()
            if category == "door":
                categories[str(row["id"])] = LABEL_DOOR
            elif category == "window":
                categories[str(row["id"])] = LABEL_WINDOW
    return categories


def compose_semantic_labels(
    geometry_map: np.ndarray, *, door_mask: np.ndarray, window_mask: np.ndarray,
    exterior_door_mask: np.ndarray | None = None, exterior_window_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Overlay internal/exterior opening semantics on a binary geometry map.

    Geometry stays indoor-only, so exterior openings intentionally paint a
    coloured line over its black exterior contour.  This makes the semantic
    relationship visible without making the exterior navigable in ``map.png``.
    """

    geometry = np.asarray(geometry_map, dtype=np.uint8)
    door = np.asarray(door_mask, dtype=bool)
    window = np.asarray(window_mask, dtype=bool)
    exterior_door = np.zeros_like(door) if exterior_door_mask is None else np.asarray(exterior_door_mask, dtype=bool)
    exterior_window = np.zeros_like(window) if exterior_window_mask is None else np.asarray(exterior_window_mask, dtype=bool)
    if (geometry.ndim != 2 or door.shape != geometry.shape or window.shape != geometry.shape
            or exterior_door.shape != geometry.shape or exterior_window.shape != geometry.shape):
        raise ValueError("geometry map and opening masks must be equal-size 2-D arrays")
    labels = np.where(geometry == 255, LABEL_FREE, LABEL_WALL).astype(np.uint8)
    labels[exterior_window] = LABEL_EXTERIOR_WINDOW
    labels[exterior_door] = LABEL_EXTERIOR_DOOR
    labels[window] = LABEL_WINDOW
    labels[door] = LABEL_DOOR
    return labels


def labels_to_rgb(labels: np.ndarray) -> np.ndarray:
    """Encode free/wall/door/window labels with a review-friendly fixed palette."""

    array = np.asarray(labels)
    if array.ndim != 2 or np.any(array < LABEL_FREE) or np.any(array > LABEL_EXTERIOR_WINDOW):
        raise ValueError("semantic labels must be a 2-D array containing labels 0..5")
    return PALETTE[array.astype(np.uint8)]


def structural_free_wall_overlap_pixels(geometry_map: np.ndarray, wall_mask: np.ndarray) -> int:
    """Count impossible cells that are free in the map but blocked by stage wall mesh."""

    geometry = np.asarray(geometry_map, dtype=np.uint8)
    wall = np.asarray(wall_mask, dtype=np.uint8)
    if geometry.ndim != 2 or wall.shape != geometry.shape:
        raise ValueError("geometry and wall mask must be equal-size 2-D arrays")
    return int(np.count_nonzero((geometry == 255) & (wall > 0)))


def classify_opening_scope(
    door_mask: np.ndarray, window_mask: np.ndarray, outer_boundary_mask: np.ndarray, *, spacing_m: float,
) -> np.ndarray:
    """Label opening pixels as internal or on the closed exterior contour.

    A one-decimetre tolerance covers the wall thickness / raster quantisation
    mismatch between an independently rasterised opening asset and the stage
    boundary.  The result is intentionally separate from semantic labels so
    geometry and semantic baselines remain unaffected.
    """

    door = np.asarray(door_mask, dtype=bool)
    window = np.asarray(window_mask, dtype=bool)
    outer = np.asarray(outer_boundary_mask, dtype=bool)
    if spacing_m <= 0 or door.shape != window.shape or door.shape != outer.shape:
        raise ValueError("opening and outer-boundary masks must share shape and positive spacing")
    opening = door | window
    near_outer = ndimage.binary_dilation(outer, iterations=max(1, int(np.ceil(0.10 / spacing_m))))
    scope = np.zeros(door.shape, dtype=np.uint8)
    scope[opening] = OPENING_SCOPE_INTERNAL
    scope[opening & near_outer] = OPENING_SCOPE_EXTERIOR
    return scope


def split_opening_masks_for_training(
    instance_masks: list[tuple[np.uint8, np.ndarray]], *, indoor_mask: np.ndarray,
    free_mask: np.ndarray, outer_boundary_mask: np.ndarray, spacing_m: float,
) -> dict[str, np.ndarray]:
    """Separate indoor semantic targets from exterior and terrace-only opening meshes.

    The HSSD opening asset contains physical door leaves and window frames,
    which can extend far from the wall plane.  Classification is therefore
    performed per *instance*, not per coloured pixel: an asset touching the
    outer contour is exterior; an asset with no indoor-support neighbourhood is
    terrace/outside-only and is not part of the indoor map at all.
    """

    indoor = np.asarray(indoor_mask, dtype=bool)
    free = np.asarray(free_mask, dtype=bool)
    outer = np.asarray(outer_boundary_mask, dtype=bool)
    if spacing_m <= 0 or indoor.shape != outer.shape or free.shape != indoor.shape:
        raise ValueError("indoor, free and outer masks must share shape and positive spacing")
    names = ("training_door", "training_window", "exterior_door", "exterior_window", "dropped_door", "dropped_window")
    groups = {name: np.zeros(indoor.shape, dtype=bool) for name in names}
    near_indoor = ndimage.binary_dilation(indoor, iterations=max(1, int(np.ceil(0.20 / spacing_m))))
    near_free = ndimage.binary_dilation(free, iterations=max(1, int(np.ceil(0.20 / spacing_m))))
    near_outer = ndimage.binary_dilation(outer, iterations=max(1, int(np.ceil(0.10 / spacing_m))))
    for label, mask in instance_masks:
        current = np.asarray(mask, dtype=bool)
        if current.shape != indoor.shape:
            raise ValueError("instance mask must share the indoor map shape")
        suffix = "door" if label == LABEL_DOOR else "window"
        if not np.any(current & near_indoor) or not np.any(current & near_free):
            target = f"dropped_{suffix}"
        elif np.any(current & near_outer):
            target = f"exterior_{suffix}"
        else:
            target = f"training_{suffix}"
        groups[target] |= current
    return groups


def raster_from_metadata(metadata_path: Path) -> WorldRaster:
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    bounds = metadata["world_bounds_m"]
    return WorldRaster(
        x_min_m=float(bounds["x_min"]), x_max_m=float(bounds["x_max"]),
        z_min_m=float(bounds["z_min"]), z_max_m=float(bounds["z_max"]),
        spacing_m=float(metadata["map_spacing_m"]),
    )


def _flatten_mesh(path: Path):
    import trimesh

    loaded = trimesh.load(path, force="scene", process=False)
    return loaded.copy() if isinstance(loaded, trimesh.Trimesh) else loaded.to_geometry().copy()


def instance_transform_components(instance: dict[str, object]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Habitat instance transform values, including its implicit identity rotation."""

    return (
        np.asarray(instance.get("non_uniform_scale", [1.0, 1.0, 1.0]), dtype=np.float64),
        np.asarray(instance.get("rotation", [1.0, 0.0, 0.0, 0.0]), dtype=np.float64),
        np.asarray(instance["translation"], dtype=np.float64),
    )


def _transform_triangles(mesh, instance: dict[str, object]) -> np.ndarray:
    import trimesh

    scale, rotation, translation = instance_transform_components(instance)
    matrix = trimesh.transformations.quaternion_matrix(rotation)
    matrix[:3, :3] = matrix[:3, :3] @ np.diag(scale)
    matrix[:3, 3] = translation
    transformed = mesh.copy()
    transformed.apply_transform(matrix)
    return np.asarray(transformed.triangles, dtype=np.float64)


def opening_masks_for_scene(
    *, scene_instance_path: Path, openings_root: Path, categories: dict[str, np.uint8],
    raster: WorldRaster, cut_height_m: float, wall_thickness_m: float,
) -> tuple[np.ndarray, np.ndarray, list[tuple[np.uint8, np.ndarray]], dict[str, int]]:
    """Project HSSD opening assets at camera height into separate door/window masks."""

    payload = json.loads(Path(scene_instance_path).read_text(encoding="utf-8"))
    masks = {
        LABEL_DOOR: np.zeros((raster.height, raster.width), dtype=bool),
        LABEL_WINDOW: np.zeros((raster.height, raster.width), dtype=bool),
    }
    counts = {"door_instances": 0, "window_instances": 0, "unclassified_opening_instances": 0}
    instance_masks: list[tuple[np.uint8, np.ndarray]] = []
    cache: dict[str, object] = {}
    for instance in payload.get("object_instances", []):
        template_id = opening_template_id(str(instance.get("template_name", "")))
        config_path = Path(openings_root) / f"{template_id}.object_config.json"
        if not config_path.is_file():
            continue
        label = categories.get(template_id)
        if label is None:
            counts["unclassified_opening_instances"] += 1
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        asset_path = config_path.parent / str(config["render_asset"])
        if template_id not in cache:
            cache[template_id] = _flatten_mesh(asset_path)
        triangles = _transform_triangles(cache[template_id], instance)
        instance_mask = wall_mask_from_vertical_triangles(
            triangles, raster, cut_height_m=cut_height_m, wall_thickness_m=wall_thickness_m,
        )
        masks[label] |= instance_mask
        instance_masks.append((label, instance_mask))
        counts["door_instances" if label == LABEL_DOOR else "window_instances"] += 1
    return masks[LABEL_DOOR], masks[LABEL_WINDOW], instance_masks, counts


def export_scene(
    *, scene_id: str, floorplans_root: Path, hssd_root: Path, categories: dict[str, np.uint8],
    output_root: Path, cut_height_m: float, wall_thickness_m: float,
) -> dict[str, object]:
    """Export one ready scene without mutating the source floorplan directory."""

    source = Path(floorplans_root) / scene_id
    geometry = np.asarray(Image.open(source / "map.png").convert("L"), dtype=np.uint8)
    structural_wall = np.asarray(Image.open(source / "wall_mask.png").convert("L"), dtype=np.uint8)
    metadata_path = source / "floorplan_metadata.json"
    raster = raster_from_metadata(metadata_path)
    if geometry.shape != (raster.height, raster.width):
        raise ValueError(f"map shape does not match metadata raster: {scene_id}")
    free_wall_overlap_pixels = structural_free_wall_overlap_pixels(geometry, structural_wall)
    if free_wall_overlap_pixels:
        raise ValueError(f"geometry map has {free_wall_overlap_pixels} free pixels inside stage-wall support: {scene_id}")
    raw_door, raw_window, instance_masks, counts = opening_masks_for_scene(
        scene_instance_path=Path(hssd_root) / "scenes-uncluttered" / f"{scene_id}.scene_instance.json",
        openings_root=Path(hssd_root) / "objects" / "openings", categories=categories, raster=raster,
        cut_height_m=cut_height_m, wall_thickness_m=wall_thickness_m,
    )
    indoor = np.asarray(Image.open(source / "indoor_mask.png").convert("L"), dtype=np.uint8) > 0
    outer_boundary = np.asarray(Image.open(source / "outer_boundary_mask.png").convert("L"), dtype=np.uint8) > 0
    groups = split_opening_masks_for_training(
        instance_masks, indoor_mask=indoor, free_mask=geometry == 255,
        outer_boundary_mask=outer_boundary, spacing_m=raster.spacing_m,
    )
    door, window = groups["training_door"], groups["training_window"]
    labels = compose_semantic_labels(
        geometry, door_mask=door, window_mask=window,
        exterior_door_mask=groups["exterior_door"], exterior_window_mask=groups["exterior_window"],
    )
    raw_labels = compose_semantic_labels(geometry, door_mask=raw_door, window_mask=raw_window)
    scope = np.zeros(geometry.shape, dtype=np.uint8)
    scope[door | window] = OPENING_SCOPE_INTERNAL
    scope[groups["exterior_door"] | groups["exterior_window"]] = OPENING_SCOPE_EXTERIOR
    geometry_output = Path(output_root) / "geometry_maps" / scene_id
    semantic_output = Path(output_root) / "semantic_maps" / scene_id
    geometry_output.mkdir(parents=True, exist_ok=True)
    semantic_output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / "map.png", geometry_output / "map.png")
    shutil.copy2(metadata_path, geometry_output / "floorplan_metadata.json")
    Image.fromarray(labels_to_rgb(labels)).save(semantic_output / "semantic_map.png")
    np.save(semantic_output / "semantic_labels.npy", labels)
    Image.fromarray(labels_to_rgb(raw_labels)).save(semantic_output / "semantic_map_raw_openings.png")
    np.save(semantic_output / "opening_scope.npy", scope)
    record: dict[str, object] = {
        "scene_id": scene_id,
        "schema_version": "hssd-unloc-semantic-v2",
        "label_legend": {
            "0": "free", "1": "wall_or_exterior", "2": "interior_door", "3": "interior_window_or_glass",
            "4": "exterior_door", "5": "exterior_window_or_glass",
        },
        "coordinate_convention": "+Y top-down; +X right; +Z down in stored raster rows",
        "cut_height_m": cut_height_m,
        "wall_thickness_m": wall_thickness_m,
        "structural_wall_free_overlap_pixels": free_wall_overlap_pixels,
        "door_pixels": int(door.sum()), "window_pixels": int(window.sum()),
        "raw_door_pixels": int(raw_door.sum()), "raw_window_pixels": int(raw_window.sum()),
        "exterior_door_pixels": int(groups["exterior_door"].sum()),
        "exterior_window_pixels": int(groups["exterior_window"].sum()),
        "dropped_door_pixels": int(groups["dropped_door"].sum()),
        "dropped_window_pixels": int(groups["dropped_window"].sum()),
        "internal_opening_pixels": int((scope == OPENING_SCOPE_INTERNAL).sum()),
        "exterior_opening_pixels": int((scope == OPENING_SCOPE_EXTERIOR).sum()),
        **counts,
    }
    (semantic_output / "semantic_metadata.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record


def _ready_scene_ids(curation_manifest: Path) -> list[str]:
    manifest = json.loads(Path(curation_manifest).read_text(encoding="utf-8"))
    return [str(record["scene_id"]) for record in manifest["scenes"]
            if record["status"] in {"auto_accepted", "manual_accepted"}]


def _export_job(job: tuple[str, Path, Path, dict[str, np.uint8], Path, float, float]) -> dict[str, object]:
    return export_scene(
        scene_id=job[0], floorplans_root=job[1], hssd_root=job[2], categories=job[3],
        output_root=job[4], cut_height_m=job[5], wall_thickness_m=job[6],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hssd-root", type=Path, required=True)
    parser.add_argument("--floorplans-root", type=Path, required=True)
    parser.add_argument("--curation-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scene", action="append", help="Export only these ready scene IDs.")
    parser.add_argument("--cut-height-m", type=float, default=1.5)
    parser.add_argument("--wall-thickness-m", type=float, default=0.10)
    parser.add_argument("--workers", type=int, default=1, help="Independent scene-export processes.")
    args = parser.parse_args()
    ready = _ready_scene_ids(args.curation_manifest)
    requested = args.scene if args.scene else ready
    unknown = set(requested) - set(ready)
    if unknown:
        raise ValueError(f"requested scenes are not curation-approved: {sorted(unknown)}")
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    categories = load_opening_categories(args.hssd_root / "metadata" / "fpmodels-with-decomposed.csv")
    jobs = [(
        scene_id, args.floorplans_root, args.hssd_root, categories, args.output_root,
        args.cut_height_m, args.wall_thickness_m,
    ) for scene_id in requested]
    if args.workers == 1 or len(jobs) == 1:
        records = [_export_job(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as executor:
            records = list(executor.map(_export_job, jobs))
    payload = {"schema_version": "hssd-unloc-semantic-export-v1", "scene_count": len(records), "records": records}
    manifest_output = args.output_root / "manifests" / "semantic_export_manifest.json"
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"scene_count": len(records), "output": str(args.output_root)}, sort_keys=True))


if __name__ == "__main__":
    main()

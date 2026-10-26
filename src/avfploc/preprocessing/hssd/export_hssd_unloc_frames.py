#!/usr/bin/env python3
"""Export the approved HSSD RGB renders as independent UnLoc training frames.

The exporter deliberately writes only small target/index artifacts.  RGB files
stay in the reviewed renderer staging root, avoiding a second 70k-image copy
while the SSD/file-server I/O path is being stabilised.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml


UNLOC_IMAGE_WIDTH_PX = 640
UNLOC_FEATURE_COLUMNS = 40
UNLOC_FOCAL_LENGTH_OVER_WIDTH = 3.0 / 8.0
UNLOC_YAW_BINS = 36


def unloc_feature_column_bearings_deg(
    *, image_width_px: int = UNLOC_IMAGE_WIDTH_PX, feature_columns: int = UNLOC_FEATURE_COLUMNS,
    focal_length_over_width: float = UNLOC_FOCAL_LENGTH_OVER_WIDTH,
) -> np.ndarray:
    """Bearings of UnLoc's 40 horizontal feature columns, RGB left to right."""

    if image_width_px <= 0 or feature_columns <= 0 or focal_length_over_width <= 0:
        raise ValueError("image width, feature count, and focal-length ratio must be positive")
    # Each attention column pools one contiguous 16-pixel source interval.
    pixel_x = (np.arange(feature_columns, dtype=np.float64) + 0.5) * image_width_px / feature_columns - 0.5
    principal_x = (image_width_px - 1.0) / 2.0
    focal_length_px = focal_length_over_width * image_width_px
    return np.rad2deg(np.arctan((pixel_x - principal_x) / focal_length_px))


def _nearest_yaw_bins(bearing_deg: np.ndarray, *, yaw_bins: int = UNLOC_YAW_BINS) -> np.ndarray:
    if yaw_bins <= 0:
        raise ValueError("yaw_bins must be positive")
    return np.rint(np.asarray(bearing_deg, dtype=np.float64) * yaw_bins / 360.0).astype(np.int64) % yaw_bins


def depth40_from_structural_desdf(
    record: dict[str, object], *, desdf: np.ndarray, map_spacing_m: float, desdf_spacing_m: float,
) -> np.ndarray:
    """Return a 40-column camera-depth target from the matching structural map.

    ``desdf`` stores Euclidean ray range.  The UnLoc head predicts pinhole
    camera-forward depth, so each off-axis range is multiplied by cos(bearing).
    The world/map yaw convention is kept physical: RGB screen-left is negative
    logical bearing (toward -Z when yaw is +X) and screen-right is positive.
    """

    field = np.asarray(desdf, dtype=np.float32)
    if field.ndim != 3 or field.shape[2] != UNLOC_YAW_BINS:
        raise ValueError("HSSD structural DeSDF must have shape [row, col, 36]")
    if map_spacing_m <= 0 or desdf_spacing_m <= 0:
        raise ValueError("map and DeSDF spacing must be positive")
    ratio = int(round(desdf_spacing_m / map_spacing_m))
    if ratio < 1 or not math.isclose(ratio * map_spacing_m, desdf_spacing_m, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("DeSDF spacing must be an integer multiple of map spacing")
    try:
        source_row, source_col = int(record["grid_row"]), int(record["grid_col"])
        logical_yaw_deg = float(record["logical_yaw_deg"])
        width_px = int(record["camera_width_px"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("record is missing grid/yaw/camera metadata") from error
    if width_px != UNLOC_IMAGE_WIDTH_PX:
        raise ValueError(f"UnLoc adapter expects {UNLOC_IMAGE_WIDTH_PX}px RGB, got {width_px}")
    row, col = source_row // ratio, source_col // ratio
    if not (0 <= row < field.shape[0] and 0 <= col < field.shape[1]):
        raise ValueError("capture pose falls outside the structural DeSDF")
    bearings_deg = unloc_feature_column_bearings_deg(image_width_px=width_px)
    yaw_indices = _nearest_yaw_bins(logical_yaw_deg + bearings_deg)
    ray_ranges_m = field[row, col, yaw_indices]
    depth_m = ray_ranges_m * np.cos(np.deg2rad(bearings_deg))
    if not np.isfinite(depth_m).all() or np.any(depth_m <= 0):
        raise ValueError("structural target contains non-positive or non-finite depth")
    return depth_m.astype(np.float32)


def _load_desdf(path: Path) -> tuple[np.ndarray, float, float]:
    payload = np.load(path, allow_pickle=True)
    if isinstance(payload, np.ndarray) and payload.dtype == object and payload.shape == ():
        payload = payload.item()
    if not isinstance(payload, dict):
        raise ValueError(f"unsupported DeSDF payload: {path}")
    return (
        np.asarray(payload["desdf"], dtype=np.float32),
        float(payload["map_spacing_m"]),
        float(payload["desdf_spacing_m"]),
    )


def export_scene_frames(scene_id: str, rgb_scene_root: Path, floorplan_scene_root: Path, output_root: Path) -> dict[str, object]:
    """Write one scene's `depth40.npy` plus a one-row-per-accepted-RGB index."""

    rgb_scene_root, floorplan_scene_root, output_root = map(Path, (rgb_scene_root, floorplan_scene_root, output_root))
    manifest_path = rgb_scene_root / "pilot_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != 1600:
        raise ValueError(f"{scene_id}: expected all 1,600 scheduled RGB decisions")
    desdf, map_spacing_m, desdf_spacing_m = _load_desdf(floorplan_scene_root / "desdf.npy")
    accepted = sorted((row for row in records if isinstance(row, dict) and row.get("accepted")), key=lambda row: int(row["sample_index"]))
    if not accepted:
        raise ValueError(f"{scene_id}: no accepted RGB observations")
    samples: list[dict[str, object]] = []
    targets: list[np.ndarray] = []
    seen_indices: set[int] = set()
    for target_index, record in enumerate(accepted):
        sample_index = int(record["sample_index"])
        if sample_index in seen_indices:
            raise ValueError(f"{scene_id}: duplicate accepted sample index {sample_index}")
        seen_indices.add(sample_index)
        image = rgb_scene_root / str(record["image"])
        if not image.is_file():
            raise FileNotFoundError(f"{scene_id}: accepted RGB is missing: {image}")
        target = depth40_from_structural_desdf(
            record, desdf=desdf, map_spacing_m=map_spacing_m, desdf_spacing_m=desdf_spacing_m,
        )
        pose = record.get("pose_xyz_m")
        if not isinstance(pose, list) or len(pose) != 3:
            raise ValueError(f"{scene_id}: capture record lacks pose_xyz_m")
        targets.append(target)
        samples.append({
            "scene_id": scene_id,
            "sample_index": sample_index,
            "target_index": target_index,
            "image_path": str(image.resolve()),
            "logical_yaw_deg": float(record["logical_yaw_deg"]),
            "pose_xzyaw": [float(pose[0]), float(pose[2]), float(record["logical_yaw_deg"])],
            "camera_contract": {
                "width_px": int(record["camera_width_px"]),
                "hfov_deg": float(record["camera_hfov_deg"]),
                "feature_columns": UNLOC_FEATURE_COLUMNS,
                "focal_length_over_width": UNLOC_FOCAL_LENGTH_OVER_WIDTH,
            },
        })
    scene_out = output_root / "frames" / scene_id
    scene_out.mkdir(parents=True, exist_ok=True)
    np.save(scene_out / "depth40.npy", np.stack(targets).astype(np.float32))
    (scene_out / "sample_metadata.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in samples), encoding="utf-8",
    )
    return {
        "scene_id": scene_id,
        "accepted_frames": len(samples),
        "depth40_path": str((scene_out / "depth40.npy").resolve()),
        "sample_metadata_path": str((scene_out / "sample_metadata.jsonl").resolve()),
    }


def approved_split(split: dict[str, list[str]], *, rgb_root: Path, floorplans_root: Path) -> dict[str, list[str]]:
    """Remove curation-excluded scenes without silently changing split membership."""

    selected: dict[str, list[str]] = {}
    for split_name in ("train", "val", "test"):
        rows = split.get(split_name)
        if not isinstance(rows, list):
            raise ValueError(f"split lacks list {split_name!r}")
        selected[split_name] = [
            str(scene_id) for scene_id in rows
            if (Path(rgb_root) / str(scene_id) / "pilot_manifest.json").is_file()
            and (Path(floorplans_root) / str(scene_id) / "desdf.npy").is_file()
        ]
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-root", type=Path, required=True)
    parser.add_argument("--floorplans-root", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), action="append")
    parser.add_argument("--workers", type=int, default=8,
                        help="parallel scene exporters; RGB is indexed in place and never copied")
    args = parser.parse_args()
    split = yaml.safe_load(args.split_file.read_text(encoding="utf-8"))
    selected = approved_split(split, rgb_root=args.rgb_root, floorplans_root=args.floorplans_root)
    requested = args.split or ["train", "val", "test"]
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    output_root = args.output_root
    (output_root / "manifests").mkdir(parents=True, exist_ok=True)
    for split_name in requested:
        scenes = selected[split_name]
        def export(scene: str) -> dict[str, object]:
            return export_scene_frames(scene, args.rgb_root / scene, args.floorplans_root / scene, output_root)
        with ThreadPoolExecutor(max_workers=min(args.workers, len(scenes))) as executor:
            summaries = list(executor.map(export, scenes))
        (output_root / "manifests" / f"{split_name}_scenes.json").write_text(
            json.dumps({"split": split_name, "scenes": summaries}, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
        print(json.dumps({"split": split_name, "scenes": len(summaries), "frames": sum(row["accepted_frames"] for row in summaries)}))


if __name__ == "__main__":
    main()

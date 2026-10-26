#!/usr/bin/env python3
"""Publish the original UnLoc DeSDF occluder masks beside canonical HSSD data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from .audit_hssd_unloc_geometry import _atomic_text, _split_scenes
from .hssd_furnished_2dfp_rays import verify_scene_target_parity
from .replenish_hssd_unloc_furnished_views import _source_rows
from .render_hssd_unloc_rgb_pilot import _raster_from_metadata


def publish(source_floorplans: Path, dataset_root: Path) -> dict[str, object]:
    source_floorplans, dataset_root = Path(source_floorplans), Path(dataset_root)
    splits = _split_scenes(dataset_root)
    scene_owner = {scene: split for split, scenes in splits.items() for scene in scenes}
    if len(scene_owner) != sum(len(scenes) for scenes in splits.values()):
        raise ValueError("v2 scene splits overlap")
    output_root = dataset_root / "UnLoc" / "structural_targets"
    records: list[dict[str, object]] = []
    for scene_id in sorted(scene_owner):
        source = source_floorplans / scene_id / "desdf_occluder_mask.png"
        geometry_path = dataset_root / "2DFP" / scene_id / "map.png"
        if not source.is_file():
            raise FileNotFoundError(f"{scene_id}: original DeSDF occluder mask missing")
        geometry = np.asarray(Image.open(geometry_path).convert("L"), dtype=np.uint8)
        mask = np.asarray(Image.open(source).convert("L"), dtype=np.uint8) > 0
        if mask.shape != geometry.shape:
            raise ValueError(f"{scene_id}: source mask/geometry dimensions differ")
        capture, targets = _source_rows(dataset_root, scene_id)
        accepted = sorted((row for row in capture if row.get("accepted")), key=lambda row: int(row["sample_index"]))
        raster = _raster_from_metadata(dataset_root / "2DFP" / scene_id / "floorplan_metadata.json")
        parity = verify_scene_target_parity(accepted, targets, geometry, raster, occluder_mask=mask)
        if not parity["passed"]:
            raise ValueError(f"{scene_id}: original occluder mask does not reproduce v2 targets: {parity}")
        destination = output_root / scene_id / "desdf_occluder_mask.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        shutil.copyfile(source, temporary)
        temporary.replace(destination)
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        records.append({"scene_id": scene_id, "split": scene_owner[scene_id],
                        "mask_sha256": digest, "v2_target_parity": parity})
    summary = {"scenes": len(records), "split_counts": {key: len(value) for key, value in splits.items()},
               "all_v2_targets_reproduced": True, "records": records}
    _atomic_text(output_root / "manifest.json", json.dumps(summary, sort_keys=True, indent=2) + "\n")
    return {"scenes": len(records), "split_counts": summary["split_counts"],
            "all_v2_targets_reproduced": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-floorplans", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(publish(args.source_floorplans, args.dataset_root), sort_keys=True))


if __name__ == "__main__":
    main()

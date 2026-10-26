"""Build the immutable single-floor HSSD manifest for visual UnLoc training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import yaml
from .splits import load_split

from .hssd_unloc_dataset import (
    build_capture_schedule,
    build_split,
    inspect_scene_heights,
    indoor_region_annotations,
    scan_semantic_scenes,
    sample_poses,
    world_raster_for_regions,
    write_floorplan_artifacts,
)


DEFAULT_TEST_SCENE = "104862687_172226883"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hssd-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--split-manifest", type=Path, help="Existing JSON/YAML Train/Val/Test lists; no resplitting.")
    mode.add_argument("--legacy-create-split", action="store_true", help="Explicitly reproduce the initial 160-scene split, not the curated release.")
    parser.add_argument("--test-scene", default=DEFAULT_TEST_SCENE)
    parser.add_argument("--seed", type=int, default=31415)
    parser.add_argument("--scene", action="append", default=[], help="Eligible scene ID to build as a floorplan pilot; repeatable.")
    parser.add_argument("--map-spacing-m", type=float, default=0.01)
    parser.add_argument("--padding-m", type=float, default=0.30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    semantic_dir = args.hssd_root / "semantics" / "scenes"
    accepted, rejected = scan_semantic_scenes(semantic_dir)
    split = (load_split(args.split_manifest) if args.split_manifest else
             build_split(accepted, test_scene=args.test_scene, seed=args.seed))
    selected = set(sum(split.values(), []))
    if not selected.issubset(accepted):
        raise ValueError(f"split scenes missing/ineligible: {sorted(selected - set(accepted))}")
    if not set(args.scene).issubset(selected):
        raise ValueError("requested scene is outside the supplied split")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError("use a new empty output directory; existing datasets are not overwritten")
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "eligibility_report.json").write_text(
        json.dumps({
            "hssd_root": str(args.hssd_root),
            "accepted_scene_ids": accepted,
            "accepted_count": len(accepted),
            "rejected_scenes": rejected,
            "rejected_count": len(rejected),
            "split_manifest": str(args.split_manifest) if args.split_manifest else None,
            "legacy_split_creation": args.legacy_create_split,
            "fixed_test_scene": args.test_scene if args.legacy_create_split else None,
            "split_seed": args.seed if args.legacy_create_split else None,
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_root / "split.yaml").write_text(
        yaml.safe_dump(split, sort_keys=False), encoding="utf-8",
    )
    built: list[str] = []
    for scene_id in args.scene:
        if scene_id not in accepted:
            raise ValueError(f"floorplan pilot scene is not eligible: {scene_id}")
        semantic_path = semantic_dir / f"{scene_id}.semantic_config.json"
        payload = json.loads(semantic_path.read_text(encoding="utf-8"))
        regions = indoor_region_annotations(payload.get("region_annotations", []))
        if not inspect_scene_heights(semantic_path).eligible:
            raise AssertionError(f"scene unexpectedly failed eligibility after manifest scan: {scene_id}")
        raster = world_raster_for_regions(regions, spacing_m=args.map_spacing_m, padding_m=args.padding_m)
        write_floorplan_artifacts(
            args.output_root / "floorplans" / scene_id, regions, raster,
            scene_id=scene_id, semantic_path=semantic_path,
        )
        scene_output = args.output_root / "floorplans" / scene_id
        free_mask = np.asarray(Image.open(scene_output / "map.png"), dtype=np.uint8) == 255
        floor_y_m = float(regions[0].get("floor_height", 0.0))
        poses_rc = sample_poses(free_mask, raster, count=200, min_clearance_m=0.10, seed=args.seed)
        schedule = build_capture_schedule(poses_rc, raster, floor_y_m=floor_y_m)
        np.save(scene_output / "pose_grid_rc.npy", poses_rc)
        with (scene_output / "capture_schedule.jsonl").open("w", encoding="utf-8") as stream:
            for record in schedule:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
        built.append(scene_id)
    print(f"wrote {len(accepted)} eligible / {len(rejected)} rejected scene records and {len(built)} floorplans to {args.output_root}")


if __name__ == "__main__":
    main()

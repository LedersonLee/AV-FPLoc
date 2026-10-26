#!/usr/bin/env python3
"""Export a pass-only HSSD UnLoc adapter without copying RGB or depth arrays."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _resolve_relative(hssd_root: Path, value: str) -> Path:
    path = Path(value)
    result = path if path.is_absolute() else hssd_root / path
    result = result.resolve()
    if not result.is_relative_to(hssd_root.resolve()):
        raise ValueError(f"adapter reference escapes HSSD root: {value}")
    return result


def export_clean_adapter(
    hssd_root: Path, *, splits: tuple[str, ...] = ("train", "val", "test"),
    audit_root: Path | None = None, output_root: Path | None = None,
) -> dict[str, object]:
    hssd_root = Path(hssd_root).resolve()
    base = hssd_root / "UnLoc"
    audit_root = Path(audit_root) if audit_root else base / "geometry_audit_v1"
    output_root = Path(output_root) if output_root else base / "frame_adapter_v3_geometry_clean"
    v2 = base / "frame_adapter_v2"
    # Preflight every scene before publishing any v3 manifest.
    prepared: dict[str, list[tuple[dict, list[dict]]]] = {}
    all_keys: set[tuple[str, int]] = set()
    for split in splits:
        if split not in ("train", "val", "test"):
            raise ValueError(f"unsupported split {split}")
        manifest = json.loads((v2 / "manifests" / f"{split}_scenes.json").read_text())
        scene_rows = []
        for scene in manifest["scenes"]:
            scene_id = str(scene["scene_id"])
            metadata_file = _resolve_relative(hssd_root, str(scene["sample_metadata_path"]))
            depth_file = _resolve_relative(hssd_root, str(scene["depth40_path"]))
            source_rows = [json.loads(line) for line in metadata_file.open(encoding="utf-8")]
            if len(source_rows) != int(scene["accepted_frames"]):
                raise ValueError(f"{scene_id}: v2 manifest count mismatch")
            depth = np.load(depth_file, mmap_mode="r")
            if depth.shape != (len(source_rows), 40):
                raise ValueError(f"{scene_id}: depth40 shape mismatch")
            audit_file = audit_root / "scenes" / f"{scene_id}.jsonl"
            audit_rows = [json.loads(line) for line in audit_file.open(encoding="utf-8")]
            source_by_index = {int(row["sample_index"]): row for row in source_rows}
            audit_by_index = {int(row["sample_index"]): row for row in audit_rows}
            if len(source_by_index) != len(source_rows) or set(source_by_index) != set(audit_by_index) or len(audit_rows) != len(source_rows):
                raise ValueError(f"{scene_id}: audit/source keys differ or duplicate")
            keep = []
            for index, row in source_by_index.items():
                key = (scene_id, index)
                if key in all_keys:
                    raise ValueError(f"duplicate cross-split frame {key}")
                all_keys.add(key)
                verdict = audit_by_index[index]
                if verdict["split"] != split or verdict["scene_id"] != scene_id:
                    raise ValueError(f"{scene_id}/{index}: audit split/scene mismatch")
                if verdict["decision"] != "pass":
                    continue
                target_index = int(row["target_index"])
                if target_index != int(verdict["target_index"]) or not 0 <= target_index < len(depth):
                    raise ValueError(f"{scene_id}/{index}: target index mismatch")
                if not _resolve_relative(hssd_root, str(row["image_path"])).is_file():
                    raise FileNotFoundError(f"{scene_id}/{index}: missing RGB")
                keep.append(row)
            if not keep:
                raise ValueError(f"{scene_id}: zero passing views; clean adapter blocked")
            scene_rows.append((scene, keep))
        prepared[split] = scene_rows
    summary: dict[str, object] = {"adapter_version": "v3_geometry_clean", "by_split": {}, "source_views": len(all_keys)}
    for split, scene_rows in prepared.items():
        output_scenes = []
        retained = 0
        for scene, keep in scene_rows:
            scene_id = str(scene["scene_id"])
            destination = output_root / "frames" / scene_id / "sample_metadata.jsonl"
            _atomic_text(destination, "".join(json.dumps(row, sort_keys=True) + "\n" for row in keep))
            output_scenes.append({"scene_id": scene_id, "accepted_frames": len(keep),
                                  "depth40_path": scene["depth40_path"],
                                  "sample_metadata_path": str(destination.relative_to(hssd_root))})
            retained += len(keep)
        _atomic_text(output_root / "manifests" / f"{split}_scenes.json",
                     json.dumps({"split": split, "scenes": output_scenes}, sort_keys=True, indent=2) + "\n")
        summary["by_split"][split] = {"scenes": len(output_scenes), "retained_views": retained}
    summary["retained_views"] = sum(row["retained_views"] for row in summary["by_split"].values())
    _atomic_text(output_root / "summary.json", json.dumps(summary, sort_keys=True, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hssd-root", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    print(json.dumps(export_clean_adapter(args.hssd_root, audit_root=args.audit_root,
                                          output_root=args.output_root), sort_keys=True))


if __name__ == "__main__":
    main()

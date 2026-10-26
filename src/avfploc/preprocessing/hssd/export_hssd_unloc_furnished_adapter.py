#!/usr/bin/env python3
"""Publish a split-preserving adapter of furnished-QA pass views only."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from .export_hssd_unloc_clean_adapter import _atomic_text, _resolve_relative
from .hssd_furnished_depth_policy import AUDIT_DIRECTORY, FurnishedAuditPolicy


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _atomic_array(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.save(stream, values)
    temporary.replace(path)


def _replacement_row(record: dict, source: dict, scene_id: str, root: Path) -> tuple[dict, np.ndarray]:
    if record.get("scene_id") != scene_id or record.get("decision") != "pass":
        raise ValueError(f"{scene_id}: nonpassing or cross-scene replacement")
    image_value = str(record.get("image_path", ""))
    expected_prefix = f"UnLoc/RGB/{scene_id}/rgb_nohit_replacements/"
    if not image_value.startswith(expected_prefix) or not _resolve_relative(root, image_value).is_file():
        raise ValueError(f"{scene_id}: replacement RGB path missing or cross-scene")
    target = np.asarray(record.get("target_depth40_m"), dtype=np.float32)
    if target.shape != (40,) or not np.isfinite(target).all() or (target <= 0).any():
        raise ValueError(f"{scene_id}: replacement target must have 40 positive finite rays")
    pose = record.get("pose_xyz_m")
    if not isinstance(pose, list) or len(pose) != 3:
        raise ValueError(f"{scene_id}: replacement pose missing")
    yaw = float(record["logical_yaw_deg"])
    row = {
        "camera_contract": source["camera_contract"], "image_path": image_value,
        "logical_yaw_deg": yaw, "pose_xzyaw": [float(pose[0]), float(pose[2]), yaw],
        "sample_index": int(record["sample_index"]), "scene_id": scene_id,
        "furnished_qa": {"policy": FurnishedAuditPolicy().version, "reliable_no_hit": 0,
                          "warning_reasons": record.get("warning_reasons", [])},
    }
    return row, target


def export_adapter(
    hssd_root: Path, *, audit_root: Path | None = None, replacements_root: Path | None = None,
    output_root: Path | None = None,
) -> dict[str, object]:
    """Validate all scenes first, then write immutable-source-independent v4 data."""

    root = Path(hssd_root).resolve()
    unloc = root / "UnLoc"
    v2 = unloc / "frame_adapter_v2"
    audit_root = Path(audit_root) if audit_root else unloc / AUDIT_DIRECTORY
    replacements_root = Path(replacements_root) if replacements_root else audit_root / "replacements"
    output_root = Path(output_root) if output_root else unloc / "frame_adapter_v4_furnished_qa"
    prepared: dict[str, list[tuple[str, list[dict], np.ndarray, int, str]]] = {}
    scene_owner: dict[str, str] = {}
    global_ids: set[tuple[str, int]] = set()
    shortfalls: list[dict] = []
    for split in ("train", "val", "test"):
        manifest = json.loads((v2 / "manifests" / f"{split}_scenes.json").read_text(encoding="utf-8"))
        if manifest.get("split") != split:
            raise ValueError(f"{split}: v2 manifest split mismatch")
        prepared[split] = []
        for scene in manifest["scenes"]:
            scene_id = str(scene["scene_id"])
            if scene_id in scene_owner:
                raise ValueError(f"{scene_id}: scene occurs in multiple splits")
            scene_owner[scene_id] = split
            source_rows = _read_jsonl(_resolve_relative(root, str(scene["sample_metadata_path"])))
            depth = np.load(_resolve_relative(root, str(scene["depth40_path"])), mmap_mode="r")
            target_count = int(scene["accepted_frames"])
            if depth.shape != (target_count, 40) or len(source_rows) != target_count:
                raise ValueError(f"{scene_id}: v2 row/target count mismatch")
            original_ids = [int(row["sample_index"]) for row in source_rows]
            if len(set(original_ids)) != target_count:
                raise ValueError(f"{scene_id}: duplicate source sample IDs")
            audit_path = audit_root / "scenes" / f"{scene_id}.jsonl"
            audit_marker_path = audit_path.with_suffix(".complete.json")
            if not audit_marker_path.is_file():
                raise ValueError(f"{scene_id}: historical furnished audit incomplete")
            audit_marker = json.loads(audit_marker_path.read_text(encoding="utf-8"))
            if audit_marker.get("policy", {}).get("version") != FurnishedAuditPolicy().version:
                raise ValueError(f"{scene_id}: historical audit policy is not final")
            audit_rows = _read_jsonl(audit_path)
            audit_by_id = {int(row["sample_index"]): row for row in audit_rows}
            if len(audit_rows) != target_count or set(audit_by_id) != set(original_ids):
                raise ValueError(f"{scene_id}: furnished audit keys do not match v2")
            replacement_path = replacements_root / scene_id / "replacement_manifest.json"
            replacement = json.loads(replacement_path.read_text(encoding="utf-8"))
            if replacement.get("scene_id") != scene_id or replacement.get("split") != split:
                raise ValueError(f"{scene_id}: replacement split/scene mismatch")
            if replacement.get("audit_ledger_sha256") != audit_marker.get("ledger_sha256"):
                raise ValueError(f"{scene_id}: replacement ledger is stale against the current audit")
            if not replacement.get("complete") or int(replacement.get("target_count", -1)) != target_count:
                raise ValueError(f"{scene_id}: replacement ledger incomplete or target count mismatch")
            rows: list[dict] = []
            targets: list[np.ndarray] = []
            for source in source_rows:
                index = int(source["sample_index"])
                verdict = audit_by_id[index]
                if verdict.get("scene_id") != scene_id or verdict.get("split") != split:
                    raise ValueError(f"{scene_id}/{index}: historical audit split mismatch")
                if verdict["decision"] != "pass":
                    continue
                target_index = int(source["target_index"])
                if target_index != int(verdict["target_index"]) or not 0 <= target_index < target_count:
                    raise ValueError(f"{scene_id}/{index}: target index mismatch")
                if not _resolve_relative(root, str(source["image_path"])).is_file():
                    raise FileNotFoundError(f"{scene_id}/{index}: historical RGB missing")
                rows.append({**source, "furnished_qa": {
                    "policy": FurnishedAuditPolicy().version,
                    "reliable_no_hit": int(verdict.get("counts", {}).get("reliable_no_hit", 0)),
                    "warning_reasons": verdict.get("warning_reasons", []),
                }})
                targets.append(np.asarray(depth[target_index], dtype=np.float32))
            passing_replacements = [row for row in replacement.get("records", []) if row.get("accepted")]
            if int(replacement.get("replacement_accepted_count", -1)) != len(passing_replacements):
                raise ValueError(f"{scene_id}: accepted replacement count mismatch")
            if passing_replacements and not replacement.get("target_parity", {}).get("passed"):
                raise ValueError(f"{scene_id}: replacement target parity failed")
            if passing_replacements and not source_rows:
                raise ValueError(f"{scene_id}: no source camera contract")
            for record in passing_replacements:
                row, target = _replacement_row(record, source_rows[0], scene_id, root)
                rows.append(row)
                targets.append(target)
            ids = [int(row["sample_index"]) for row in rows]
            if len(ids) != len(set(ids)) or any((scene_id, index) in global_ids for index in ids):
                raise ValueError(f"{scene_id}: duplicate combined sample IDs")
            global_ids.update((scene_id, index) for index in ids)
            if not rows:
                raise ValueError(f"{scene_id}: no QA-pass views; cannot retain scene in trainer")
            declared_shortfall = int(replacement.get("shortfall", -1))
            if target_count - len(rows) != declared_shortfall or declared_shortfall < 0:
                raise ValueError(f"{scene_id}: replacement shortfall does not reconcile")
            if declared_shortfall:
                shortfalls.append({"scene_id": scene_id, "split": split,
                                   "shortfall": declared_shortfall, "status": replacement.get("status")})
            for target_index, row in enumerate(rows):
                row["target_index"] = target_index
                if row.get("scene_id") != scene_id:
                    raise ValueError(f"{scene_id}: cross-scene metadata row")
            prepared[split].append((scene_id, rows, np.stack(targets).astype(np.float32),
                                    target_count, str(replacement.get("status", ""))))
    if len(scene_owner) != sum(len(rows) for rows in prepared.values()):
        raise AssertionError("v2 splits are not disjoint")
    summary: dict[str, object] = {
        "adapter_version": "v4_furnished_qa", "source_adapter": "v2", "all_splits_disjoint": True,
        "qa_policy": FurnishedAuditPolicy().to_dict(), "structural_target_max_range_m": 10.0,
        "by_split": {}, "shortfalls": shortfalls,
        "total_shortfall": sum(row["shortfall"] for row in shortfalls),
        "training_ready": not shortfalls,
    }
    for split, scenes in prepared.items():
        output_scenes = []
        for scene_id, rows, targets, original_count, status in scenes:
            frame = output_root / "frames" / scene_id
            metadata = frame / "sample_metadata.jsonl"
            array = frame / "depth40.npy"
            _atomic_array(array, targets)
            _atomic_text(metadata, "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
            output_scenes.append({
                "scene_id": scene_id, "accepted_frames": len(rows),
                "original_target_frames": original_count, "shortfall": original_count - len(rows),
                "replacement_status": status,
                "sample_metadata_path": str(metadata.relative_to(root)),
                "depth40_path": str(array.relative_to(root)),
            })
        _atomic_text(output_root / "manifests" / f"{split}_scenes.json",
                     json.dumps({"split": split, "scenes": output_scenes}, sort_keys=True, indent=2) + "\n")
        summary["by_split"][split] = {
            "scenes": len(scenes), "views": sum(scene["accepted_frames"] for scene in output_scenes),
            "original_views": sum(scene["original_target_frames"] for scene in output_scenes),
        }
    _atomic_text(output_root / "summary.json", json.dumps(summary, sort_keys=True, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hssd-root", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path)
    parser.add_argument("--replacements-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    print(json.dumps(export_adapter(args.hssd_root, audit_root=args.audit_root,
                                    replacements_root=args.replacements_root,
                                    output_root=args.output_root), sort_keys=True))


if __name__ == "__main__":
    main()

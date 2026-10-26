#!/usr/bin/env python3
"""Create a shareable, review-backed HSSD floorplan curation manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


VALID_STATUSES = {
    "auto_accepted",
    "manual_accepted",
    "needs_regeneration",
    "on_hold",
    "excluded",
}


def _share_paths(scene_id: str) -> dict[str, str]:
    return {
        "geometry_map_png": f"geometry_maps/{scene_id}/map.png",
        "semantic_map_png": f"semantic_maps/{scene_id}/semantic_map.png",
        "semantic_labels_npy": f"semantic_maps/{scene_id}/semantic_labels.npy",
        "metadata_json": f"manifests/scenes/{scene_id}.json",
    }


def build_manifest(
    *,
    scene_ids: list[str],
    automatically_accepted: set[str],
    failure_categories: dict[str, str],
    overrides: dict[str, dict[str, str]],
) -> dict[str, object]:
    """Combine automatic validation and explicit human review without ambiguity."""
    unknown = set(overrides) - set(scene_ids)
    if unknown:
        raise ValueError(f"curation overrides reference unknown scenes: {sorted(unknown)}")
    records: list[dict[str, object]] = []
    for scene_id in scene_ids:
        status = "auto_accepted" if scene_id in automatically_accepted else "on_hold"
        reason = "automatic structural validation passed" if scene_id in automatically_accepted else (
            f"automatic review: {failure_categories.get(scene_id, 'missing diagnostic')}"
        )
        reviewed = False
        if scene_id in overrides:
            override = overrides[scene_id]
            status = str(override["status"])
            reason = str(override["reason"])
            reviewed = True
        if status not in VALID_STATUSES:
            raise ValueError(f"unsupported curation status for {scene_id}: {status}")
        records.append({
            "scene_id": scene_id,
            "status": status,
            "reason": reason,
            "human_reviewed": reviewed,
            "automatic_failure_category": failure_categories.get(scene_id),
            "share_paths": _share_paths(scene_id),
        })
    summary = {status: sum(record["status"] == status for record in records) for status in sorted(VALID_STATUSES)}
    summary["ready_for_geometry_training"] = summary["auto_accepted"] + summary["manual_accepted"]
    summary["not_ready"] = len(records) - summary["ready_for_geometry_training"]
    return {
        "schema_version": "hssd-unloc-curation-v1",
        "geometry_root": "geometry_maps",
        "semantic_root": "semantic_maps",
        "scenes": records,
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--passed-list", type=Path, required=True)
    parser.add_argument("--failure-audit", type=Path, required=True)
    parser.add_argument("--manual-overrides", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scene_ids = [line.strip() for line in args.scene_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    passed = {line.strip() for line in args.passed_list.read_text(encoding="utf-8").splitlines() if line.strip()}
    audit = json.loads(args.failure_audit.read_text(encoding="utf-8"))
    failures = {str(record["scene_id"]): str(record["category"]) for record in audit.get("records", [])}
    overrides_document = json.loads(args.manual_overrides.read_text(encoding="utf-8"))
    overrides = {str(record["scene_id"]): record for record in overrides_document["overrides"]}
    manifest = build_manifest(
        scene_ids=scene_ids,
        automatically_accepted=passed,
        failure_categories=failures,
        overrides=overrides,
    )
    manifest["manual_override_source"] = str(args.manual_overrides)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest["summary"], sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Check every published HSSD row against raw no-hit evidence and fixed splits."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from .audit_hssd_unloc_geometry import _atomic_text, _hash_file
from .hssd_furnished_depth_policy import AUDIT_DIRECTORY, FurnishedAuditPolicy


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _no_hits(row: dict) -> int:
    depth = np.asarray([np.nan if v is None else v for v in row["furnished_depth40_m"]])
    ambiguous = np.asarray(row["ambiguous40"], dtype=bool)
    if depth.shape != (40,) or ambiguous.shape != (40,):
        raise ValueError("raw feature evidence must contain 40 depth/ambiguity values")
    return int(np.count_nonzero((~np.isfinite(depth) | (depth <= 0)) & ~ambiguous))


def validate(root: Path) -> dict:
    root = root.resolve()
    adapter = root / "UnLoc" / "frame_adapter_v4_furnished_qa"
    audit = root / "UnLoc" / AUDIT_DIRECTORY
    summary = json.loads((adapter / "summary.json").read_text())
    assert summary["training_ready"] and summary["total_shortfall"] == 0
    assert summary["qa_policy"]["version"] == FurnishedAuditPolicy().version
    owners: set[str] = set()
    report = {"passed": True, "policy": FurnishedAuditPolicy().version, "by_split": {},
              "included_reliable_no_hit_rays": 0, "target_mismatched_rays": 0}
    for split in ("train", "val", "test"):
        source = root / "UnLoc" / "frame_adapter_v2" / "manifests" / f"{split}_scenes.json"
        previous = json.loads(source.read_text())
        current = json.loads((adapter / "manifests" / f"{split}_scenes.json").read_text())
        assert previous["split"] == current["split"] == split
        before = {s["scene_id"]: s for s in previous["scenes"]}
        after = {s["scene_id"]: s for s in current["scenes"]}
        assert len(after) == len(current["scenes"]) and set(before) == set(after)
        assert not owners.intersection(after)
        owners.update(after)
        counts = Counter(scenes=len(after))
        for scene, entry in after.items():
            old = before[scene]
            original = _rows(root / old["sample_metadata_path"])
            old_targets = np.load(root / old["depth40_path"])
            ledger = audit / "scenes" / f"{scene}.jsonl"
            marker = json.loads(ledger.with_suffix(".complete.json").read_text())
            assert marker["scene_id"] == scene and marker["split"] == split
            assert marker["policy"]["version"] == FurnishedAuditPolicy().version
            assert marker["ledger_sha256"] == _hash_file(ledger)
            assert marker["input_hashes"]["adapter_targets"] == _hash_file(root / old["depth40_path"])
            evidence = {int(r["sample_index"]): r for r in _rows(ledger)}
            assert set(evidence) == {int(r["sample_index"]) for r in original}
            replacement = json.loads((audit / "replacements" / scene / "replacement_manifest.json").read_text())
            assert replacement["scene_id"] == scene and replacement["split"] == split
            assert replacement["complete"] and replacement["shortfall"] == 0
            assert replacement["audit_ledger_sha256"] == marker["ledger_sha256"]
            assert replacement["target_parity"]["passed"]
            added = {int(r["sample_index"]): r for r in replacement["records"] if r.get("accepted")}
            kept = {int(r["sample_index"]): r for r in original if evidence[int(r["sample_index"])]["decision"] == "pass"}
            assert not set(kept).intersection(added)
            rows = _rows(root / entry["sample_metadata_path"])
            targets = np.load(root / entry["depth40_path"])
            assert len(rows) == old["accepted_frames"] == entry["accepted_frames"]
            assert targets.shape == (len(rows), 40) and np.isfinite(targets).all() and (targets > 0).all()
            assert set(kept) | set(added) == {int(r["sample_index"]) for r in rows}
            assert len({int(r["sample_index"]) for r in rows}) == len(rows)
            original_capture = json.loads((root / "UnLoc" / "RGB" / scene / "pilot_manifest.json").read_text())
            old_positions = np.asarray([r["pose_xyz_m"] for r in original_capture["records"]])[:, [0, 2]]
            new_positions = np.asarray([r["pose_xyz_m"] for r in added.values()])
            if len(new_positions):
                distance = cKDTree(old_positions).query(new_positions[:, [0, 2]])[0]
                assert float(distance.min()) >= 0.25 - 1e-4
            for index, row in enumerate(rows):
                sample = int(row["sample_index"])
                assert row["scene_id"] == scene and row["target_index"] == index
                image = str(row["image_path"])
                assert image.startswith(f"UnLoc/RGB/{scene}/") and (root / image).is_file()
                raw = evidence[sample] if sample in kept else added[sample]
                assert raw["scene_id"] == scene and raw["split"] == split
                assert _no_hits(raw) == 0
                reference = old_targets[int(kept[sample]["target_index"])] if sample in kept else np.asarray(raw["target_depth40_m"], dtype=np.float32)
                assert np.array_equal(targets[index], reference)
                if sample in added:
                    with Image.open(root / image) as rgb:
                        assert rgb.size == (640, 480)
            removed = [r for r in evidence.values() if r["decision"] != "pass"]
            assert all(_no_hits(r) > 0 and r["reasons"] == ["furnished_no_hit"] for r in removed)
            assert len(removed) == len(added) == replacement["replacement_accepted_count"]
            counts.update(views=len(rows), retained_originals=len(kept), excluded_no_hit=len(removed), replacements=len(added))
        report["by_split"][split] = dict(counts)
    report["scenes"] = len(owners)
    report["views"] = sum(v["views"] for v in report["by_split"].values())
    _atomic_text(adapter / "verification.json", json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    print(json.dumps(validate(parser.parse_args().dataset_root), sort_keys=True))

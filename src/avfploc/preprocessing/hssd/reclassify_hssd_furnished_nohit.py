#!/usr/bin/env python3
"""Reclassify saved furnished ray evidence without rerendering or changing splits."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

from .audit_hssd_unloc_furnished_depth import _input_hashes, scene_is_complete
from .audit_hssd_unloc_geometry import _atomic_text, _hash_file, _split_scenes
from .hssd_furnished_depth_policy import AUDIT_DIRECTORY, FurnishedAuditPolicy, classify_furnished_view


def _float40(values) -> np.ndarray:
    return np.asarray([np.nan if value is None else value for value in values], dtype=np.float32)


def reclassify_scene(payload: tuple) -> dict:
    scene, split, root, source, output = payload
    ledger = source / "scenes" / f"{scene}.jsonl"
    marker = json.loads(ledger.with_suffix(".complete.json").read_text())
    if marker.get("scene_id") != scene or marker.get("split") != split:
        raise ValueError(f"{scene}: source ledger scene/split mismatch")
    if not scene_is_complete(ledger, expected_count=int(marker["expected_count"]),
                             input_hashes=_input_hashes(scene, root)):
        raise ValueError(f"{scene}: source evidence is incomplete or stale")
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    policy = FurnishedAuditPolicy()
    transitions = Counter()
    for row in rows:
        if row.get("scene_id") != scene or row.get("split") != split:
            raise ValueError(f"{scene}: row crossed the fixed scene split")
        old = row["decision"]
        if any(row.get(key) is None for key in (
            "furnished_depth40_m", "ambiguous40", "target_depth40_m", "exact_map_depth40_m", "scope40",
        )):
            raise ValueError(f"{scene}/{row['sample_index']}: raw depth evidence missing")
        row["previous_decision"] = old
        row["previous_reasons"] = row.get("reasons", [])
        row.update(classify_furnished_view(
            _float40(row["furnished_depth40_m"]), np.asarray(row["ambiguous40"], dtype=bool),
            _float40(row["target_depth40_m"]), _float40(row["exact_map_depth40_m"]),
            None if row.get("stage_depth40_m") is None else _float40(row["stage_depth40_m"]),
            np.asarray(row["scope40"], dtype=np.uint8), policy,
            stage_ambiguous40=None if row.get("stage_ambiguous40") is None else np.asarray(row["stage_ambiguous40"], dtype=bool),
        ))
        transitions[f"{old}_to_{row['decision']}"] += 1
    target = output / "scenes" / ledger.name
    _atomic_text(target, "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    new_marker = dict(marker)
    new_marker.update({
        "policy": policy.to_dict(), "ledger_sha256": _hash_file(target),
        "source_ledger": str(ledger.relative_to(root)), "source_ledger_sha256": _hash_file(ledger),
        "source_policy": marker["policy"], "audit_method": "reclassify_saved_furnished_evidence",
    })
    _atomic_text(target.with_suffix(".complete.json"), json.dumps(new_marker, sort_keys=True, indent=2) + "\n")
    return {"scene_id": scene, "split": split, "transitions": dict(transitions),
            **dict(Counter(row["decision"] for row in rows)),
            "retained_with_warnings": sum(bool(row.get("warning_reasons")) and row["decision"] == "pass" for row in rows),
            "reliable_no_hit_rays": sum(row["counts"].get("reliable_no_hit", 0) for row in rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    source = args.source_root or args.dataset_root / "UnLoc" / "furnished_depth_qa_v1"
    output = args.output_root or args.dataset_root / "UnLoc" / AUDIT_DIRECTORY
    if source.resolve() == output.resolve():
        parser.error("source evidence must be preserved in a separate directory")
    work = [(scene, split, args.dataset_root, source, output)
            for split, scenes in _split_scenes(args.dataset_root).items() for scene in scenes]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(reclassify_scene, work))
    summary = {"policy": FurnishedAuditPolicy().to_dict(), "scenes": len(results),
               "totals": {k: sum(r.get(k, 0) for r in results) for k in ("pass", "review", "fail")},
               "retained_with_warnings": sum(r["retained_with_warnings"] for r in results),
               "reliable_no_hit_rays": sum(r["reliable_no_hit_rays"] for r in results),
               "scene_results": results}
    _atomic_text(output / "summary.json", json.dumps(summary, sort_keys=True, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "scene_results"}, sort_keys=True))


if __name__ == "__main__":
    main()

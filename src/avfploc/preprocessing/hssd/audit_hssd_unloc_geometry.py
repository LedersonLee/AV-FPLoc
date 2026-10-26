#!/usr/bin/env python3
"""Audit all captured HSSD UnLoc views against stage-only 3D depth."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Callable

import numpy as np

from .hssd_geometry_audit_policy import AuditPolicy, classify_view
from .hssd_geometry_audit_rays import render_scene_depths


def _hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def load_scene_inputs(
    scene_id: str, hssd_root: Path,
) -> tuple[list[dict], dict[int, tuple[dict, np.ndarray]], dict[int, list[str]]]:
    """Join accepted capture records to v2 targets by sample index, fail closed."""

    base = Path(hssd_root) / "UnLoc"
    rgb_dir = base / "RGB" / scene_id
    frame_dir = base / "frame_adapter_v2" / "frames" / scene_id
    source = json.loads((rgb_dir / "pilot_manifest.json").read_text(encoding="utf-8"))["records"]
    accepted = [r for r in source if isinstance(r, dict) and r.get("accepted")]
    accepted.sort(key=lambda r: int(r["sample_index"]))
    capture_count = Counter(int(r["sample_index"]) for r in accepted)
    adapter_count: Counter[int] = Counter()
    adapter: dict[int, dict] = {}
    with (frame_dir / "sample_metadata.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            index = int(row["sample_index"])
            adapter_count[index] += 1
            adapter[index] = row
    targets = np.load(frame_dir / "depth40.npy", mmap_mode="r")
    joined: dict[int, tuple[dict, np.ndarray]] = {}
    issues: dict[int, list[str]] = defaultdict(list)
    for capture in accepted:
        index = int(capture["sample_index"])
        if capture_count[index] != 1:
            issues[index].append("duplicate_capture_row")
        if adapter_count[index] == 0:
            issues[index].append("missing_adapter_row")
            continue
        if adapter_count[index] != 1:
            issues[index].append("duplicate_adapter_row")
            continue
        row = adapter[index]
        if row.get("scene_id") != scene_id:
            issues[index].append("adapter_scene_mismatch")
        if float(row.get("logical_yaw_deg", float("nan"))) != float(capture.get("logical_yaw_deg", float("nan"))):
            issues[index].append("adapter_yaw_mismatch")
        pose = capture.get("pose_xyz_m")
        if not isinstance(pose, list) or len(pose) != 3 or not np.allclose(
            np.asarray(row.get("pose_xzyaw", [np.nan] * 3), dtype=float)[:2],
            [pose[0], pose[2]], atol=1e-4, rtol=0,
        ):
            issues[index].append("adapter_pose_mismatch")
        image = rgb_dir / str(capture.get("image", ""))
        if not image.is_file():
            issues[index].append("missing_rgb")
        target_index = int(row.get("target_index", -1))
        if target_index < 0 or target_index >= len(targets):
            issues[index].append("invalid_target_index")
        elif np.asarray(targets[target_index]).shape != (40,):
            issues[index].append("invalid_target_shape")
        elif not issues.get(index):
            joined[index] = (row, np.asarray(targets[target_index], dtype=np.float32).copy())
    return accepted, joined, dict(issues)


def audit_scene(
    scene_id: str, split: str, hssd_root: Path, policy: AuditPolicy,
    *, gpu_device_id: int = 0, render: Callable = render_scene_depths,
) -> list[dict]:
    accepted, joined, issues = load_scene_inputs(scene_id, hssd_root)
    render_inputs = [r for r in accepted if int(r["sample_index"]) in joined]
    source_root = Path(hssd_root) / "source" / "hssd-hab_local_20261012"
    try:
        rendered = render(scene_id, render_inputs, source_root, gpu_device_id=gpu_device_id)
        render_error = None
    except Exception as error:
        rendered = {}
        render_error = f"{type(error).__name__}: {error}"
    rows = []
    for capture in accepted:
        index = int(capture["sample_index"])
        row = joined.get(index)
        result: dict[str, object] = {
            "scene_id": scene_id, "split": split, "sample_index": index,
            "target_index": None if row is None else int(row[0]["target_index"]),
            "image_path": None if row is None else row[0]["image_path"],
            "pose_xyz_m": capture.get("pose_xyz_m"),
            "logical_yaw_deg": capture.get("logical_yaw_deg"),
            "habitat_yaw_deg": capture.get("habitat_yaw_deg"),
            "stage_depth40_m": None, "target_depth40_m": None,
            "ambiguous40": None,
        }
        if index in issues:
            verdict = {"decision": "fail", "reasons": sorted(issues[index]), "counts": {}, "ray_classes": []}
        elif index not in rendered:
            verdict = {"decision": "fail", "reasons": ["render_error" if render_error else "missing_rendered_sample"],
                       "counts": {}, "ray_classes": []}
            if render_error:
                result["processing_error"] = render_error
        else:
            stage, ambiguous = rendered[index]
            target = row[1]
            verdict = classify_view(stage, ambiguous, target, policy)
            result["stage_depth40_m"] = [None if not np.isfinite(v) else float(v) for v in stage]
            result["target_depth40_m"] = target.astype(float).tolist()
            result["ambiguous40"] = np.asarray(ambiguous, dtype=bool).tolist()
        result.update(verdict)
        rows.append(result)
    return rows


def _split_scenes(hssd_root: Path) -> dict[str, list[str]]:
    manifest_root = hssd_root / "UnLoc" / "frame_adapter_v2" / "manifests"
    splits: dict[str, list[str]] = {}
    for split in ("train", "val", "test"):
        payload = json.loads((manifest_root / f"{split}_scenes.json").read_text())
        splits[split] = [str(row["scene_id"]) for row in payload["scenes"]]
    all_scenes = sum(splits.values(), [])
    if len(all_scenes) != len(set(all_scenes)):
        raise ValueError("v2 scene splits overlap")
    return splits


def _scene_task(args: tuple) -> tuple[str, str, list[dict], dict[str, str]]:
    scene, split, hssd_root, policy, gpu_device_id = args
    rows = audit_scene(scene, split, hssd_root, policy, gpu_device_id=gpu_device_id)
    base = hssd_root / "UnLoc"
    rgb = base / "RGB" / scene / "pilot_manifest.json"
    frame = base / "frame_adapter_v2" / "frames" / scene
    stage = hssd_root / "source" / "hssd-hab_local_20261012" / "stages" / f"{scene}.glb"
    hashes = {name: _hash_file(path) for name, path in (
        ("capture_manifest_sha256", rgb),
        ("adapter_metadata_sha256", frame / "sample_metadata.jsonl"),
        ("target_depth_sha256", frame / "depth40.npy"),
        ("stage_glb_sha256", stage),
    )}
    return scene, split, rows, hashes


def _aggregate(output_root: Path, splits: dict[str, list[str]]) -> dict:
    decision = Counter()
    by_split: dict[str, Counter] = defaultdict(Counter)
    by_scene: dict[str, dict] = {}
    by_yaw: dict[str, Counter] = defaultdict(Counter)
    source_poses_by_split: dict[str, set] = defaultdict(set)
    passing_yaws_by_split: dict[str, dict[tuple, set]] = defaultdict(lambda: defaultdict(set))
    unique: set[tuple[str, int]] = set()
    for split, scenes in splits.items():
        for scene in scenes:
            file = output_root / "scenes" / f"{scene}.jsonl"
            if not file.is_file():
                continue
            local = Counter()
            local_source_poses: set[tuple] = set()
            local_passing_yaws: dict[tuple, set] = defaultdict(set)
            for line in file.open(encoding="utf-8"):
                row = json.loads(line)
                key = (scene, int(row["sample_index"]))
                if key in unique:
                    raise ValueError(f"duplicate audited key {key}")
                unique.add(key)
                status = row["decision"]
                local[status] += 1
                decision[status] += 1
                by_split[split][status] += 1
                by_yaw[str(row["logical_yaw_deg"])][status] += 1
                pose = tuple(row.get("pose_xyz_m") or ())
                pose_key = (scene, pose)
                local_source_poses.add(pose)
                source_poses_by_split[split].add(pose_key)
                if status == "pass":
                    local_passing_yaws[pose].add(row["logical_yaw_deg"])
                    passing_yaws_by_split[split][pose_key].add(row["logical_yaw_deg"])
            by_scene[scene] = {
                "split": split, **dict(local), "total": sum(local.values()),
                "source_poses": len(local_source_poses),
                "passing_poses": len(local_passing_yaws),
                "passing_yaws_per_pose": {str(k): v for k, v in sorted(Counter(map(len, local_passing_yaws.values())).items())},
            }
    split_summary = {}
    for split, counts in by_split.items():
        pose_yaws = passing_yaws_by_split[split]
        split_summary[split] = {
            **dict(counts), "source_poses": len(source_poses_by_split[split]),
            "passing_poses": len(pose_yaws),
            "passing_yaws_per_pose": {str(k): v for k, v in sorted(Counter(map(len, pose_yaws.values())).items())},
        }
    return {"total": sum(decision.values()), "completed_scenes": len(by_scene),
            "decisions": dict(decision), "by_split": split_summary,
            "by_scene": by_scene, "by_yaw": {k: dict(v) for k, v in by_yaw.items()}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hssd-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--gpu-device-id", type=int, default=0)
    parser.add_argument("--scenes", nargs="*")
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 4:
        raise ValueError("GPU worker count must be 1–4")
    hssd_root = args.hssd_root.resolve()
    output_root = args.output_root or hssd_root / "UnLoc" / "geometry_audit_v1"
    splits = _split_scenes(hssd_root)
    selected = set(args.scenes) if args.scenes else None
    if selected is not None and selected - set(sum(splits.values(), [])):
        raise ValueError("unknown scene requested")
    policy = AuditPolicy()
    policy_text = json.dumps(policy.to_dict(), sort_keys=True, indent=2) + "\n"
    policy_file = output_root / "policy.json"
    if policy_file.exists() and policy_file.read_text() != policy_text:
        raise ValueError("existing audit policy differs; use a new output version")
    _atomic_text(policy_file, policy_text)
    tasks = []
    for split, scenes in splits.items():
        for scene in scenes:
            if selected is not None and scene not in selected:
                continue
            if (output_root / "scenes" / f"{scene}.jsonl").is_file() and (output_root / "scenes" / f"{scene}.meta.json").is_file():
                continue
            tasks.append((scene, split, hssd_root, policy, args.gpu_device_id))
    if args.workers == 1:
        iterator = map(_scene_task, tasks)
    else:
        executor = ProcessPoolExecutor(max_workers=args.workers)
        iterator = executor.map(_scene_task, tasks)
    try:
        for completed, (scene, split, rows, hashes) in enumerate(iterator, 1):
            _atomic_text(output_root / "scenes" / f"{scene}.jsonl",
                         "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
            _atomic_text(output_root / "scenes" / f"{scene}.meta.json",
                         json.dumps({"scene_id": scene, "split": split, "rows": len(rows),
                                     "policy": policy.version, **hashes}, sort_keys=True, indent=2) + "\n")
            counts = dict(Counter(r["decision"] for r in rows))
            print(json.dumps({"done": completed, "remaining": len(tasks) - completed,
                              "scene": scene, "rows": len(rows), "decisions": counts}), flush=True)
    finally:
        if args.workers > 1:
            executor.shutdown(wait=True)
    summary = _aggregate(output_root, splits)
    _atomic_text(output_root / "summary.json", json.dumps(summary, sort_keys=True, indent=2) + "\n")
    csv_path = output_root / "summary.csv"
    lines = ["scene_id,split,total,pass,review,fail\n"]
    for scene, row in sorted(summary["by_scene"].items()):
        lines.append(f'{scene},{row["split"]},{row["total"]},{row.get("pass",0)},{row.get("review",0)},{row.get("fail",0)}\n')
    _atomic_text(csv_path, "".join(lines))
    print(json.dumps({"summary": summary["decisions"], "total": summary["total"],
                      "completed_scenes": summary["completed_scenes"]}), flush=True)


if __name__ == "__main__":
    main()

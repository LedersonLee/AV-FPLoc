"""A policy update restores natural views while preserving source evidence."""

import json
from pathlib import Path
import sys

from avfploc.preprocessing.hssd import reclassify_hssd_furnished_nohit as module


def test_reclassification_keeps_ambiguity_and_rejects_even_one_real_nohit(tmp_path, monkeypatch):
    source, output = tmp_path / "old", tmp_path / "new"
    ledger = source / "scenes" / "scene.jsonl"
    ledger.parent.mkdir(parents=True)
    rows = []
    for index in range(3):
        depth = [6.0] * 40
        ambiguous = [False] * 40
        if index in (1, 2):
            depth[0] = None
            ambiguous[0] = index == 1
        rows.append({"scene_id": "scene", "split": "test", "sample_index": index,
                     "decision": "review", "reasons": ["old_rule"],
                     "furnished_depth40_m": depth, "ambiguous40": ambiguous,
                     "target_depth40_m": [1.0] * 40, "exact_map_depth40_m": [1.0] * 40,
                     "stage_depth40_m": None, "scope40": [0] * 40})
    ledger.write_text("".join(json.dumps(r) + "\n" for r in rows))
    original = ledger.read_bytes()
    ledger.with_suffix(".complete.json").write_text(json.dumps({
        "scene_id": "scene", "split": "test", "expected_count": 3,
        "policy": {"version": "old"}, "sensor_mode": "narrow"}))
    monkeypatch.setattr(module, "_input_hashes", lambda *a: {})
    monkeypatch.setattr(module, "scene_is_complete", lambda *a, **k: True)
    result = module.reclassify_scene(("scene", "test", tmp_path, source, output))
    derived = [json.loads(line) for line in (output / "scenes" / "scene.jsonl").read_text().splitlines()]
    assert [r["decision"] for r in derived] == ["pass", "pass", "fail"]
    assert result["pass"] == 2 and result["fail"] == 1
    assert all(r["split"] == "test" for r in derived)
    assert ledger.read_bytes() == original

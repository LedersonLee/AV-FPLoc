"""Regression contract for explicit HSSD floorplan curation decisions."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "src" / "avfploc" / "preprocessing" / "hssd" / "build_hssd_unloc_curation_manifest.py"


def load_module():
    return __import__("avfploc.preprocessing.hssd.build_hssd_unloc_curation_manifest", fromlist=["*"])


class HSSDUnLocCurationManifestTest(unittest.TestCase):
    def test_manual_review_overrides_automatic_status_and_declares_separate_artifact_roots(self):
        curation = load_module()
        manifest = curation.build_manifest(
            scene_ids=["clean", "manual", "repair", "excluded"],
            automatically_accepted={"clean"},
            failure_categories={"manual": "anchor_snap_tolerance", "repair": "false_open"},
            overrides={
                "manual": {"status": "manual_accepted", "reason": "mesh QA confirms map"},
                "repair": {"status": "needs_regeneration", "reason": "fill wall void"},
                "excluded": {"status": "excluded", "reason": "room omitted"},
            },
        )

        records = {record["scene_id"]: record for record in manifest["scenes"]}
        self.assertEqual(records["clean"]["status"], "auto_accepted")
        self.assertEqual(records["manual"]["status"], "manual_accepted")
        self.assertEqual(records["repair"]["status"], "needs_regeneration")
        self.assertEqual(records["excluded"]["status"], "excluded")
        self.assertEqual(manifest["summary"]["ready_for_geometry_training"], 2)
        self.assertEqual(records["clean"]["share_paths"]["geometry_map_png"], "geometry_maps/clean/map.png")
        self.assertEqual(records["clean"]["share_paths"]["semantic_map_png"], "semantic_maps/clean/semantic_map.png")


if __name__ == "__main__":
    unittest.main()

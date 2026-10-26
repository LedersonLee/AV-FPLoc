"""Connectivity reconciliation classifications that do not need Habitat."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[2] / "src" / "avfploc" / "preprocessing" / "hssd" / "reconcile_hssd_unloc_connectivity.py"


def load_module():
    return __import__("avfploc.preprocessing.hssd.reconcile_hssd_unloc_connectivity", fromlist=["*"])


class ConnectivityClassificationTest(unittest.TestCase):
    def test_path_outside_indoor_is_external_only_when_all_outside_samples_hit_exterior(self):
        reconcile = load_module()
        raster = reconcile.WorldRaster(0.0, 4.0, 0.0, 4.0, 1.0)
        indoor = np.ones((raster.height, raster.width), dtype=bool)
        indoor[2, 2] = False
        exterior = np.zeros_like(indoor)
        exterior[2, 2] = True
        summary = reconcile.path_indoor_summary(
            np.asarray([[1.0, 2.0], [3.0, 2.0]]), indoor, raster, exterior_mask=exterior,
        )
        self.assertFalse(summary["stays_inside"])
        self.assertTrue(summary["external_only"])

    def test_external_route_does_not_join_two_indoor_building_components(self):
        reconcile = load_module()
        regions = ["00:hall", "01:bedroom", "02:annex"]
        reachable = {
            ("00:hall", "01:bedroom"): True,
            ("00:hall", "02:annex"): True,
            ("01:bedroom", "02:annex"): False,
        }

        components = reconcile.indoor_navmesh_components(
            regions,
            reachable,
            exterior_only_pairs={("00:hall", "02:annex")},
        )

        self.assertEqual(components, [["00:hall", "01:bedroom"], ["02:annex"]])

    def test_disconnected_buildings_remain_eligible_when_each_map_component_is_clean(self):
        reconcile = load_module()

        self.assertTrue(reconcile.bulk_approved(
            unresolved_false_closed_count=0,
            false_open_count=0,
            skipped_pairs=[],
            indoor_navmesh_component_count=2,
        ))


if __name__ == "__main__":
    unittest.main()

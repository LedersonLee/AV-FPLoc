"""Contracts for HSSD RGB pilot scheduling and exterior-opening exclusion."""

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path
import tempfile

import numpy as np


SCRIPT = Path(__file__).resolve().parents[2] / "src" / "avfploc" / "preprocessing" / "hssd" / "render_hssd_unloc_rgb_pilot.py"


def load_module():
    return __import__("avfploc.preprocessing.hssd.render_hssd_unloc_rgb_pilot", fromlist=["*"])


class HSSDUnLocRGBPilotContractTest(unittest.TestCase):
    def test_simulator_backend_overrides_empty_hssd_default_lighting_key(self):
        """HSSD declares an empty default_lighting string, which aborts this Habitat build."""
        renderer = load_module()

        class Backend:
            pass

        class FakeHabitat:
            class gfx:
                DEFAULT_LIGHTING_KEY = "default"

            @staticmethod
            def SimulatorConfiguration():
                return Backend()

        backend = renderer.configure_simulator_backend(
            FakeHabitat, root=Path("/dataset"), scene_id="scene-a", gpu_device_id=3,
        )

        self.assertEqual(backend.scene_id, "/dataset/scenes/scene-a.scene_instance.json")
        self.assertEqual(backend.scene_dataset_config_file, "/dataset/hssd-hab.scene_dataset_config.json")
        self.assertEqual(backend.scene_light_setup, "default")
        self.assertTrue(backend.override_scene_light_defaults)

    def test_sanitized_scene_instance_removes_only_known_crashing_asset(self):
        """A renderer-specific derived scene must preserve all valid furniture."""
        renderer = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.scene_instance.json"
            source.write_text(json.dumps({"object_instances": [
                {"template_name": "valid-a"},
                {"template_name": "broken-template"},
                {"template_name": "valid-b"},
            ], "stage_instance": {"template_name": "stage"}}), encoding="utf-8")

            output, removed = renderer.write_sanitized_scene_instance(
                source, Path(tmp) / "derived", excluded_templates={"broken-template"},
            )

            self.assertEqual(removed, ["broken-template"])
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(
                [entry["template_name"] for entry in payload["object_instances"]], ["valid-a", "valid-b"],
            )
            self.assertEqual(payload["stage_instance"]["template_name"], "stage")

    def test_fov_rays_use_requested_camera_extent_and_logical_yaw(self):
        """Catches a left/right or camera-FoV mismatch in floorplan ray filtering."""
        renderer = load_module()
        bearings = renderer.fov_ray_bearings(logical_yaw_deg=0.0, hfov_deg=100.0, ray_count=5)
        np.testing.assert_allclose(bearings, [-50.0, -25.0, 0.0, 25.0, 50.0])

    def test_habitat_yaw_conversion_preserves_camera_forward_in_the_map_frame(self):
        """Catches a mirrored GT-pose arrow or RGB yaw convention."""
        renderer = load_module()
        for logical_yaw in (0.0, 45.0, 90.0, 135.0, 180.0, 270.0):
            expected = renderer.logical_camera_forward_xz(logical_yaw)
            rendered = renderer.habitat_camera_forward_xz(-logical_yaw - 90.0)
            np.testing.assert_allclose(rendered, expected, atol=1e-7)

    def test_exterior_opening_filter_rejects_only_views_dominated_by_exterior_hits(self):
        """Catches rejection of all window views instead of only exterior-dominated ones."""
        renderer = load_module()
        self.assertFalse(renderer.exclude_for_exterior_opening(np.array([0, 0, 2, 0, 0]), max_fraction=0.25))
        self.assertTrue(renderer.exclude_for_exterior_opening(np.array([2, 2, 0, 2, 0]), max_fraction=0.25))

    def test_raycast_detects_exterior_opening_even_when_geometry_marks_portals_free(self):
        """Catches a filter that steps straight through a deliberately-free exterior portal."""
        renderer = load_module()
        raster = renderer.WorldRaster(0.0, 4.0, 0.0, 4.0, 1.0)
        geometry = np.full((5, 5), 255, dtype=np.uint8)
        scope = np.zeros((5, 5), dtype=np.uint8)
        scope[2, 3] = 2
        hit = renderer._first_hit_scope(
            x_m=1.0, z_m=2.0, bearing_deg=0.0, geometry=geometry, scope=scope, raster=raster, max_range_m=3.0,
        )
        self.assertEqual(hit, 2)

    def test_even_pose_subset_retains_coverage_across_the_200_pose_schedule(self):
        """Catches a pilot that accidentally renders only the first room's poses."""
        renderer = load_module()
        records = [{"pose_index": index, "sample_index": index * 8} for index in range(200)]
        subset = renderer.even_pose_subset(records, pose_count=5)
        self.assertEqual([record["pose_index"] for record in subset], [0, 50, 100, 149, 199])

    def test_vertical_clearance_rejects_a_pose_when_one_body_height_is_inside_geometry(self):
        """Catches 1.5-m-only sampling that accepts a camera inside a shelf."""
        renderer = load_module()
        clear = np.full((3, 4, 20, 20), 1.0, dtype=np.float32)
        # At the middle height, one of the four horizontal sweeps sees a
        # nearby obstacle across the central view band.
        clear[1, 2, 7:13, :] = 0.08

        report = renderer.vertical_clearance_report(
            clear, heights_m=(0.5, 1.0, 1.5), min_clearance_m=0.30,
        )

        self.assertFalse(report["accepted"])
        self.assertLess(report["minimum_clearance_m"], 0.30)
        self.assertEqual(report["failing_heights_m"], [1.0])

    def test_vertical_clearance_accepts_pose_clear_at_all_body_heights(self):
        """Catches an over-strict test that discards normal furnished-room views."""
        renderer = load_module()
        clear = np.full((3, 4, 20, 20), 0.65, dtype=np.float32)

        report = renderer.vertical_clearance_report(
            clear, heights_m=(0.5, 1.0, 1.5), min_clearance_m=0.30,
        )

        self.assertTrue(report["accepted"])
        self.assertAlmostEqual(report["minimum_clearance_m"], 0.65)

    def test_completed_scene_requires_manifest_and_every_accepted_rgb_file(self):
        """Catches parallel resume skipping a scene with a truncated RGB write."""
        renderer = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            scene = Path(tmp) / "scene-a"
            (scene / "rgb").mkdir(parents=True)
            manifest = {
                "scene_id": "scene-a",
                "pose_count_requested": 200,
                "records": [
                    {"accepted": True, "image": "rgb/00000.png"},
                    {"accepted": False, "exclusion_reason": "insufficient_vertical_clearance"},
                ],
            }
            (scene / "pilot_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            self.assertFalse(renderer.scene_render_is_complete(scene, pose_count=200))
            (scene / "rgb" / "00000.png").write_bytes(b"png")
            self.assertTrue(renderer.scene_render_is_complete(scene, pose_count=200))
            self.assertFalse(renderer.scene_render_is_complete(scene, pose_count=199))

    def test_canonical_rgb_folder_requires_explicit_legacy_ungated_opt_in(self):
        renderer = load_module()
        canonical = Path("/ssd1/AV-FPLoc/datasets/HSSD/UnLoc/RGB")
        with self.assertRaisesRegex(ValueError, "QA gate"):
            renderer.ensure_legacy_capture_destination(canonical, allow_legacy_ungated=False)
        renderer.ensure_legacy_capture_destination(canonical, allow_legacy_ungated=True)
        renderer.ensure_legacy_capture_destination(Path("/tmp/hssd-pilot"), allow_legacy_ungated=False)


if __name__ == "__main__":
    unittest.main()

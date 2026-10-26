"""Contracts for the HSSD vision-only UnLoc dataset builder."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


SCRIPT = Path(__file__).resolve().parents[2] / "src" / "avfploc" / "preprocessing" / "hssd" / "hssd_unloc_dataset.py"


def load_dataset_module():
    return __import__("avfploc.preprocessing.hssd.hssd_unloc_dataset", fromlist=["*"])


class HSSDUnLocDatasetContractTest(unittest.TestCase):
    def test_scene_height_inspection_accepts_only_a_single_zero_floor(self):
        self.assertTrue(
            SCRIPT.is_file(),
            "hssd_unloc_dataset must expose the HSSD eligibility contract",
        )
        dataset = load_dataset_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            single = root / "single.semantic_config.json"
            multi = root / "multi.semantic_config.json"
            single.write_text(json.dumps({"region_annotations": [
                {"floor_height": 0.0}, {"floor_height": 0.00009},
            ]}), encoding="utf-8")
            multi.write_text(json.dumps({"region_annotations": [
                {"floor_height": 0.0}, {"floor_height": 2.8},
            ]}), encoding="utf-8")
            self.assertTrue(dataset.inspect_scene_heights(single).eligible)
            rejected = dataset.inspect_scene_heights(multi)
            self.assertFalse(rejected.eligible)
            self.assertIn("multiple", rejected.reason)

    def test_split_is_disjoint_sized_and_keeps_requested_scene_in_test(self):
        self.assertTrue(SCRIPT.is_file())
        dataset = load_dataset_module()
        scene_ids = [f"scene_{index:03d}" for index in range(160)]
        split = dataset.build_split(scene_ids, test_scene="scene_009", seed=31415)
        self.assertEqual({key: len(value) for key, value in split.items()}, {
            "train": 120, "val": 20, "test": 20,
        })
        self.assertIn("scene_009", split["test"])
        self.assertEqual(sum(map(len, split.values())), len(set().union(*split.values())))
        self.assertEqual(split, dataset.build_split(scene_ids, test_scene="scene_009", seed=31415))

    def test_world_raster_round_trip_preserves_x_and_z_without_display_flip(self):
        self.assertTrue(SCRIPT.is_file())
        dataset = load_dataset_module()
        raster = dataset.WorldRaster(
            x_min_m=-9.2, x_max_m=-0.5,
            z_min_m=-5.9, z_max_m=3.4,
            spacing_m=0.1,
        )
        row, col = raster.world_to_rc(x_m=-1.8, z_m=0.4)
        x_m, z_m = raster.rc_to_world(row, col)
        self.assertAlmostEqual(x_m, -1.8)
        self.assertAlmostEqual(z_m, 0.4)
        self.assertEqual((row, col), (63, 74))

    def test_world_raster_is_a_physical_topdown_view_with_positive_z_downward(self):
        """With +Y above the floor, screen-down must correspond to world +Z."""
        dataset = load_dataset_module()
        raster = dataset.WorldRaster(
            x_min_m=-2.0, x_max_m=2.0,
            z_min_m=-3.0, z_max_m=3.0,
            spacing_m=1.0,
        )

        self.assertEqual(raster.world_to_rc(-2.0, -3.0), (0, 0))
        self.assertEqual(raster.world_to_rc(-2.0, 3.0), (6, 0))
        self.assertEqual(raster.rc_to_world(1, 3), (1.0, -2.0))

    def test_semantic_scan_records_accepted_and_rejected_scene_reasons(self):
        self.assertTrue(SCRIPT.is_file())
        dataset = load_dataset_module()
        with tempfile.TemporaryDirectory() as tmp:
            semantic_dir = Path(tmp)
            (semantic_dir / "accepted.semantic_config.json").write_text(
                json.dumps({"region_annotations": [{"floor_height": 0.0}]}),
                encoding="utf-8",
            )
            (semantic_dir / "rejected.semantic_config.json").write_text(
                json.dumps({"region_annotations": [{"floor_height": 0.0}, {"floor_height": 2.8}]}),
                encoding="utf-8",
            )
            accepted, rejected = dataset.scan_semantic_scenes(semantic_dir)
        self.assertEqual(accepted, ["accepted"])
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["scene_id"], "rejected")
        self.assertIn("multiple", rejected[0]["reason"])

    def test_outdoor_labeled_regions_are_excluded_from_the_indoor_map_extent(self):
        dataset = load_dataset_module()
        regions = [
            {"label": "living room", "poly_loop": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                                                         [1.0, 0.0, 1.0], [0.0, 0.0, 1.0]]},
            {"label": "balcony", "poly_loop": [[10.0, 0.0, 0.0], [11.0, 0.0, 0.0],
                                                     [11.0, 0.0, 1.0], [10.0, 0.0, 1.0]]},
        ]
        raster = dataset.world_raster_for_regions(regions, spacing_m=0.1, padding_m=0.2)
        self.assertLess(raster.x_max_m, 2.0)
        artifacts = dataset.build_structural_floorplan(
            regions, raster, mesh_wall_mask=np.zeros((raster.height, raster.width), dtype=bool),
        )
        self.assertGreater(artifacts.free_mask.sum(), 0)

    def test_structural_floorplan_keeps_region_boundaries_as_walls_and_exterior_blocked(self):
        dataset = load_dataset_module()
        raster = dataset.WorldRaster(
            x_min_m=-1.0, x_max_m=5.0,
            z_min_m=-1.0, z_max_m=5.0,
            spacing_m=1.0,
        )
        artifacts = dataset.build_structural_floorplan([
            {"poly_loop": [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0],
                           [4.0, 0.0, 4.0], [0.0, 0.0, 4.0]]}
        ], raster, wall_thickness_m=0.0)
        wall_row, wall_col = raster.world_to_rc(0.0, 2.0)
        free_row, free_col = raster.world_to_rc(2.0, 2.0)
        exterior_row, exterior_col = raster.world_to_rc(-1.0, -1.0)
        self.assertTrue(artifacts.wall_mask[wall_row, wall_col])
        self.assertTrue(artifacts.indoor_mask[free_row, free_col])
        self.assertTrue(artifacts.free_mask[free_row, free_col])
        self.assertTrue(artifacts.exterior_mask[exterior_row, exterior_col])
        self.assertEqual(artifacts.map_png[free_row, free_col], 255)
        self.assertEqual(artifacts.map_png[wall_row, wall_col], 0)
        self.assertEqual(artifacts.map_png[exterior_row, exterior_col], 0)

    def test_mesh_wall_mask_overrides_semantic_shared_boundary_but_keeps_exterior_blocked(self):
        dataset = load_dataset_module()
        raster = dataset.WorldRaster(
            x_min_m=-1.0, x_max_m=5.0,
            z_min_m=-1.0, z_max_m=3.0,
            spacing_m=0.5,
        )
        regions = [
            {"poly_loop": [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 0.0, 2.0], [0.0, 0.0, 2.0]]},
            {"poly_loop": [[2.0, 0.0, 0.0], [4.0, 0.0, 0.0], [4.0, 0.0, 2.0], [2.0, 0.0, 2.0]]},
        ]
        mesh_wall = np.zeros((raster.height, raster.width), dtype=bool)
        for x_m in range(0, 5):
            for z_m in (0, 2):
                mesh_wall[raster.world_to_rc(float(x_m), float(z_m))] = True
        for z_m in range(0, 3):
            for x_m in (0, 4):
                mesh_wall[raster.world_to_rc(float(x_m), float(z_m))] = True
        artifacts = dataset.build_structural_floorplan(regions, raster, mesh_wall_mask=mesh_wall)
        shared_row, shared_col = raster.world_to_rc(2.0, 1.0)
        exterior_row, exterior_col = raster.world_to_rc(-1.0, -1.0)
        self.assertTrue(artifacts.free_mask[shared_row, shared_col])
        self.assertEqual(artifacts.map_png[shared_row, shared_col], 255)
        self.assertEqual(artifacts.map_png[exterior_row, exterior_col], 0)

    def test_mesh_wall_mode_closes_outer_boundary_but_preserves_an_internal_gap(self):
        dataset = load_dataset_module()
        raster = dataset.WorldRaster(
            x_min_m=-1.0, x_max_m=7.0,
            z_min_m=-1.0, z_max_m=7.0,
            spacing_m=1.0,
        )
        regions = [{"poly_loop": [[0.0, 0.0, 0.0], [6.0, 0.0, 0.0],
                                  [6.0, 0.0, 6.0], [0.0, 0.0, 6.0]]}]
        mesh_wall = np.zeros((raster.height, raster.width), dtype=bool)
        for z_m in (0, 1, 2, 4, 5, 6):
            mesh_wall[raster.world_to_rc(3.0, float(z_m))] = True
        artifacts = dataset.build_structural_floorplan(
            regions, raster, mesh_wall_mask=mesh_wall,
        )
        centre = raster.world_to_rc(2.0, 2.0)
        edge = raster.world_to_rc(0.0, 2.0)
        internal_gap = raster.world_to_rc(3.0, 3.0)
        exterior = raster.world_to_rc(-1.0, 3.0)
        self.assertTrue(artifacts.free_mask[centre])
        self.assertTrue(artifacts.wall_mask[edge])
        self.assertTrue(artifacts.outer_boundary_mask[edge])
        self.assertTrue(artifacts.free_mask[internal_gap])
        self.assertFalse(artifacts.wall_mask[internal_gap])
        self.assertEqual(artifacts.map_png[exterior], 0)

    def test_small_semantic_annotation_gap_is_an_indoor_doorway_not_exterior(self):
        """Adjacent rooms may leave centimetre-scale unlabeled strips in HSSD."""

        dataset = load_dataset_module()
        raster = dataset.WorldRaster(
            x_min_m=-0.5, x_max_m=2.7,
            z_min_m=-0.5, z_max_m=1.5,
            spacing_m=0.05,
        )
        regions = [
            {"poly_loop": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                           [1.0, 0.0, 1.0], [0.0, 0.0, 1.0]]},
            {"poly_loop": [[1.15, 0.0, 0.0], [2.15, 0.0, 0.0],
                           [2.15, 0.0, 1.0], [1.15, 0.0, 1.0]]},
        ]
        artifacts = dataset.build_structural_floorplan(
            regions, raster, wall_thickness_m=0.0,
            mesh_wall_mask=np.zeros((raster.height, raster.width), dtype=bool),
        )
        doorway = raster.world_to_rc(1.10, 0.50)
        self.assertTrue(artifacts.indoor_mask[doorway])
        self.assertFalse(artifacts.outer_boundary_mask[doorway])
        self.assertTrue(artifacts.free_mask[doorway])

    def test_semantic_only_bootstrap_closes_its_derived_outer_contour(self):
        """The manifest bootstrap must not leak after semantic seam closure."""

        dataset = load_dataset_module()
        raster = dataset.WorldRaster(
            x_min_m=-0.5, x_max_m=2.7,
            z_min_m=-0.5, z_max_m=1.5,
            spacing_m=0.05,
        )
        regions = [
            {"poly_loop": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                           [1.0, 0.0, 1.0], [0.0, 0.0, 1.0]]},
            {"poly_loop": [[1.15, 0.0, 0.0], [2.15, 0.0, 0.0],
                           [2.15, 0.0, 1.0], [1.15, 0.0, 1.0]]},
        ]
        artifacts = dataset.build_structural_floorplan(regions, raster, wall_thickness_m=0.0)
        edge = raster.world_to_rc(0.0, 0.50)
        self.assertTrue(artifacts.wall_mask[edge])
        self.assertTrue(artifacts.outer_boundary_mask[edge])

    def test_connectivity_audit_flags_a_navmesh_open_pair_that_map_blocks(self):
        """A closed internal doorway must not pass map QA merely by pose snapping."""

        dataset = load_dataset_module()
        free = np.ones((7, 7), dtype=bool)
        free[:, 3] = False
        report = dataset.audit_map_connectivity(
            free,
            {
                "left_room": (3, 1),
                "right_room": (3, 5),
            },
            {("left_room", "right_room"): True},
        )
        self.assertEqual(report["component_count"], 2)
        self.assertEqual(report["false_closed_pair_count"], 1)
        self.assertEqual(report["false_open_pair_count"], 0)
        pair = report["pairs"][0]
        self.assertTrue(pair["navmesh_reachable"])
        self.assertFalse(pair["map_connected"])
        self.assertEqual(pair["status"], "false_closed")

    def test_navmesh_path_repair_opens_only_an_internal_corridor(self):
        """A Habitat-confirmed path may repair a missed door, never the exterior contour."""

        dataset = load_dataset_module()
        raster = dataset.WorldRaster(
            x_min_m=0.0, x_max_m=6.0,
            z_min_m=0.0, z_max_m=6.0,
            spacing_m=1.0,
        )
        wall = np.zeros((raster.height, raster.width), dtype=bool)
        wall[:, 3] = True
        outer = np.zeros_like(wall)
        outer[[0, -1], :] = True
        outer[:, [0, -1]] = True
        wall |= outer
        repaired = dataset.carve_internal_navmesh_path(
            wall,
            indoor_mask=np.ones_like(wall),
            outer_boundary_mask=outer,
            raster=raster,
            path_xz_m=np.asarray([[1.0, 3.0], [5.0, 3.0]]),
            corridor_radius_m=0.0,
        )
        internal_door = raster.world_to_rc(3.0, 3.0)
        exterior_edge = raster.world_to_rc(0.0, 3.0)
        self.assertFalse(repaired[internal_door])
        self.assertTrue(repaired[exterior_edge])

    def test_internal_portal_is_free_but_outer_boundary_stops_desdf_ray(self):
        dataset = load_dataset_module()
        raster = dataset.WorldRaster(
            x_min_m=-1.0, x_max_m=7.0,
            z_min_m=-1.0, z_max_m=7.0,
            spacing_m=1.0,
        )
        regions = [{"poly_loop": [[0.0, 0.0, 0.0], [6.0, 0.0, 0.0],
                                  [6.0, 0.0, 6.0], [0.0, 0.0, 6.0]]}]
        mesh_wall = np.zeros((raster.height, raster.width), dtype=bool)
        for x_m in range(0, 7):
            for z_m in (0, 6):
                mesh_wall[raster.world_to_rc(float(x_m), float(z_m))] = True
        for z_m in range(0, 7):
            for x_m in (0, 6):
                mesh_wall[raster.world_to_rc(float(x_m), float(z_m))] = True
        portal = np.zeros_like(mesh_wall)
        outer_portal = raster.world_to_rc(3.0, 5.0)
        internal_portal = raster.world_to_rc(3.0, 4.0)
        portal[outer_portal] = True
        portal[internal_portal] = True
        artifacts = dataset.build_structural_floorplan(
            regions, raster, mesh_wall_mask=mesh_wall, portal_mask=portal,
        )
        portal_row, portal_col = internal_portal
        self.assertTrue(artifacts.portal_mask[portal_row, portal_col])
        self.assertFalse(artifacts.wall_mask[portal_row, portal_col])
        self.assertTrue(artifacts.free_mask[portal_row, portal_col])
        self.assertEqual(artifacts.map_png[portal_row, portal_col], 255)
        self.assertFalse(artifacts.portal_mask[outer_portal])
        self.assertTrue(artifacts.outer_boundary_mask[outer_portal])
        self.assertTrue(artifacts.wall_mask[outer_portal])
        desdf = dataset.build_directional_desdf(
            artifacts.free_mask, occluder_mask=artifacts.desdf_occluder_mask,
            spacing_m=1.0, yaw_bins=4, max_range_m=8.0,
        )
        source_row, source_col = raster.world_to_rc(3.0, 3.0)
        # Bin 1 (90°) is world +Z: screen-down in the physical top-down map.
        self.assertEqual(float(desdf[source_row, source_col, 1]), 2.0)

    def test_directional_desdf_reports_first_wall_distance_in_metres(self):
        dataset = load_dataset_module()
        free = np.ones((3, 4), dtype=bool)
        free[:, 3] = False
        desdf = dataset.build_directional_desdf(
            free, spacing_m=0.1, yaw_bins=4, max_range_m=1.0,
        )
        self.assertAlmostEqual(float(desdf[1, 1, 0]), 0.2)
        self.assertEqual(desdf.shape, (3, 4, 4))

    def test_directional_desdf_can_raycast_at_one_centimetre_and_store_at_ten_centimetres(self):
        dataset = load_dataset_module()
        free = np.ones((20, 30), dtype=bool)
        free[:, 20:] = False
        desdf = dataset.build_directional_desdf(
            free, spacing_m=0.01, output_spacing_m=0.1,
            yaw_bins=4, max_range_m=1.0,
        )
        self.assertEqual(desdf.shape, (2, 3, 4))
        self.assertAlmostEqual(float(desdf[1, 1, 0]), 0.1)

    def test_directional_desdf_preserves_diagonal_ray_geometry(self):
        dataset = load_dataset_module()
        free = np.ones((8, 8), dtype=bool)
        free[2, 4] = False
        desdf = dataset.build_directional_desdf(
            free, spacing_m=0.1, yaw_bins=8, max_range_m=1.0,
        )
        # Bin 7 is up-right in row-down raster coordinates; its first hit is
        # two cells up-right, not a cardinal 0.2 m step.
        self.assertAlmostEqual(float(desdf[4, 2, 7]), np.sqrt(8) * 0.1, places=5)

    def test_floorplan_writer_emits_f3loc_map_masks_desdf_and_metric_metadata(self):
        dataset = load_dataset_module()
        regions = [{"poly_loop": [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0],
                                  [4.0, 0.0, 4.0], [0.0, 0.0, 4.0]]}]
        raster = dataset.world_raster_for_regions(regions, spacing_m=1.0, padding_m=1.0)
        self.assertEqual((raster.x_min_m, raster.z_min_m), (-1.0, -1.0))
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "scene"
            metadata = dataset.write_floorplan_artifacts(
                output, regions, raster,
                scene_id="synthetic", semantic_path=Path("semantic.json"),
            )
            self.assertTrue((output / "map.png").is_file())
            self.assertTrue((output / "indoor_mask.png").is_file())
            self.assertTrue((output / "outer_boundary_mask.png").is_file())
            self.assertTrue((output / "wall_mask.png").is_file())
            self.assertTrue((output / "portal_mask.png").is_file())
            self.assertTrue((output / "desdf_occluder_mask.png").is_file())
            self.assertTrue((output / "desdf.npy").is_file())
            self.assertEqual(np.load(output / "desdf.npy", allow_pickle=True).item()["desdf"].shape[-1], 36)
            self.assertEqual(np.asarray(Image.open(output / "map.png")).dtype, np.uint8)
        self.assertEqual(metadata["scene_id"], "synthetic")
        self.assertEqual(metadata["map_png_polarity"], "255=free,0=blocked")
        self.assertEqual(metadata["portal_policy"], "door_window_glass=free_non_occluding")

    def test_pose_sampler_keeps_clearance_and_schedule_has_all_eight_yaws_at_eye_height(self):
        dataset = load_dataset_module()
        raster = dataset.WorldRaster(
            x_min_m=0.0, x_max_m=3.9,
            z_min_m=0.0, z_max_m=3.9,
            spacing_m=0.1,
        )
        free = np.ones((40, 40), dtype=bool)
        free[[0, -1], :] = False
        free[:, [0, -1]] = False
        poses = dataset.sample_poses(
            free, raster, count=200, min_clearance_m=0.10, seed=42,
        )
        self.assertEqual(poses.shape, (200, 2))
        self.assertTrue(np.all(free[poses[:, 0], poses[:, 1]]))
        self.assertTrue(np.array_equal(
            poses, dataset.sample_poses(free, raster, count=200, min_clearance_m=0.10, seed=42),
        ))
        schedule = dataset.build_capture_schedule(poses, raster, floor_y_m=0.0)
        self.assertEqual(len(schedule), 1600)
        self.assertEqual({record["logical_yaw_deg"] for record in schedule}, set(range(0, 360, 45)))
        self.assertTrue(all(record["pose_xyz_m"][1] == 1.5 for record in schedule))
        self.assertEqual(next(record["habitat_yaw_deg"] for record in schedule if record["logical_yaw_deg"] == 90), -180.0)
        self.assertTrue(all(abs(record["camera_vfov_deg"] - 90.0) < 1e-3 for record in schedule))

    def test_navmesh_snap_metrics_rejects_a_pose_outside_horizontal_tolerance(self):
        dataset = load_dataset_module()
        poses = np.asarray([[0.0, 1.5, 0.0], [1.0, 1.5, 1.0]], dtype=np.float32)

        def snap(point):
            return point if point[0] == 0.0 else np.asarray([1.25, 1.5, 1.0], dtype=np.float32)

        metrics = dataset.navmesh_snap_metrics(poses, snap, horizontal_tolerance_m=0.10)
        self.assertEqual(metrics["pose_count"], 2)
        self.assertEqual(metrics["within_tolerance_count"], 1)
        self.assertFalse(metrics["all_within_tolerance"])
        self.assertAlmostEqual(metrics["max_horizontal_snap_distance_m"], 0.25)

    def test_stage_slice_rasterizes_vertical_mesh_surface_at_camera_height(self):
        dataset = load_dataset_module()
        raster = dataset.WorldRaster(
            x_min_m=-1.0, x_max_m=1.0,
            z_min_m=-1.0, z_max_m=4.0,
            spacing_m=1.0,
        )
        triangles = np.array([
            [[0.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 3.0, 3.0]],
            [[0.0, 0.0, 0.0], [0.0, 3.0, 3.0], [0.0, 0.0, 3.0]],
        ], dtype=np.float64)
        wall = dataset.wall_mask_from_vertical_triangles(
            triangles, raster, cut_height_m=1.5, wall_thickness_m=0.0,
        )
        row, col = raster.world_to_rc(0.0, 2.0)
        self.assertTrue(wall[row, col])
        self.assertFalse(wall[row, col + 1])

    def test_stage_slice_fills_the_narrow_core_between_two_wall_faces(self):
        """The physical wall volume must never remain a white map corridor."""

        dataset = load_dataset_module()
        raster = dataset.WorldRaster(
            x_min_m=-0.1, x_max_m=0.3,
            z_min_m=-0.1, z_max_m=1.1,
            spacing_m=0.01,
        )
        triangles = np.array([
            [[0.00, 0.0, 0.0], [0.00, 3.0, 0.0], [0.00, 3.0, 1.0]],
            [[0.00, 0.0, 0.0], [0.00, 3.0, 1.0], [0.00, 0.0, 1.0]],
            [[0.12, 0.0, 0.0], [0.12, 3.0, 0.0], [0.12, 3.0, 1.0]],
            [[0.12, 0.0, 0.0], [0.12, 3.0, 1.0], [0.12, 0.0, 1.0]],
        ], dtype=np.float64)
        wall = dataset.wall_mask_from_vertical_triangles(
            triangles, raster, cut_height_m=1.5, wall_thickness_m=0.02,
        )
        core = raster.world_to_rc(0.06, 0.50)
        self.assertTrue(wall[core])

    def test_stage_slice_fills_a_long_enclosed_wall_core_but_keeps_a_wider_room_void(self):
        """Catches double-sided HSSD walls whose 0.5-m hollow core remains free."""
        dataset = load_dataset_module()
        wall = np.zeros((80, 80), dtype=bool)
        # A 0.5-m-wide, 5.8-m-long wall volume represented only by its four
        # shell faces.  The enclosed white core must not be a pose candidate.
        wall[10:70, 20] = wall[10:70, 26] = True
        wall[10, 20:27] = wall[69, 20:27] = True
        # A distinct 0.9-m-wide enclosed area remains too wide to infer as a
        # wall core; this guards against blanket hole filling.
        wall[10:70, 40] = wall[10:70, 50] = True
        wall[10, 40:51] = wall[69, 40:51] = True

        filled = dataset.fill_enclosed_wall_cores(wall, spacing_m=0.10)

        self.assertTrue(filled[40, 23])
        self.assertFalse(filled[40, 45])

    def test_stage_slice_keeps_a_door_width_gap_open(self):
        dataset = load_dataset_module()
        raster = dataset.WorldRaster(
            x_min_m=-0.1, x_max_m=1.1,
            z_min_m=-0.1, z_max_m=1.1,
            spacing_m=0.01,
        )
        triangles = np.array([
            [[0.00, 0.0, 0.0], [0.00, 3.0, 0.0], [0.00, 3.0, 1.0]],
            [[0.00, 0.0, 0.0], [0.00, 3.0, 1.0], [0.00, 0.0, 1.0]],
            [[0.80, 0.0, 0.0], [0.80, 3.0, 0.0], [0.80, 3.0, 1.0]],
            [[0.80, 0.0, 0.0], [0.80, 3.0, 1.0], [0.80, 0.0, 1.0]],
        ], dtype=np.float64)
        wall = dataset.wall_mask_from_vertical_triangles(
            triangles, raster, cut_height_m=1.5, wall_thickness_m=0.02,
        )
        doorway = raster.world_to_rc(0.40, 0.50)
        self.assertFalse(wall[doorway])

    def test_stage_slice_clips_structure_outside_semantic_indoor_crop(self):
        dataset = load_dataset_module()
        raster = dataset.WorldRaster(
            x_min_m=-1.0, x_max_m=1.0,
            z_min_m=-1.0, z_max_m=2.0,
            spacing_m=1.0,
        )
        triangles = np.array([
            [[0.0, 0.0, -2.0], [0.0, 3.0, -2.0], [0.0, 3.0, 4.0]],
            [[0.0, 0.0, -2.0], [0.0, 3.0, 4.0], [0.0, 0.0, 4.0]],
        ], dtype=np.float64)
        wall = dataset.wall_mask_from_vertical_triangles(
            triangles, raster, cut_height_m=1.5, wall_thickness_m=0.0,
        )
        row, col = raster.world_to_rc(0.0, 1.0)
        self.assertTrue(wall[row, col])


if __name__ == "__main__":
    unittest.main()

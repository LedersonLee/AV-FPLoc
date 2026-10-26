"""Behavioral contracts for the portable HSSD geometry/semantic map export."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[2] / "src" / "avfploc" / "preprocessing" / "hssd" / "build_hssd_unloc_semantic_maps.py"


def load_module():
    return __import__("avfploc.preprocessing.hssd.build_hssd_unloc_semantic_maps", fromlist=["*"])


class HSSDSemanticMapContractTest(unittest.TestCase):
    def test_opening_metadata_maps_template_instances_to_door_or_window(self):
        """Catches a regression that silently discards HSSD opening semantics."""
        semantic = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            metadata = Path(tmp) / "fpmodels.csv"
            metadata.write_text(
                "id,main_category\n202-0,door\n220-44,window\n999-9,chair\n",
                encoding="utf-8",
            )
            categories = semantic.load_opening_categories(metadata)

        self.assertEqual(categories["202-0"], semantic.LABEL_DOOR)
        self.assertEqual(categories["220-44"], semantic.LABEL_WINDOW)
        self.assertNotIn("999-9", categories)
        self.assertEqual(semantic.opening_template_id("202-0_part_3"), "202-0")

    def test_door_and_window_masks_overlay_wall_labels_without_changing_free_space(self):
        """Catches a regression that turns the entire wall or room into a portal."""
        semantic = load_module()
        geometry = np.array([
            [0, 0, 0, 0, 0],
            [0, 255, 255, 255, 0],
            [0, 255, 255, 255, 0],
            [0, 0, 0, 0, 0],
        ], dtype=np.uint8)
        door = np.zeros_like(geometry, dtype=bool)
        window = np.zeros_like(geometry, dtype=bool)
        door[0, 2] = True
        window[3, 1] = True

        labels = semantic.compose_semantic_labels(geometry, door_mask=door, window_mask=window)

        expected = np.array([
            [1, 1, 2, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 3, 1, 1, 1],
        ], dtype=np.uint8)
        np.testing.assert_array_equal(labels, expected)

    def test_semantic_rgb_palette_keeps_geometry_map_and_semantic_map_distinct(self):
        """Catches accidental reuse of binary map.png as the semantic export."""
        semantic = load_module()
        labels = np.array([[0, 1, 2, 3, 4, 5]], dtype=np.uint8)
        rgb = semantic.labels_to_rgb(labels)
        expected = np.array([[[255, 255, 255], [0, 0, 0], [220, 38, 38], [37, 99, 235], [245, 158, 11], [6, 182, 212]]], dtype=np.uint8)
        np.testing.assert_array_equal(rgb, expected)

    def test_opening_instance_without_rotation_uses_habitat_identity_quaternion(self):
        """Catches a KeyError on valid HSSD window assets without a rotation field."""
        semantic = load_module()
        scale, rotation, translation = semantic.instance_transform_components({
            "translation": [-8.0, 1.45, -6.09],
            "non_uniform_scale": [-1.0, 1.0, 1.0],
        })
        np.testing.assert_array_equal(scale, [-1.0, 1.0, 1.0])
        np.testing.assert_array_equal(rotation, [1.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_equal(translation, [-8.0, 1.45, -6.09])

    def test_opening_scope_marks_outer_contour_openings_as_exterior_only(self):
        """Catches RGB filtering that confuses an exterior window with an internal door."""
        semantic = load_module()
        door = np.zeros((5, 5), dtype=bool)
        window = np.zeros((5, 5), dtype=bool)
        outer = np.zeros((5, 5), dtype=bool)
        outer[0, :] = True
        door[2, 2] = True
        window[0, 3] = True

        scope = semantic.classify_opening_scope(door, window, outer, spacing_m=0.10)

        self.assertEqual(scope[2, 2], semantic.OPENING_SCOPE_INTERNAL)
        self.assertEqual(scope[0, 3], semantic.OPENING_SCOPE_EXTERIOR)
        self.assertEqual(scope[4, 4], semantic.OPENING_SCOPE_NONE)

    def test_opening_scope_keeps_exterior_openings_but_drops_assets_without_free_space_contact(self):
        """Catches deletion of exterior openings and wall-embedded semantic artifacts."""
        semantic = load_module()
        indoor = np.zeros((7, 7), dtype=bool)
        indoor[2:5, 2:5] = True
        outer = np.zeros((7, 7), dtype=bool)
        outer[0, :] = True
        internal_door = np.zeros((7, 7), dtype=bool)
        exterior_window = np.zeros((7, 7), dtype=bool)
        terrace_window = np.zeros((7, 7), dtype=bool)
        internal_door[3, 3] = True
        exterior_window[0, 3] = True
        terrace_window[6, 6] = True

        groups = semantic.split_opening_masks_for_training(
            [(semantic.LABEL_DOOR, internal_door), (semantic.LABEL_WINDOW, exterior_window), (semantic.LABEL_WINDOW, terrace_window)],
            indoor_mask=indoor, free_mask=indoor, outer_boundary_mask=outer, spacing_m=0.10,
        )

        self.assertTrue(groups["training_door"][3, 3])
        self.assertTrue(groups["exterior_window"][0, 3])
        self.assertTrue(groups["dropped_window"][6, 6])
        self.assertFalse(groups["training_window"].any())

    def test_wall_embedded_opening_is_dropped_even_if_near_the_outer_contour(self):
        """Catches labels painted on occupied outer-wall pixels without a usable free side."""
        semantic = load_module()
        free = np.zeros((9, 9), dtype=bool)
        free[4, 4] = True
        outer = np.zeros((9, 9), dtype=bool)
        outer[0, :] = True
        wall_embedded_window = np.zeros((9, 9), dtype=bool)
        wall_embedded_window[0, 1] = True

        groups = semantic.split_opening_masks_for_training(
            [(semantic.LABEL_WINDOW, wall_embedded_window)], indoor_mask=free,
            free_mask=free, outer_boundary_mask=outer, spacing_m=0.10,
        )

        self.assertTrue(groups["dropped_window"][0, 1])
        self.assertFalse(groups["exterior_window"].any())

    def test_structural_free_wall_overlap_is_a_detectable_invariant_violation(self):
        """Catches a future export that turns a sampled physical wall core white."""
        semantic = load_module()
        geometry = np.array([[255, 0], [255, 255]], dtype=np.uint8)
        wall = np.array([[0, 0], [255, 0]], dtype=np.uint8)

        self.assertEqual(semantic.structural_free_wall_overlap_pixels(geometry, wall), 1)

    def test_exterior_openings_are_visible_as_distinct_semantic_classes(self):
        """Catches removal of genuine indoor-to-exterior doors/windows from the semantic map."""
        semantic = load_module()
        geometry = np.zeros((2, 3), dtype=np.uint8)
        door = np.zeros_like(geometry, dtype=bool)
        exterior_window = np.zeros_like(geometry, dtype=bool)
        door[0, 1] = True
        exterior_window[1, 2] = True

        labels = semantic.compose_semantic_labels(
            geometry, door_mask=door, window_mask=np.zeros_like(door),
            exterior_door_mask=np.zeros_like(door), exterior_window_mask=exterior_window,
        )

        self.assertEqual(labels[0, 1], semantic.LABEL_DOOR)
        self.assertEqual(labels[1, 2], semantic.LABEL_EXTERIOR_WINDOW)
        np.testing.assert_array_equal(semantic.labels_to_rgb(labels)[1, 2], semantic.PALETTE[semantic.LABEL_EXTERIOR_WINDOW])


if __name__ == "__main__":
    unittest.main()

"""Pure contracts shared by the HSSD vision-only UnLoc dataset tools."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
from PIL import Image
from scipy import ndimage


FLOOR_HEIGHT_TOLERANCE_M = 1e-3
# HSSD's adjacent region polygons sometimes stop short of each other at an
# actual doorway/corridor.  This is annotation quantisation, not an exterior
# gap.  It is deliberately smaller than the 0.30 m raster padding and is
# cross-checked later against Habitat connectivity before bulk acceptance.
SEMANTIC_GAP_CLOSURE_M = 0.20
# Stage walls are commonly represented by two parallel vertical surfaces.  A
# core thinner than this is wall volume, not a navigable passage.  Real door
# apertures are substantially wider and remain open for the portal policy.
MAX_WALL_CORE_GAP_M = 0.16
# Some HSSD stages encode one thick architectural wall as two widely separated
# surface shells.  Such a closed, narrow cavity is wall volume, not a room.
MAX_ENCLOSED_WALL_CORE_NARROW_SPAN_M = 0.70
EXTERIOR_REGION_LABELS = frozenset({"outdoor", "balcony", "porch/terrace/deck/driveway"})


@dataclass(frozen=True)
class SceneEligibility:
    """Single-floor eligibility derived from HSSD region annotations."""

    eligible: bool
    floor_heights_m: tuple[float, ...]
    reason: str


def indoor_region_annotations(regions: list[dict[str, object]]) -> list[dict[str, object]]:
    """Exclude HSSD's explicitly exterior/open-air semantic regions."""

    indoor = [
        region for region in regions
        if str(region.get("label", "")).strip().lower() not in EXTERIOR_REGION_LABELS
    ]
    if not indoor:
        raise ValueError("semantic scene has no non-exterior indoor regions")
    return indoor


def inspect_scene_heights(semantic_path: Path) -> SceneEligibility:
    """Accept only scenes whose semantic regions share the zero-height floor."""

    payload = json.loads(Path(semantic_path).read_text(encoding="utf-8"))
    regions = payload.get("region_annotations", [])
    if not regions:
        return SceneEligibility(False, (), "no region annotations")
    heights = sorted({
        0.0 if abs(float(region.get("floor_height", 0.0))) <= FLOOR_HEIGHT_TOLERANCE_M
        else round(float(region.get("floor_height", 0.0)), 4)
        for region in regions
    })
    height_tuple = tuple(heights)
    if len(height_tuple) != 1:
        return SceneEligibility(False, height_tuple, "multiple semantic floor heights")
    if height_tuple[0] != 0.0:
        return SceneEligibility(False, height_tuple, "single semantic floor is not world zero")
    return SceneEligibility(True, height_tuple, "single zero-height floor")


def scan_semantic_scenes(semantic_dir: Path) -> tuple[list[str], list[dict[str, object]]]:
    """Return sorted eligible IDs and auditable rejected-scene records."""

    accepted: list[str] = []
    rejected: list[dict[str, object]] = []
    suffix = ".semantic_config.json"
    for semantic_path in sorted(Path(semantic_dir).glob(f"*{suffix}")):
        scene_id = semantic_path.name.removesuffix(suffix)
        result = inspect_scene_heights(semantic_path)
        if result.eligible:
            accepted.append(scene_id)
        else:
            rejected.append({
                "scene_id": scene_id,
                "floor_heights_m": list(result.floor_heights_m),
                "reason": result.reason,
                "semantic_path": str(semantic_path),
            })
    return accepted, rejected


def build_split(scene_ids: list[str], test_scene: str, seed: int) -> dict[str, list[str]]:
    """Build the fixed 120/20/20 split and force one scene into test."""

    canonical_ids = sorted(set(scene_ids))
    if len(canonical_ids) != 160:
        raise ValueError(f"expected exactly 160 eligible scenes, got {len(canonical_ids)}")
    if test_scene not in canonical_ids:
        raise ValueError(f"fixed test scene is not eligible: {test_scene}")
    remaining = [scene_id for scene_id in canonical_ids if scene_id != test_scene]
    random.Random(seed).shuffle(remaining)
    split = {
        "test": [test_scene, *remaining[:19]],
        "val": remaining[19:39],
        "train": remaining[39:],
    }
    if [len(split[key]) for key in ("train", "val", "test")] != [120, 20, 20]:
        raise AssertionError("split construction violated the fixed 120/20/20 contract")
    return split


@dataclass(frozen=True)
class WorldRaster:
    """Metric X/Z to physical top-down raster mapping; rows increase with world +Z."""

    x_min_m: float
    x_max_m: float
    z_min_m: float
    z_max_m: float
    spacing_m: float

    def __post_init__(self) -> None:
        if self.spacing_m <= 0:
            raise ValueError("spacing_m must be positive")
        if self.x_max_m <= self.x_min_m or self.z_max_m <= self.z_min_m:
            raise ValueError("world bounds must have positive extent")

    @property
    def height(self) -> int:
        return int(round((self.z_max_m - self.z_min_m) / self.spacing_m)) + 1

    @property
    def width(self) -> int:
        return int(round((self.x_max_m - self.x_min_m) / self.spacing_m)) + 1

    def world_to_rc(self, x_m: float, z_m: float) -> tuple[int, int]:
        col = int(round((x_m - self.x_min_m) / self.spacing_m))
        row = int(round((z_m - self.z_min_m) / self.spacing_m))
        if not (0 <= row < self.height and 0 <= col < self.width):
            raise ValueError(f"world point outside raster: x={x_m}, z={z_m}")
        reconstructed_x, reconstructed_z = self.rc_to_world(row, col)
        if abs(reconstructed_x - x_m) > self.spacing_m / 2 + 1e-9:
            raise ValueError(f"x is not representable on the raster: {x_m}")
        if abs(reconstructed_z - z_m) > self.spacing_m / 2 + 1e-9:
            raise ValueError(f"z is not representable on the raster: {z_m}")
        return row, col

    def rc_to_world(self, row: int, col: int) -> tuple[float, float]:
        if not (0 <= row < self.height and 0 <= col < self.width):
            raise ValueError(f"raster index outside bounds: row={row}, col={col}")
        return (
            self.x_min_m + col * self.spacing_m,
            self.z_min_m + row * self.spacing_m,
        )


@dataclass(frozen=True)
class FloorplanArtifacts:
    """Structural masks and their F3Loc/UnLoc-compatible PNG encoding."""

    indoor_mask: np.ndarray
    outer_boundary_mask: np.ndarray
    wall_mask: np.ndarray
    portal_mask: np.ndarray
    desdf_occluder_mask: np.ndarray
    exterior_mask: np.ndarray
    free_mask: np.ndarray
    map_png: np.ndarray


def _polygon_mask(vertices_xz: np.ndarray, raster: WorldRaster) -> np.ndarray:
    """Return the lattice points strictly inside one X/Z polygon."""

    if vertices_xz.ndim != 2 or vertices_xz.shape[1] != 2 or len(vertices_xz) < 3:
        raise ValueError("each semantic poly_loop needs at least three X/Z vertices")
    cols = np.arange(raster.width, dtype=np.float64)
    rows = np.arange(raster.height, dtype=np.float64)
    x_values = raster.x_min_m + cols * raster.spacing_m
    z_values = raster.z_min_m + rows * raster.spacing_m
    x_grid, z_grid = np.meshgrid(x_values, z_values)
    inside = np.zeros(x_grid.shape, dtype=bool)
    previous = vertices_xz[-1]
    for current in vertices_xz:
        xi, zi = current
        xj, zj = previous
        crosses = (zi > z_grid) != (zj > z_grid)
        with np.errstate(divide="ignore", invalid="ignore"):
            crossing_x = (xj - xi) * (z_grid - zi) / (zj - zi) + xi
        inside ^= crosses & (x_grid < crossing_x)
        previous = current
    return inside


def _rasterize_boundary(vertices_xz: np.ndarray, raster: WorldRaster) -> np.ndarray:
    """Mark every semantic polygon edge on the metric raster."""

    boundary = np.zeros((raster.height, raster.width), dtype=bool)
    for start, end in zip(vertices_xz, np.roll(vertices_xz, -1, axis=0)):
        boundary |= _rasterize_segment(start, end, raster)
    return boundary


def _rasterize_segment(start_xz: np.ndarray, end_xz: np.ndarray, raster: WorldRaster) -> np.ndarray:
    """Mark one X/Z line segment at half-pixel sampling density."""

    line = np.zeros((raster.height, raster.width), dtype=bool)
    length_m = float(np.linalg.norm(end_xz - start_xz))
    count = max(2, int(np.ceil(length_m / (raster.spacing_m / 2))) + 1)
    for x_m, z_m in np.linspace(start_xz, end_xz, count):
        try:
            row, col = raster.world_to_rc(float(x_m), float(z_m))
        except ValueError:
            continue
        line[row, col] = True
    return line


def fill_enclosed_wall_cores(wall_mask: np.ndarray, *, spacing_m: float) -> np.ndarray:
    """Fill a closed, narrow cavity between two shells of one physical wall.

    A normal doorway/corridor is connected to surrounding free space and is
    therefore never a candidate here.  We only fill complement components
    enclosed by wall support whose *short* bounding-box span is no wider than
    the conservative HSSD thick-wall limit.  This removes double-sided mesh
    wall cores without blanket-filling rooms.
    """

    wall = np.asarray(wall_mask, dtype=bool)
    if wall.ndim != 2 or spacing_m <= 0:
        raise ValueError("wall_mask must be 2-D and spacing_m must be positive")
    free_labels, count = ndimage.label(~wall)
    edge_labels = np.unique(np.concatenate((
        free_labels[0, :], free_labels[-1, :], free_labels[:, 0], free_labels[:, -1],
    )))
    enclosed = wall.copy()
    for label in range(1, count + 1):
        if label in edge_labels:
            continue
        rows, cols = np.where(free_labels == label)
        if not len(rows):
            continue
        short_span_m = min(rows.max() - rows.min() + 1, cols.max() - cols.min() + 1) * spacing_m
        if short_span_m <= MAX_ENCLOSED_WALL_CORE_NARROW_SPAN_M:
            enclosed[rows, cols] = True
    return enclosed


def wall_mask_from_vertical_triangles(
    triangles: np.ndarray, raster: WorldRaster, *, cut_height_m: float, wall_thickness_m: float = 0.10,
) -> np.ndarray:
    """Slice stage triangles at camera height and rasterize only vertical intersections."""

    faces = np.asarray(triangles, dtype=np.float64)
    if faces.ndim != 3 or faces.shape[1:] != (3, 3):
        raise ValueError("triangles must have shape (N, 3, 3) in HSSD X/Y/Z coordinates")
    if wall_thickness_m < 0:
        raise ValueError("wall_thickness_m must be non-negative")
    boundary = np.zeros((raster.height, raster.width), dtype=bool)
    for triangle in faces:
        intersections: list[np.ndarray] = []
        for start, end in zip(triangle, np.roll(triangle, -1, axis=0)):
            start_delta, end_delta = start[1] - cut_height_m, end[1] - cut_height_m
            if np.isclose(start_delta, 0.0) and np.isclose(end_delta, 0.0):
                continue
            if (start_delta <= 0 <= end_delta) or (end_delta <= 0 <= start_delta):
                denominator = end[1] - start[1]
                if np.isclose(denominator, 0.0):
                    continue
                fraction = (cut_height_m - start[1]) / denominator
                if 0.0 <= fraction <= 1.0:
                    intersections.append(start + fraction * (end - start))
        unique: list[np.ndarray] = []
        for point in intersections:
            if not any(np.allclose(point, previous, atol=1e-8) for previous in unique):
                unique.append(point)
        if len(unique) == 2:
            boundary |= _rasterize_segment(unique[0][[0, 2]], unique[1][[0, 2]], raster)
    radius_cells = int(np.ceil((wall_thickness_m / 2) / raster.spacing_m))
    wall = ndimage.binary_dilation(boundary, iterations=radius_cells) if radius_cells else boundary
    core_radius_cells = max(1, int(np.ceil(MAX_WALL_CORE_GAP_M / (2 * raster.spacing_m))))
    # Preserve every source face while closing only a thin void between two
    # faces; otherwise the architectural wall's centre becomes a free pixel
    # corridor in map.png.
    wall |= ndimage.binary_closing(wall, iterations=core_radius_cells)
    return fill_enclosed_wall_cores(wall, spacing_m=raster.spacing_m)


def build_structural_floorplan(
    regions: list[dict[str, object]], raster: WorldRaster, *, wall_thickness_m: float = 0.10,
    mesh_wall_mask: np.ndarray | None = None, portal_mask: np.ndarray | None = None,
) -> FloorplanArtifacts:
    """Rasterize indoor support, non-portal walls, and deSDF occluders."""

    if wall_thickness_m < 0:
        raise ValueError("wall_thickness_m must be non-negative")
    regions = indoor_region_annotations(regions)
    indoor = np.zeros((raster.height, raster.width), dtype=bool)
    boundary = np.zeros_like(indoor)
    for region in regions:
        loop = np.asarray(region.get("poly_loop", []), dtype=np.float64)
        if loop.ndim != 2 or loop.shape[1] < 3:
            raise ValueError("region annotation is missing an X/Y/Z poly_loop")
        vertices_xz = loop[:, [0, 2]]
        indoor |= _polygon_mask(vertices_xz, raster)
        boundary |= _rasterize_boundary(vertices_xz, raster)
    if not indoor.any():
        raise ValueError("semantic regions yielded no indoor raster cells")
    gap_cells = int(np.ceil(SEMANTIC_GAP_CLOSURE_M / raster.spacing_m))
    if gap_cells:
        indoor = ndimage.binary_closing(indoor, iterations=gap_cells) | indoor
    portal = np.zeros_like(indoor) if portal_mask is None else np.asarray(portal_mask, dtype=bool)
    if portal.shape != indoor.shape:
        raise ValueError("portal_mask shape must match raster")
    # The exterior contour is deliberately closed in the canonical indoor
    # floorplan. It is a separate class from interior walls so QA can verify
    # that only genuine in-scene apertures stay open.
    # An outer wall borders the *connected exterior*, not every boundary of a
    # semantic polygon.  In particular, tiny inter-room annotation gaps are
    # not exterior and must not be converted into a closed wall strip.
    exterior_seed = np.zeros_like(indoor)
    exterior_seed[0, :] = exterior_seed[-1, :] = True
    exterior_seed[:, 0] = exterior_seed[:, -1] = True
    exterior = ndimage.binary_propagation(exterior_seed & ~indoor, mask=~indoor)
    outer_boundary = indoor & ndimage.binary_dilation(exterior)
    radius_cells = max(0, int(np.ceil((wall_thickness_m / 2) / raster.spacing_m)) - 1)
    if radius_cells:
        outer_boundary = ndimage.binary_dilation(outer_boundary, iterations=radius_cells) & indoor
    if mesh_wall_mask is not None:
        wall = np.asarray(mesh_wall_mask, dtype=bool).copy()
        if wall.shape != indoor.shape:
            raise ValueError("mesh_wall_mask shape must match raster")
        # Structural geometry remains authoritative inside the scene, while
        # the semantic exterior contour is intentionally closed.
        wall |= outer_boundary
    else:
        radius_cells = int(np.ceil((wall_thickness_m / 2) / raster.spacing_m))
        wall = ndimage.binary_dilation(boundary, iterations=radius_cells) if radius_cells else boundary
        # The semantic-only bootstrap has no stage slice yet.  Its derived
        # exterior contour must therefore become an explicit wall as well,
        # including a contour introduced by closing a tiny annotation seam.
        wall |= outer_boundary
    # User-approved contract: portal footprints are openings, never walls.
    # Exception: exterior door/window/glass candidates stay blocked because
    # this is an indoor-only floorplan. Only an internal portal can open a
    # structural wall.
    portal &= indoor & ~outer_boundary
    wall &= ~portal
    open_space = ~wall
    # "Exterior" is a candidate-domain concept, not visual reachability.
    # An indoor cell remains indoor even when a deSDF ray can leave through a
    # window or exterior door.
    exterior = ~indoor
    if mesh_wall_mask is None:
        # Without a stage mesh, semantic polygon boundaries are the fallback
        # structural walls and must form a closed indoor enclosure.
        border = np.zeros_like(open_space)
        border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
        nonportal_open_space = open_space & ~portal
        invalid_exterior = ndimage.binary_propagation(border & nonportal_open_space, mask=nonportal_open_space)
        leaked_indoor = indoor & invalid_exterior
        if leaked_indoor.any():
            raise ValueError(f"structural wall boundary leaks {int(leaked_indoor.sum())} indoor raster cells to exterior")
    # Candidate positions are restricted by semantic indoor support rather
    # than by visual reachability through a portal. This keeps outdoors out
    # of the candidate grid while leaving portals free in map.png.
    free = indoor & ~wall
    if not free.any():
        raise ValueError("structural wall raster left no free indoor cells")
    return FloorplanArtifacts(
        indoor_mask=indoor,
        outer_boundary_mask=outer_boundary,
        wall_mask=wall,
        portal_mask=portal,
        desdf_occluder_mask=wall,
        exterior_mask=exterior,
        free_mask=free,
        map_png=np.where(free, 255, 0).astype(np.uint8),
    )


def audit_map_connectivity(
    free_mask: np.ndarray,
    region_points_rc: dict[str, tuple[int, int]],
    navmesh_reachable: dict[tuple[str, str], bool],
) -> dict[str, object]:
    """Compare the map's free-space components with Habitat reachability.

    ``region_points_rc`` contains one verified representative free cell per
    semantic region.  A pair reachable in Habitat but split in ``map.png`` is
    a false closure (typically a missed doorway); the inverse is a false
    opening.  Four-neighbour connectivity deliberately avoids diagonal corner
    contact being treated as a passable doorway.
    """

    free = np.asarray(free_mask, dtype=bool)
    if free.ndim != 2 or not free.any():
        raise ValueError("free_mask must be a non-empty two-dimensional boolean grid")
    if len(region_points_rc) < 2:
        raise ValueError("at least two representative region points are required")
    labels, component_count = ndimage.label(
        free,
        structure=np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8),
    )
    components: dict[str, int] = {}
    for name, point in sorted(region_points_rc.items()):
        if len(point) != 2:
            raise ValueError(f"region {name!r} needs a (row, col) point")
        row, col = (int(point[0]), int(point[1]))
        if not (0 <= row < free.shape[0] and 0 <= col < free.shape[1]):
            raise ValueError(f"region {name!r} representative point is outside map bounds")
        component = int(labels[row, col])
        if component == 0:
            raise ValueError(f"region {name!r} representative point is blocked in map.png")
        components[name] = component

    pairs: list[dict[str, object]] = []
    for left, right in sorted((tuple(sorted(pair)) for pair in navmesh_reachable)):
        if left == right or left not in components or right not in components:
            raise ValueError("every navmesh pair must name two distinct representative regions")
        reachable = bool(navmesh_reachable.get((left, right), navmesh_reachable.get((right, left))))
        map_connected = components[left] == components[right]
        status = (
            "match" if reachable == map_connected else
            "false_closed" if reachable else "false_open"
        )
        pairs.append({
            "left_region": left,
            "right_region": right,
            "left_component": components[left],
            "right_component": components[right],
            "navmesh_reachable": reachable,
            "map_connected": map_connected,
            "status": status,
        })
    return {
        "component_count": int(component_count),
        "region_components": components,
        "pairs": pairs,
        "false_closed_pair_count": sum(pair["status"] == "false_closed" for pair in pairs),
        "false_open_pair_count": sum(pair["status"] == "false_open" for pair in pairs),
    }


def carve_internal_navmesh_path(
    wall_mask: np.ndarray,
    *,
    indoor_mask: np.ndarray,
    outer_boundary_mask: np.ndarray,
    raster: WorldRaster,
    path_xz_m: np.ndarray,
    corridor_radius_m: float,
) -> np.ndarray:
    """Open a corridor on an interior Habitat path without touching the exterior.

    This repair is intentionally narrow and is permitted only for a path
    already accepted by Habitat's structural navmesh.  The caller remains
    responsible for checking that the path stays within the semantic indoor
    footprint; this function enforces the raster-side invariant that no outer
    boundary cell can be cleared.
    """

    wall = np.asarray(wall_mask, dtype=bool)
    indoor = np.asarray(indoor_mask, dtype=bool)
    outer = np.asarray(outer_boundary_mask, dtype=bool)
    if wall.shape != (raster.height, raster.width) or indoor.shape != wall.shape or outer.shape != wall.shape:
        raise ValueError("wall, indoor, and outer-boundary masks must match raster shape")
    points = np.asarray(path_xz_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError("path_xz_m must have shape (N, 2), N >= 2")
    if corridor_radius_m < 0:
        raise ValueError("corridor_radius_m must be non-negative")
    path_cells = np.zeros_like(wall)
    for start, end in zip(points, points[1:]):
        length_m = float(np.linalg.norm(end - start))
        count = max(2, int(np.ceil(length_m / (raster.spacing_m / 2))) + 1)
        for x_m, z_m in np.linspace(start, end, count):
            try:
                row, col = raster.world_to_rc(float(x_m), float(z_m))
            except ValueError:
                continue
            path_cells[row, col] = True
    radius_cells = int(np.ceil(corridor_radius_m / raster.spacing_m))
    if radius_cells:
        path_cells = ndimage.binary_dilation(path_cells, iterations=radius_cells)
    clearable = path_cells & indoor & ~outer
    return wall & ~clearable


def build_directional_desdf(
    free_mask: np.ndarray, *, spacing_m: float, output_spacing_m: float | None = None,
    yaw_bins: int = 36, max_range_m: float = 10.0, occluder_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Raycast first explicit occluder from every free map cell in metres."""

    free = np.asarray(free_mask, dtype=bool)
    if free.ndim != 2 or not free.any():
        raise ValueError("free_mask must be a non-empty two-dimensional boolean grid")
    occluder = ~free if occluder_mask is None else np.asarray(occluder_mask, dtype=bool)
    if occluder.shape != free.shape:
        raise ValueError("occluder_mask shape must match free_mask")
    output_spacing_m = spacing_m if output_spacing_m is None else output_spacing_m
    if spacing_m <= 0 or output_spacing_m < spacing_m or yaw_bins <= 0 or max_range_m <= 0:
        raise ValueError("spacing, yaw_bins, and max_range must be positive")
    ratio = int(round(output_spacing_m / spacing_m))
    if not math.isclose(ratio * spacing_m, output_spacing_m, rel_tol=0, abs_tol=1e-9):
        raise ValueError("output_spacing_m must be an integer multiple of spacing_m")
    output = np.zeros((free.shape[0] // ratio, free.shape[1] // ratio, yaw_bins), dtype=np.float32)
    max_steps = int(np.ceil(max_range_m / spacing_m))
    angles = np.arange(yaw_bins, dtype=np.float64) * (2 * np.pi / yaw_bins)
    out_rows, out_cols = np.indices(output.shape[:2], dtype=np.int32)
    source_rows, source_cols = out_rows * ratio, out_cols * ratio
    source_is_free = free[source_rows, source_cols]
    for yaw_bin, angle in enumerate(angles):
        ray_values = np.full(output.shape[:2], max_range_m, dtype=np.float32)
        unresolved = source_is_free.copy()
        last_offset = (0, 0)
        for step in range(1, max_steps + 1):
            delta_row = int(np.rint(step * np.sin(angle)))
            delta_col = int(np.rint(step * np.cos(angle)))
            if (delta_row, delta_col) == last_offset:
                continue
            last_offset = (delta_row, delta_col)
            probe_rows = source_rows + delta_row
            probe_cols = source_cols + delta_col
            in_bounds = (
                (probe_rows >= 0) & (probe_rows < free.shape[0])
                & (probe_cols >= 0) & (probe_cols < free.shape[1])
            )
            hit = np.zeros_like(unresolved)
            hit[in_bounds] = unresolved[in_bounds] & occluder[probe_rows[in_bounds], probe_cols[in_bounds]]
            if hit.any():
                ray_values[hit] = min(float(np.hypot(delta_row, delta_col)) * spacing_m, max_range_m)
                unresolved[hit] = False
            if not unresolved.any():
                break
        output[..., yaw_bin] = np.where(source_is_free, ray_values, 0.0)
    return output


def world_raster_for_regions(
    regions: list[dict[str, object]], *, spacing_m: float, padding_m: float = 0.30,
) -> WorldRaster:
    """Fit an outward-rounded metric raster around semantic room polygons."""

    if spacing_m <= 0 or padding_m < 0:
        raise ValueError("spacing_m must be positive and padding_m non-negative")
    regions = indoor_region_annotations(regions)
    points: list[np.ndarray] = []
    for region in regions:
        loop = np.asarray(region.get("poly_loop", []), dtype=np.float64)
        if loop.ndim != 2 or loop.shape[1] < 3:
            raise ValueError("region annotation is missing an X/Y/Z poly_loop")
        points.append(loop[:, [0, 2]])
    if not points:
        raise ValueError("at least one region is required")
    stacked = np.concatenate(points, axis=0)
    minimum = stacked.min(axis=0) - padding_m
    maximum = stacked.max(axis=0) + padding_m
    return WorldRaster(
        x_min_m=math.floor(minimum[0] / spacing_m) * spacing_m,
        x_max_m=math.ceil(maximum[0] / spacing_m) * spacing_m,
        z_min_m=math.floor(minimum[1] / spacing_m) * spacing_m,
        z_max_m=math.ceil(maximum[1] / spacing_m) * spacing_m,
        spacing_m=spacing_m,
    )


def _sha256_if_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_floorplan_artifacts(
    output: Path, regions: list[dict[str, object]], raster: WorldRaster, *, scene_id: str, semantic_path: Path,
    wall_thickness_m: float = 0.10, desdf_spacing_m: float = 0.10,
    mesh_wall_mask: np.ndarray | None = None, portal_mask: np.ndarray | None = None,
) -> dict[str, object]:
    """Write canonical map, QA masks, and metric deSDF with provenance."""

    output = Path(output)
    if desdf_spacing_m < raster.spacing_m:
        desdf_spacing_m = raster.spacing_m
    artifacts = build_structural_floorplan(
        regions, raster, wall_thickness_m=wall_thickness_m, mesh_wall_mask=mesh_wall_mask,
        portal_mask=portal_mask,
    )
    desdf = build_directional_desdf(
        artifacts.free_mask, spacing_m=raster.spacing_m, output_spacing_m=desdf_spacing_m,
        # Do not interpret semantic candidate-domain exclusions (notably
        # exterior cells reached through a doorway) as physical wall hits.
        occluder_mask=artifacts.desdf_occluder_mask,
    )
    output.mkdir(parents=True, exist_ok=True)
    Image.fromarray(artifacts.map_png).save(output / "map.png")
    Image.fromarray(np.where(artifacts.indoor_mask, 255, 0).astype(np.uint8)).save(output / "indoor_mask.png")
    Image.fromarray(np.where(artifacts.outer_boundary_mask, 255, 0).astype(np.uint8)).save(
        output / "outer_boundary_mask.png"
    )
    Image.fromarray(np.where(artifacts.wall_mask, 255, 0).astype(np.uint8)).save(output / "wall_mask.png")
    Image.fromarray(np.where(artifacts.portal_mask, 255, 0).astype(np.uint8)).save(output / "portal_mask.png")
    Image.fromarray(np.where(artifacts.desdf_occluder_mask, 255, 0).astype(np.uint8)).save(
        output / "desdf_occluder_mask.png"
    )
    np.save(output / "desdf.npy", {
        "l": 0,
        "t": 0,
        "desdf": desdf,
        "map_spacing_m": raster.spacing_m,
        "desdf_spacing_m": desdf_spacing_m,
        "origin_x_m": raster.x_min_m,
        "origin_z_min_m": raster.z_min_m,
        "origin_z_max_m": raster.z_max_m,
        "yaw_bins": int(desdf.shape[-1]),
        "max_range_m": 10.0,
    })
    metadata: dict[str, object] = {
        "schema_version": 1,
        "scene_id": scene_id,
        "semantic_path": str(semantic_path),
        "semantic_sha256": _sha256_if_file(Path(semantic_path)),
        "world_bounds_m": {
            "x_min": raster.x_min_m, "x_max": raster.x_max_m,
            "z_min": raster.z_min_m, "z_max": raster.z_max_m,
        },
        "map_spacing_m": raster.spacing_m,
        "desdf_spacing_m": desdf_spacing_m,
        "map_png_polarity": "255=free,0=blocked",
        "world_to_pixel": "col=(x-x_min)/spacing; row=(z-z_min)/spacing",
        "raster_view": "+Y top-down: +X right, +Z down",
        "wall_thickness_m": wall_thickness_m,
        "semantic_gap_closure_m": SEMANTIC_GAP_CLOSURE_M,
        "exterior_definition": "raster_edge_connected_complement_of_closed_semantic_indoor_support",
        "wall_source": "stage_mesh_slice" if mesh_wall_mask is not None else "semantic_region_boundaries",
        "portal_policy": "door_window_glass=free_non_occluding",
        "desdf_occluder_source": "non_portal_walls",
    }
    (output / "floorplan_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return metadata


def sample_poses(
    free_mask: np.ndarray, raster: WorldRaster, *, count: int = 200, min_clearance_m: float = 0.10, seed: int,
) -> np.ndarray:
    """Sample metric-grid poses with coverage and a bounded near-wall stratum."""

    free = np.asarray(free_mask, dtype=bool)
    if free.shape != (raster.height, raster.width):
        raise ValueError("free_mask shape must match raster")
    if count <= 0 or min_clearance_m < 0:
        raise ValueError("count must be positive and min_clearance_m non-negative")
    clearance_m = ndimage.distance_transform_edt(free) * raster.spacing_m
    valid = free & (clearance_m + 1e-9 >= min_clearance_m)
    full_candidates = np.argwhere(valid)
    proposal_stride = max(1, int(math.ceil(0.10 / raster.spacing_m)))
    proposal_mask = valid.copy()
    proposal_mask &= (np.indices(valid.shape)[0] % proposal_stride == 0)
    proposal_mask &= (np.indices(valid.shape)[1] % proposal_stride == 0)
    candidates = np.argwhere(proposal_mask)
    if len(candidates) < count:
        candidates = full_candidates
    if len(candidates) < count:
        raise ValueError(f"only {len(candidates)} navigable cells meet clearance; need {count}")
    rng = np.random.default_rng(seed)
    near_wall = candidates[clearance_m[candidates[:, 0], candidates[:, 1]] < 0.50]
    near_count = min(count // 4, len(near_wall))
    selected: list[np.ndarray] = []
    if near_count:
        selected.extend(near_wall[index] for index in rng.choice(len(near_wall), size=near_count, replace=False))
    selected_array = np.asarray(selected, dtype=np.int32).reshape(-1, 2)
    selected_keys = {tuple(row) for row in selected_array.tolist()}
    remaining = np.asarray([row for row in candidates if tuple(row) not in selected_keys], dtype=np.int32)
    if not len(selected_array):
        first = int(rng.integers(len(remaining)))
        selected_array = remaining[[first]]
        remaining = np.delete(remaining, first, axis=0)
    while len(selected_array) < count:
        if not len(remaining):
            raise AssertionError("pose sampling exhausted valid candidates")
        deltas = remaining[:, None, :] - selected_array[None, :, :]
        nearest_sq = np.min(np.sum(deltas * deltas, axis=2), axis=1)
        choice = int(np.argmax(nearest_sq))
        selected_array = np.vstack((selected_array, remaining[choice]))
        remaining = np.delete(remaining, choice, axis=0)
    return selected_array.astype(np.int32, copy=False)


def build_capture_schedule(
    poses_rc: np.ndarray, raster: WorldRaster, *, floor_y_m: float,
    yaws_deg: tuple[int, ...] = tuple(range(0, 360, 45)), camera_height_m: float = 1.5,
) -> list[dict[str, object]]:
    """Expand planar pose cells into one explicit RGB/depth record per yaw."""

    poses = np.asarray(poses_rc, dtype=np.int32)
    if poses.ndim != 2 or poses.shape[1] != 2:
        raise ValueError("poses_rc must have shape (N, 2)")
    if not yaws_deg or len(set(yaws_deg)) != len(yaws_deg):
        raise ValueError("yaws_deg must be non-empty and unique")
    camera_width_px, camera_height_px, camera_hfov_deg = 640, 480, 106.2602
    camera_vfov_deg = math.degrees(2 * math.atan(
        math.tan(math.radians(camera_hfov_deg) / 2) * camera_height_px / camera_width_px
    ))
    records: list[dict[str, object]] = []
    for pose_index, (row, col) in enumerate(poses):
        x_m, z_m = raster.rc_to_world(int(row), int(col))
        for yaw_index, yaw_deg in enumerate(yaws_deg):
            sample_index = pose_index * len(yaws_deg) + yaw_index
            records.append({
                "sample_index": sample_index,
                "pose_index": pose_index,
                "grid_row": int(row),
                "grid_col": int(col),
                "floor_xyz_m": [float(x_m), float(floor_y_m), float(z_m)],
                "pose_xyz_m": [float(x_m), float(floor_y_m + camera_height_m), float(z_m)],
                "logical_yaw_deg": int(yaw_deg),
                # Habitat identity views local −Z.  Logical yaw is measured
                # clockwise in the stored +X-right/+Z-down map frame, hence
                # its sign reverses before the −90° camera-forward offset.
                "habitat_yaw_deg": float(-yaw_deg - 90),
                "camera_height_above_floor_m": float(camera_height_m),
                "image": f"rgb/{sample_index:05d}.png",
                "depth": f"depth/{sample_index:05d}.npz",
                "camera_width_px": camera_width_px,
                "camera_height_px": camera_height_px,
                "camera_hfov_deg": camera_hfov_deg,
                "camera_vfov_deg": camera_vfov_deg,
                "depth_representation": "camera_z_depth_m",
            })
    return records


def navmesh_snap_metrics(
    poses_xyz_m: np.ndarray, snap_point: object, *, horizontal_tolerance_m: float,
) -> dict[str, object]:
    """Measure whether metric capture positions agree with a Habitat navmesh.

    ``snap_point`` deliberately stays dependency-injected so this shared module
    remains importable without Habitat-Sim.  The caller supplies
    ``sim.pathfinder.snap_point`` (with any required vector conversion).
    """

    poses = np.asarray(poses_xyz_m, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 3 or not len(poses):
        raise ValueError("poses_xyz_m must have non-empty shape (N, 3)")
    if horizontal_tolerance_m < 0:
        raise ValueError("horizontal_tolerance_m must be non-negative")
    if not callable(snap_point):
        raise TypeError("snap_point must be callable")
    distances: list[float] = []
    for pose in poses:
        snapped = np.asarray(snap_point(pose), dtype=np.float64)
        if snapped.shape != (3,) or not np.isfinite(snapped).all():
            distances.append(float("inf"))
            continue
        distances.append(float(np.linalg.norm((snapped - pose)[[0, 2]])))
    distance_array = np.asarray(distances, dtype=np.float64)
    within = distance_array <= horizontal_tolerance_m
    return {
        "pose_count": int(len(poses)),
        "horizontal_tolerance_m": float(horizontal_tolerance_m),
        "within_tolerance_count": int(within.sum()),
        "all_within_tolerance": bool(within.all()),
        "mean_horizontal_snap_distance_m": float(distance_array.mean()),
        "max_horizontal_snap_distance_m": float(distance_array.max()),
        "horizontal_snap_distances_m": distances,
    }

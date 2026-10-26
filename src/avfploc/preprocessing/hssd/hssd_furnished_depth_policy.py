#!/usr/bin/env python3
"""Keep natural visual ambiguity; reject reliable furnished depth no-hits."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .export_hssd_unloc_frames import unloc_feature_column_bearings_deg

AUDIT_DIRECTORY = "furnished_depth_qa_v2_nohit_only"


@dataclass(frozen=True)
class FurnishedAuditPolicy:
    version: str = "hssd_furnished_nohit_only_v4"
    absolute_tolerance_m: float = 0.50
    relative_tolerance: float = 0.20
    bad_contiguous_rays: int = 3
    bad_total_rays: int = 4
    short_no_hit_target_lt_m: float = 8.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _tolerance(left: np.ndarray, right: np.ndarray, policy: FurnishedAuditPolicy) -> np.ndarray:
    return np.maximum(policy.absolute_tolerance_m, policy.relative_tolerance * np.minimum(left, right))


def _longest_run(mask: np.ndarray) -> int:
    longest = current = 0
    for flagged in mask:
        current = current + 1 if bool(flagged) else 0
        longest = max(longest, current)
    return longest


def _bad_sector(mask: np.ndarray, policy: FurnishedAuditPolicy) -> bool:
    return bool(_longest_run(mask) >= policy.bad_contiguous_rays or np.count_nonzero(mask) >= policy.bad_total_rays)


def classify_furnished_view(
    furnished40: np.ndarray, ambiguous40: np.ndarray, target40: np.ndarray,
    exact_map40: np.ndarray, stage40: np.ndarray | None, scope40: np.ndarray,
    policy: FurnishedAuditPolicy, *, stage_ambiguous40: np.ndarray | None = None,
) -> dict[str, object]:
    """Only reliable no-hits gate valid views; depth differences are diagnostics.

    An ambiguous feature has no stable depth estimate and is explicitly allowed.
    A nonambiguous no-hit means both sampled sensor pixels missed geometry. It
    is excluded even when the map depth reached the legacy 10 m range cap.
    """

    furnished = np.asarray(furnished40, dtype=np.float32)
    target = np.asarray(target40, dtype=np.float32)
    exact = np.asarray(exact_map40, dtype=np.float32)
    ambiguous = np.asarray(ambiguous40, dtype=bool)
    scope = np.asarray(scope40, dtype=np.uint8)
    arrays = (furnished, target, exact, ambiguous, scope)
    if any(a.shape != (40,) for a in arrays):
        return {"decision": "fail", "reasons": ["invalid_shape"], "counts": {}, "ray_classes": []}
    if not np.isfinite(target).all() or not np.isfinite(exact).all() or np.any(target <= 0) or np.any(exact <= 0):
        return {"decision": "fail", "reasons": ["invalid_target"], "counts": {}, "ray_classes": []}
    stage = None if stage40 is None else np.asarray(stage40, dtype=np.float32)
    if stage is not None and stage.shape != (40,):
        return {"decision": "fail", "reasons": ["invalid_stage_shape"], "counts": {}, "ray_classes": []}
    stage_ambiguous = np.zeros(40, dtype=bool) if stage_ambiguous40 is None else np.asarray(stage_ambiguous40, dtype=bool)
    if stage_ambiguous.shape != (40,):
        return {"decision": "fail", "reasons": ["invalid_stage_ambiguity_shape"], "counts": {}, "ray_classes": []}

    classes = np.full(40, "matched", dtype=object)
    classes[ambiguous] = "ambiguous"
    target_disagreement = (~ambiguous) & (np.abs(target - exact) > _tolerance(target, exact, policy))
    bearing = np.deg2rad(unloc_feature_column_bearings_deg())
    map_cap_forward = 10.0 * np.cos(bearing)
    map_capped = exact >= map_cap_forward - 0.05

    hit = np.isfinite(furnished) & (furnished > 0) & ~ambiguous
    no_hit = (~np.isfinite(furnished) | (furnished <= 0)) & ~ambiguous
    short_no_hit = no_hit & ~map_capped & (exact < policy.short_no_hit_target_lt_m)
    classes[short_no_hit] = "unexplained_no_hit"
    classes[no_hit & ~short_no_hit] = "no_hit_at_cap"
    false_closed = hit & ~map_capped & ((furnished - exact) > _tolerance(furnished, exact, policy))
    classes[false_closed] = "false_closed"
    short_observation = hit & ((exact - furnished) > _tolerance(furnished, exact, policy))
    if stage is None:
        classes[short_observation] = "short_unresolved"
    else:
        stage_hit = np.isfinite(stage) & (stage > 0) & ~stage_ambiguous
        stage_matches_furnished = stage_hit & (np.abs(stage - furnished) <= _tolerance(stage, furnished, policy))
        structural_false_open = short_observation & stage_matches_furnished
        furniture = short_observation & ~structural_false_open & ~stage_ambiguous & (
            ~stage_hit | (np.abs(stage - exact) <= _tolerance(stage, exact, policy))
        )
        classes[short_observation] = "short_unresolved"
        classes[furniture] = "furniture_occlusion"
        classes[structural_false_open] = "false_open"
    classes[target_disagreement & (classes == "matched")] = "target_disagreement"

    counts = {name: int(np.count_nonzero(classes == name)) for name in (
        "matched", "ambiguous", "target_disagreement", "unexplained_no_hit", "no_hit_at_cap",
        "false_closed", "false_open", "furniture_occlusion", "short_unresolved",
    )}
    counts["target_disagreement"] = int(np.count_nonzero(target_disagreement))
    counts["exterior_opening_first_hits"] = int(np.count_nonzero(scope == 2))
    counts["reliable_no_hit"] = int(np.count_nonzero(no_hit))
    diagnostic_sectors: list[str] = []
    for label, reason in (
        ("unexplained_no_hit", "unexplained_no_hit_sector"),
        ("false_closed", "false_closed_sector"),
        ("false_open", "false_open_sector"),
    ):
        if _bad_sector(classes == label, policy):
            diagnostic_sectors.append(reason)
    warnings = [name for name in (
        "ambiguous", "false_closed", "false_open", "short_unresolved", "target_disagreement",
    ) if counts[name]]
    return {
        "decision": "fail" if no_hit.any() else "pass",
        "reasons": ["furnished_no_hit"] if no_hit.any() else ["no_reliable_no_hit"],
        "warning_reasons": warnings, "diagnostic_sectors": diagnostic_sectors,
        "counts": counts, "ray_classes": classes.tolist(),
    }

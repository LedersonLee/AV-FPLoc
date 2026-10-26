#!/usr/bin/env python3
"""Deterministic stage-vs-floorplan consistency policy for HSSD views."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class AuditPolicy:
    version: str = "hssd_stage_map_v1"
    max_range_m: float = 10.0
    no_hit_view_fraction: float = 0.10
    false_closed_target_lt_m: float = 8.0
    hit_abs_error_m: float = 1.0
    hit_rel_error: float = 0.25
    contiguous_mismatch_rays: int = 3
    total_mismatch_rays: int = 4

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _longest_run(mask: np.ndarray) -> int:
    longest = current = 0
    for value in mask:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def classify_view(
    stage_depth40: np.ndarray, ambiguous40: np.ndarray, target_depth40: np.ndarray,
    policy: AuditPolicy,
) -> dict[str, object]:
    """Classify a captured view without any learned-model score.

    Ambiguous half-pixel discontinuities are review evidence, never confident
    hit/no-hit evidence. Stage no-hit is NaN, not zero metres.
    """

    stage = np.asarray(stage_depth40, dtype=np.float32)
    target = np.asarray(target_depth40, dtype=np.float32)
    ambiguous = np.asarray(ambiguous40, dtype=bool)
    if any(a.shape != (40,) for a in (stage, target, ambiguous)):
        return {"decision": "fail", "reasons": ["invalid_shape"], "counts": {}, "ray_classes": []}
    if not np.isfinite(target).all() or np.any(target <= 0):
        return {"decision": "fail", "reasons": ["invalid_target"], "counts": {}, "ray_classes": []}
    hit = np.isfinite(stage) & (stage > 0) & ~ambiguous
    no_hit = (~np.isfinite(stage) | (stage <= 0)) & ~ambiguous
    threshold = np.maximum(policy.hit_abs_error_m, policy.hit_rel_error * np.maximum(stage, 0))
    closed_hit = hit & (stage - target > threshold)
    open_hit = hit & (target - stage > threshold)
    closed_no_hit = no_hit & (target < policy.false_closed_target_lt_m)
    mismatch = closed_hit | open_hit
    classes = np.full(40, "matched", dtype=object)
    classes[ambiguous] = "ambiguous"
    classes[no_hit] = "no_hit_at_cap"
    classes[closed_no_hit] = "false_closed_no_hit"
    classes[closed_hit] = "false_closed_hit"
    classes[open_hit] = "false_open_hit"
    counts = {name: int(np.count_nonzero(classes == name)) for name in (
        "matched", "ambiguous", "no_hit_at_cap", "false_closed_no_hit",
        "false_closed_hit", "false_open_hit",
    )}
    counts["no_hit"] = int(np.count_nonzero(no_hit))
    counts["hit_mismatch"] = int(np.count_nonzero(mismatch))
    counts["longest_hit_mismatch_run"] = max(_longest_run(closed_hit), _longest_run(open_hit))
    counts["longest_no_hit_run"] = _longest_run(no_hit)
    reasons: list[str] = []
    if counts["no_hit"] / 40 >= policy.no_hit_view_fraction:
        reasons.append("no_hit_sector")
    if counts["longest_hit_mismatch_run"] >= policy.contiguous_mismatch_rays:
        reasons.append("contiguous_hit_mismatch")
    if counts["hit_mismatch"] >= policy.total_mismatch_rays:
        reasons.append("many_hit_mismatches")
    if reasons:
        decision = "fail"
    elif counts["no_hit"] or counts["hit_mismatch"] or counts["ambiguous"]:
        decision = "review"
        reasons.append("isolated_or_ambiguous_discrepancy")
    else:
        decision = "pass"
        reasons.append("consistent")
    return {"decision": decision, "reasons": reasons, "counts": counts, "ray_classes": classes.tolist()}

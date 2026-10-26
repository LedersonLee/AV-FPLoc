"""Asymmetric furnished-observation versus structural-map decisions."""

from __future__ import annotations

import importlib
from pathlib import Path
import sys

import numpy as np



def _classify(furnished, target, *, stage=None, exact=None, ambiguous=None, scope=None):
    module = importlib.import_module("avfploc.preprocessing.hssd.hssd_furnished_depth_policy")
    furnished = np.asarray(furnished, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    return module.classify_furnished_view(
        furnished, np.zeros(40, dtype=bool) if ambiguous is None else ambiguous,
        target, target if exact is None else exact,
        None if stage is None else np.asarray(stage, dtype=np.float32),
        np.zeros(40, dtype=np.uint8) if scope is None else scope,
        module.FurnishedAuditPolicy(),
    )


def test_close_furnished_door_is_not_false_closed():
    result = _classify(np.full(40, 0.35), np.full(40, 0.23))
    assert result["decision"] == "pass"


def test_furniture_nearer_than_structural_map_is_retained():
    result = _classify(np.full(40, 2.0), np.full(40, 6.0), stage=np.full(40, 6.0))
    assert result["decision"] == "pass"
    assert result["counts"]["furniture_occlusion"] == 40


def test_false_open_is_a_diagnostic_not_an_exclusion():
    furnished = np.full(40, 6.0)
    furnished[10:13] = 2.0
    stage = np.full(40, 6.0)
    stage[10:13] = 2.0
    result = _classify(furnished, np.full(40, 6.0), stage=stage)
    assert result["decision"] == "pass"
    assert "false_open_sector" in result["diagnostic_sectors"]


def test_false_closed_at_exterior_door_is_retained_when_depth_hits():
    furnished = np.full(40, 1.0)
    furnished[10:13] = 6.0
    scope = np.zeros(40, dtype=np.uint8)
    scope[10:13] = 2
    result = _classify(furnished, np.full(40, 1.0), scope=scope)
    assert result["decision"] == "pass"
    assert "false_closed_sector" in result["diagnostic_sectors"]


def test_four_no_hits_against_short_map_depth_fail():
    furnished = np.full(40, 3.0)
    furnished[5:9] = np.nan
    result = _classify(furnished, np.full(40, 3.0))
    assert result["decision"] == "fail"
    assert "furnished_no_hit" in result["reasons"]


def test_isolated_discontinuity_is_retained():
    ambiguous = np.zeros(40, dtype=bool)
    ambiguous[1] = True
    result = _classify(np.full(40, 3.0), np.full(40, 3.0), ambiguous=ambiguous)
    assert result["decision"] == "pass"


def test_one_reliable_no_hit_is_excluded_even_at_the_map_cap():
    furnished = np.full(40, 15.0)
    furnished[0] = np.nan
    result = _classify(furnished, np.full(40, 10.0))
    assert result["decision"] == "fail"
    assert result["counts"]["reliable_no_hit"] == 1


def test_ambiguous_missing_value_is_not_a_reliable_no_hit():
    furnished = np.full(40, 3.0)
    furnished[3] = np.nan
    ambiguous = np.zeros(40, dtype=bool)
    ambiguous[3] = True
    result = _classify(furnished, np.full(40, 3.0), ambiguous=ambiguous)
    assert result["decision"] == "pass"
    assert result["counts"]["reliable_no_hit"] == 0


def test_unresolved_short_depth_is_retained_without_stage_evidence():
    result = _classify(np.full(40, 2.0), np.full(40, 6.0))
    assert result["decision"] == "pass"
    assert result["counts"]["short_unresolved"] == 40


def test_quantized_target_disagreement_is_diagnostic_when_exact_map_matches_rgb():
    exact = np.full(40, 3.0)
    exact[10:13] = 6.0
    furnished = exact.copy()
    result = _classify(furnished, np.full(40, 3.0), exact=exact)
    assert result["decision"] == "pass"
    assert result["counts"]["target_disagreement"] == 3


def test_map_range_cap_is_not_a_false_closed_wall():
    result = _classify(np.full(40, 15.0), np.full(40, 10.0), exact=np.full(40, 10.0))
    assert result["decision"] == "pass"
    assert result["counts"]["false_closed"] == 0

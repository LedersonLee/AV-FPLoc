"""Conservative, model-independent view classification."""

from pathlib import Path
import sys

import numpy as np

from avfploc.preprocessing.hssd.hssd_geometry_audit_policy import AuditPolicy, classify_view


def check(stage, target, ambiguous=None):
    return classify_view(np.asarray(stage, dtype=np.float32),
                         np.zeros(40, dtype=bool) if ambiguous is None else ambiguous,
                         np.asarray(target, dtype=np.float32), AuditPolicy())


def test_equal_hits_pass():
    r = check(np.full(40, 5), np.full(40, 5.2))
    assert r["decision"] == "pass"
    assert r["counts"]["matched"] == 40


def test_false_closed_nohit_short_target_fails():
    stage = np.full(40, 5.)
    stage[10:14] = np.nan
    r = check(stage, np.full(40, 1.))
    assert r["decision"] == "fail"
    assert "no_hit_sector" in r["reasons"]
    assert r["counts"]["false_closed_no_hit"] == 4


def test_nohit_near_cap_is_not_false_closed_but_still_excluded():
    stage = np.full(40, 5.)
    stage[10:14] = np.nan
    r = check(stage, np.full(40, 9.9))
    assert r["counts"]["false_closed_no_hit"] == 0
    assert "no_hit_sector" in r["reasons"]


def test_contiguous_false_open_hit_fails():
    target = np.full(40, 5.)
    target[11:14] = 8
    r = check(np.full(40, 5.), target)
    assert r["decision"] == "fail"
    assert "contiguous_hit_mismatch" in r["reasons"]
    assert r["counts"]["false_open_hit"] == 3


def test_single_hit_mismatch_is_review_not_fail():
    target = np.full(40, 5.)
    target[11] = 8
    r = check(np.full(40, 5.), target)
    assert r["decision"] == "review"


def test_ambiguous_edge_does_not_become_false_closed():
    stage = np.full(40, 5.)
    stage[10] = np.nan
    amb = np.zeros(40, dtype=bool)
    amb[10] = True
    r = check(stage, np.full(40, 5.), amb)
    assert r["decision"] == "review"
    assert r["counts"]["false_closed_no_hit"] == 0


def test_bad_target_and_shapes_fail_closed():
    target = np.full(40, 5.)
    target[4] = np.nan
    assert check(np.full(40, 5.), target)["decision"] == "fail"
    assert check(np.full(39, 5.), np.full(40, 5.))["decision"] == "fail"

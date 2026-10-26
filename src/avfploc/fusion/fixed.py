"""Fuse higher-is-better scores, not raw distances or calibrated probabilities."""
import numpy as np
from avfploc.evaluation.baselines import standardized


def fuse(vision, acoustic, weight=0.5, normalized=False):
    v, a = np.asarray(vision, dtype=np.float64), np.asarray(acoustic, dtype=np.float64)
    if v.shape != a.shape or v.ndim != 2 or v.shape[1] != 36 or not v.shape[0]:
        raise ValueError("Expected matching nonempty [XY,36] score banks")
    if not np.isfinite(v).all() or not np.isfinite(a).all():
        raise ValueError("Nonfinite score bank")
    if not np.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("Acoustic weight must be in [0,1]")
    if not normalized:
        v, a = standardized(v), standardized(a)
    return (1 - weight) * v + weight * a


def decode(score, xy):
    xy = np.asarray(xy)
    if xy.shape != (score.shape[0], 2) or not np.isfinite(xy).all():
        raise ValueError("Candidate coordinates do not match scores")
    cell, yaw = np.unravel_index(np.argmax(score), score.shape)
    return np.array([*xy[cell], yaw * 10.0])

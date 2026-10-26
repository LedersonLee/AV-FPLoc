"""Portable score-bank contract. No pickle, implicit reordering or GT injection."""
from pathlib import Path
import numpy as np


def load(path):
    with np.load(Path(path), allow_pickle=False) as f:
        out = {k: f[k] for k in f.files}
    for key in ("scores", "xy", "query_ids", "yaw_deg"):
        if key not in out:
            raise ValueError(f"Missing bank field: {key}")
    scores, xy = out["scores"], out["xy"]
    if scores.ndim != 3 or scores.shape[1:] != (len(xy),36) or xy.shape != (len(xy),2):
        raise ValueError("Expected scores[query,XY,36] and xy[XY,2]")
    if not np.array_equal(out["yaw_deg"], np.arange(36)*10):
        raise ValueError("Expected world yaw 0:10:350")
    if len(out["query_ids"]) != len(scores) or len(set(out["query_ids"].tolist())) != len(scores):
        raise ValueError("Invalid/duplicate query IDs")
    if not np.isfinite(scores).all() or not np.isfinite(xy).all():
        raise ValueError("Nonfinite bank")
    return out


def aligned(vision, acoustic):
    for key in ("xy", "query_ids", "yaw_deg"):
        if not np.array_equal(vision[key], acoustic[key]):
            raise ValueError(f"Vision/acoustic {key} mismatch; explicit export alignment required")
    if vision["scores"].shape != acoustic["scores"].shape:
        raise ValueError("Score shape mismatch")

"""Inference-only adaptive gate. Load only trusted checkpoints."""
import numpy as np
from avfploc.evaluation.baselines import standardized
from .features import extract
from .fixed import fuse


def load_gate(checkpoint, device="cpu"):
    import torch
    from .model import Gate
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    if not all(k in state for k in ("model", "mean", "std", "feature_names")):
        raise ValueError("Checkpoint must carry model, Train mean/std and feature_names")
    names = list(state["feature_names"])
    hidden = state.get("hidden")
    constant = "logit" in state["model"]
    if hidden is None and not constant:
        hidden = [state["model"]["net.0.weight"].shape[0], state["model"]["net.2.weight"].shape[0]]
    model = Gate(len(names), constant=constant, hidden=hidden or (32,16)).to(device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    mean, std = np.asarray(state["mean"]), np.asarray(state["std"])
    if mean.shape != (len(names),) or std.shape != mean.shape or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError("Invalid Train normalization")
    return model, mean, std, names


def predict(vision, acoustic, xy, depth, gate, schema, ray_bank=None):
    import torch
    model, mean, std, names = gate
    v, a = standardized(vision), standardized(acoustic)
    x, expected = extract(v, a, xy, depth, schema, ray_bank)
    if names != expected:
        raise ValueError("Checkpoint feature names/order differ from requested schema")
    with torch.inference_mode():
        device = next(model.parameters()).device
        weight = float(model(torch.as_tensor((x-mean)/std, dtype=torch.float32, device=device)[None]).item())
    return fuse(v, a, weight, normalized=True), weight

"""Encode prepared physical-heading features and score aligned pose candidates."""
import numpy as np


def encode(checkpoint, energy, coherence, envelope=None, device="cuda", batch=256):
    import torch
    from .model import AcousticEncoder, requires_envelope
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = AcousticEncoder(**state["config"]["model"]).to(device).eval()
    model.load_state_dict(state["model"], strict=True)
    if len(energy) != len(coherence) or not len(energy):
        raise ValueError("Feature record counts differ or empty")
    if requires_envelope(model.mode) and (envelope is None or len(envelope) != len(energy)):
        raise ValueError("Envelope stream required and must match records")
    result=[]
    with torch.inference_mode():
        for start in range(0,len(energy),batch):
            values=[torch.as_tensor(np.array(x[start:start+batch], dtype=np.float32), device=device) for x in (energy,coherence)]
            if envelope is not None:
                values.append(torch.as_tensor(np.array(envelope[start:start+batch], dtype=np.float16).astype(np.float32), device=device))
            result.append(model(*values).cpu().numpy())
    return np.concatenate(result)


def score(observed, candidates):
    """observed[Q,D], candidates[XY,36,D], cosine of normalized embeddings."""
    o,c = np.asarray(observed,dtype=np.float64),np.asarray(candidates,dtype=np.float64)
    if o.ndim!=2 or c.ndim!=3 or c.shape[1]!=36 or o.shape[-1]!=c.shape[-1]:
        raise ValueError("Embedding shapes do not match")
    for x in (o,c):
        if not np.isfinite(x).all() or not np.allclose(np.linalg.norm(x,axis=-1),1,atol=1e-4):
            raise ValueError("Expected finite unit-normalized embeddings")
    return np.einsum("qd,cyd->qcy",o,c)

"""Inference preprocessing accepts body-aligned observations, never world pose."""
import numpy as np
import torch
from torch.nn import functional as F
from track_b.models.structure import rir_spectrum

def audio_features(body_rir, sample_rate_hz=48000):
    if sample_rate_hz!=48000:raise ValueError('Expected calibrated48kHz observation')
    r=np.asarray(body_rir)
    if r.ndim!=2 or r.shape[0]!=6 or not np.isfinite(r).all():
        raise ValueError('Expected finite six-channel body-frame waveform')
    audio=np.zeros((6,16384),np.float32);n=min(r.shape[-1],16384);audio[:,:n]=r[:,:n]
    audio/=max(float(np.sqrt(np.mean(audio**2))),1e-8)
    return rir_spectrum(torch.from_numpy(audio)[None]).half()

@torch.no_grad()
def visual_features(backbone, decoded_bgr):
    image,_=backbone.image2tensor(decoded_bgr)
    z=backbone.pretrained.get_intermediate_layers(image,reshape=True)[0]
    return F.adaptive_avg_pool2d(z,(4,16)).half()

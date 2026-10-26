"""Shared observed/FP encoder; checkpoint parameter names preserved."""
import torch
from torch import nn
from torch.nn import functional as F

FEATURE_STREAMS = dict(energy=('energy',), coherence=('coherence',),
    envelope=('envelope',), both=('energy','coherence'),
    energy_envelope=('energy','envelope'), coherence_envelope=('coherence','envelope'),
    all=('energy','coherence','envelope'))


def requires_envelope(features):
    return 'envelope' in FEATURE_STREAMS[features]


class AcousticEncoder(nn.Module):
    """One weight-shared encoder for both domains; no mic-order invariance.

    GroupNorm avoids per-domain running statistics. Coarse time bins are kept
    in the energy branch (not globally pooled away). No learned yaw prior.
    """
    def __init__(self, features='both', width=32, embedding_dim=128):
        super().__init__()
        if features not in FEATURE_STREAMS or width < 8 or width % 8:
            raise ValueError('Invalid architecture/width')
        self.mode = features
        streams = FEATURE_STREAMS[features]
        self.energy = nn.Sequential(
            nn.Conv2d(6, width, 3, padding=1), nn.GroupNorm(8, width), nn.GELU(),
            nn.Conv2d(width, width*2, 3, stride=2, padding=1), nn.GroupNorm(8, width*2), nn.GELU(),
            nn.Conv2d(width*2, width*3, 3, stride=2, padding=1), nn.GroupNorm(8, width*3), nn.GELU(),
            nn.AdaptiveAvgPool2d((2, 16)), nn.Flatten(), nn.Linear(width*3*32, embedding_dim), nn.GELU()
        ) if 'energy' in streams else None
        self.coherence = nn.Sequential(
            nn.Conv1d(24, width, 3, padding=1), nn.GroupNorm(8, width), nn.GELU(),
            nn.Conv1d(width, width*2, 3, padding=1), nn.GroupNorm(8, width*2), nn.GELU(),
            nn.AdaptiveAvgPool1d(4), nn.Flatten(), nn.Linear(width*2*4, embedding_dim), nn.GELU()
        ) if 'coherence' in streams else None
        self.envelope = nn.Sequential(
            nn.Conv1d(6, width, 5, padding=2), nn.GroupNorm(8, width), nn.GELU(),
            nn.Conv1d(width, width*2, 5, stride=2, padding=2), nn.GroupNorm(8, width*2), nn.GELU(),
            nn.AdaptiveAvgPool1d(16), nn.Flatten(), nn.Linear(width*2*16, embedding_dim), nn.GELU()
        ) if 'envelope' in streams else None
        count = len(streams)
        self.projection = nn.Sequential(nn.Linear(embedding_dim*count, embedding_dim), nn.LayerNorm(embedding_dim))

    def forward(self, energy, coherence, envelope=None):
        values = []
        if self.energy is not None:
            values.append(self.energy(energy))
        if self.coherence is not None:
            values.append(self.coherence(coherence.flatten(1, 3)))
        if self.envelope is not None:
            if envelope is None:
                raise ValueError('Envelope model requires verified envelope inputs')
            values.append(self.envelope(envelope))
        return F.normalize(self.projection(torch.cat(values, dim=-1)), dim=-1)

"""Trainable visual adapter for separately cached, frozen upstream RGB features.

Upstream feature generation is part of inference and its cost must be counted.
This module does not load historical HSSD/Track A heads or claim full-backbone tuning.
"""
from torch import nn
from torch.nn import functional as F
from track_b.models.structure import StructureMatcher

class FoundationVisualAdapter(nn.Module):
    def __init__(self):
        super().__init__();self.conv=nn.Sequential(nn.Conv2d(1024,64,1),nn.GELU(),nn.Conv2d(64,64,3,padding=1),nn.GELU());self.pooled=(4,16)
        self.head=nn.Sequential(nn.Linear(64*4*16,128),nn.GELU(),nn.Linear(128,16),nn.Sigmoid())
    def forward(self,features):
        if features.ndim!=4 or features.shape[1]!=1024:raise ValueError('Expected frozen RGB features B,1024,H,W')
        return self.head(F.adaptive_avg_pool2d(self.conv(features),self.pooled).flatten(1))

class FoundationStructureMatcher(StructureMatcher):
    def __init__(self):
        super().__init__();self.visual=FoundationVisualAdapter()

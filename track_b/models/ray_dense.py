"""Long-range structural ablation; ray features contain no acoustic prediction."""
import torch
from torch import nn
from track_b.models.dense import DenseScorer


class RayDenseScorer(DenseScorer):
    def __init__(self,width=32,mode='joint',geometry='rays'):
        super().__init__(width,mode)
        if geometry not in ('rays','zero'): raise ValueError(geometry)
        self.geometry=geometry
        self.geometry_projection=nn.Sequential(nn.Linear(32,width),nn.GELU(),nn.Linear(width,width))

    def candidate_query(self,fp,poses,calibration):
        if poses.shape[-1]!=35:raise ValueError('Expected xyz/yaw + 16 ranges + 16 hit flags')
        q=super().candidate_query(fp,poses[:,:3],calibration)
        geometry=poses[:,3:]
        if self.geometry=='zero':geometry=torch.zeros_like(geometry)
        return q+self.geometry_projection(geometry)[None]

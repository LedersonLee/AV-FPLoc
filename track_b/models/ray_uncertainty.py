"""Observation-only scale heads and exhaustive weighted Laplace matching.

Targets and identifiers are intentionally absent from inference interfaces.
FP descriptors may include the preserved learned residual; scales use normalized
log1p(range)/log(21) units, not metres or reflection-order uncertainty.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class RayScaleHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.network=nn.Sequential(nn.Linear(130,32),nn.GELU(),nn.Linear(32,1))
        angles=torch.arange(16)*2*math.pi/16
        self.register_buffer('direction',torch.stack([angles.sin(),angles.cos()],-1))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.constant_(self.network[-1].bias,math.log(math.expm1(.1-.001)))

    def forward(self,observation_features):
        if observation_features.ndim!=2 or observation_features.shape[1]!=128:
            raise ValueError('Expected B,128 frozen observation features')
        batch=observation_features.shape[0]
        x=torch.cat([observation_features[:,None,:].expand(-1,16,-1),self.direction[None].expand(batch,-1,-1)],-1)
        return (F.softplus(self.network(x).squeeze(-1))+.001).clamp(max=1.)


@torch.no_grad()
def frozen_observation(predictor,observation):
    """Expose the existing hidden128 and mean without altering the expert."""
    if predictor.training or any(p.requires_grad for p in predictor.parameters()):
        raise ValueError('Expert must be frozen and in eval mode')
    pooled=F.adaptive_avg_pool2d(predictor.conv(observation),predictor.pooled).flatten(1)
    hidden=predictor.head[:2](pooled)
    return hidden,predictor.head[2:](hidden)


def uncertainty_control(scale,visible,arm,constant=None):
    if scale.ndim!=2 or scale.shape[1]!=16 or visible.shape!=(16,):
        raise ValueError('Expected B,16 scale and16 visibility values')
    valid=visible.bool()
    if not valid.any():raise ValueError('Empty visible directions')
    if arm=='U3':return scale
    if arm=='U1':
        if constant is None or not .001<=float(constant)<=1.:raise ValueError('Missing or invalid Train constant')
        return torch.full_like(scale,float(constant))
    if arm=='U2':return scale[:,valid].mean(-1,keepdim=True).expand_as(scale)
    if arm=='U4':
        # CPU generator makes the predeclared permutation device-independent.
        idx=torch.where(valid.cpu())[0]
        perm=idx[torch.randperm(len(idx),generator=torch.Generator().manual_seed(20261109))]
        result=scale.clone();result[:,idx.to(scale.device)]=scale[:,perm.to(scale.device)]
        return result
    raise ValueError(arm)


def laplace_field(mean,scale,fp,weight,chunk_size=2048):
    if mean.ndim!=2 or mean.shape[-1]!=16 or scale.shape!=mean.shape:
        raise ValueError('Expected equal B,16 mean and scale')
    if fp.ndim!=2 or fp.shape[-1]!=16 or weight.shape!=fp.shape or not len(fp):
        raise ValueError('Expected nonempty N,16 FP and weights')
    if chunk_size<1:raise ValueError('chunk_size must be positive')
    if not all(torch.isfinite(x).all() for x in [mean,scale,fp,weight]) or not (scale>0).all():
        raise ValueError('Nonfinite input or nonpositive scale')
    if (weight<0).any() or (weight.sum(-1)<=0).any():raise ValueError('Invalid candidate weights')
    fields=[]
    for start in range(0,len(fp),chunk_size):
        f,w=fp[start:start+chunk_size],weight[start:start+chunk_size]
        nll=(f[None]-mean[:,None]).abs()/scale[:,None]+torch.log(2*scale[:,None])
        fields.append(-(nll*w[None]).sum(-1)/w.sum(-1)[None])
    return torch.cat(fields,-1)


def scale_loss(mean,scale,target,visible):
    """Training-only FP-derived target supervision, preserving old FoV policy."""
    if target.shape!=mean.shape or scale.shape!=mean.shape:raise ValueError('Training shapes differ')
    if visible.sum()<=0:raise ValueError('Empty training mask')
    nll=(target.detach()-mean.detach()).abs()/scale+torch.log(2*scale)
    return (nll*visible).sum(-1).mean()/visible.sum()

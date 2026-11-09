"""Geometry-supervised observation matching; no acoustic reference synthesis.

This is the explicit non-interaction baseline, not the proposed joint-attention
model. Candidate geometry is derived from the footprint only. Ground-truth
structure appears solely in the external training loss.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def visible_rays(hfov_deg=106.2602, rays=16, device=None):
    bearing=torch.arange(rays,device=device)*2*math.pi/rays
    return (torch.atan2(bearing.sin(),bearing.cos()).abs()<=math.radians(hfov_deg/2)+1e-6).float()


def rir_spectrum(rir):
    """6 log powers + six adjacent-channel IPD cos/sin; channels never averaged."""
    shape=rir.shape
    if len(shape)!=3 or shape[1]!=6:raise ValueError('Expected B,6,T')
    z=torch.stft(rir.flatten(0,1),n_fft=256,hop_length=32,win_length=256,
                 window=torch.hann_window(256,device=rir.device),return_complex=True).reshape(shape[0],6,129,-1)[:,:,:44]
    mag=z.abs()
    cross=z*torch.roll(z,-1,dims=1).conj()
    ipd=cross/cross.abs().clamp_min(1e-8)
    return torch.cat([torch.log1p(mag),ipd.real,ipd.imag],1)


class ObservationStructure(nn.Module):
    def __init__(self,channels,pooled):
        super().__init__()
        self.conv=nn.Sequential(nn.Conv2d(channels,24,5,2,2),nn.GELU(),nn.Conv2d(24,48,3,2,1),nn.GELU(),nn.Conv2d(48,64,3,2,1),nn.GELU())
        self.pooled=pooled
        self.head=nn.Sequential(nn.Linear(64*pooled[0]*pooled[1],128),nn.GELU(),nn.Linear(128,16),nn.Sigmoid())

    def forward(self,x):return self.head(F.adaptive_avg_pool2d(self.conv(x),self.pooled).flatten(1))


class StructureMatcher(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual=ObservationStructure(3,(3,8))
        self.acoustic=ObservationStructure(18,(4,64))
        self.floorplan=nn.Sequential(nn.Conv2d(2,16,3,padding=1),nn.GELU(),nn.Conv2d(16,16,3,padding=1),nn.GELU())
        self.fp_head=nn.Sequential(nn.Linear(16*9+32,64),nn.GELU(),nn.Linear(64,16))
        nn.init.normal_(self.fp_head[-1].weight,std=.01);nn.init.zeros_(self.fp_head[-1].bias)
        self.log_scale=nn.Parameter(torch.tensor(math.log(50.)))
        self.register_buffer('stencil',torch.tensor([[x,y] for y in (-1.,0.,1.) for x in (-1.,0.,1.)]))

    def observations(self,rgb,spectrum):return self.visual(rgb),self.acoustic(spectrum)

    def fp_descriptors(self,feature,candidates,calibration,head=None):
        origin=calibration['origin']; spacing=calibration['spacing']
        angle=torch.as_tensor(calibration.get('map_angle',0.),device=candidates.device,dtype=candidates.dtype);c,s=angle.cos(),angle.sin()
        rotation=torch.stack([torch.stack([c,-s]),torch.stack([s,c])])
        xy=(candidates[:,:2]-origin)@rotation;yaw=candidates[:,2]-angle
        dx=yaw.cos()[:,None]*self.stencil[:,0]-yaw.sin()[:,None]*self.stencil[:,1]
        dy=yaw.sin()[:,None]*self.stencil[:,0]+yaw.cos()[:,None]*self.stencil[:,1]
        pix=(xy[:,None]+torch.stack([dx,dy],-1))/spacing
        wh=torch.tensor([feature.shape[-1]-1,feature.shape[-2]-1],device=feature.device)
        grid=(2*pix/wh-1)[None]
        patch=F.grid_sample(feature,grid,align_corners=True).permute(0,2,3,1).flatten(2)[0]
        geom=candidates[:,3:]
        residual=.15*torch.tanh((self.fp_head if head is None else head)(torch.cat([patch,geom],-1)))
        return geom[:,:16]+residual

    def match(self,observed,fp,weight,log_scale=None):
        # Algebraic exact sum of squared differences, without B x N x D tensor.
        error=observed.square()@weight.T+(fp.square()*weight).sum(-1)[None]-2*observed@(fp*weight).T
        return -(self.log_scale if log_scale is None else log_scale).clamp(0,8).exp()*error/weight.sum(-1).clamp_min(1)[None]

    def forward(self,rgb,spectrum,floorplan,calibration,candidate_grid,chunk_size=8192,mode='fusion',geometry_residual=True):
        if mode not in ('visual','acoustic','fusion'):raise ValueError(mode)
        v,a=self.observations(rgb,spectrum);feature=self.floorplan(floorplan)
        mask=visible_rays(calibration.get('hfov_deg',106.2602),device=v.device)
        scores=[]
        for candidates in candidate_grid.split(chunk_size):
            fp=self.fp_descriptors(feature,candidates,calibration) if geometry_residual else candidates[:,3:19]
            weight=.25+.75*candidates[:,19:35]
            sv=self.match(v,fp,weight*mask);sa=self.match(a,fp,weight)
            scores.append(sv if mode=='visual' else sa if mode=='acoustic' else .85*sv+.15*sa)
        return torch.cat(scores,dim=-1)


def auxiliary_loss(v,a,target,mode='fusion',hfov_deg=106.2602):
    mask=visible_rays(hfov_deg,device=v.device)
    vl=((v-target).square()*mask).sum(-1).mean()/mask.sum()
    al=(a-target).square().mean()
    return vl if mode=='visual' else al if mode=='acoustic' else vl+al

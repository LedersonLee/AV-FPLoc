"""Posterior support on the complete valid field; no labels or injected candidates."""
import math
import numpy as np
import torch
from torch.nn import functional as F
from track_b.models.dense import rasterize


def greedy_nms(priority,poses,k=20,radius=1.,yaw_radius=math.pi/6):
    order=np.argsort(-priority,kind='stable');chosen=[]
    for idx in order:
        if not np.isfinite(priority[idx]):continue
        if chosen:
            delta=poses[chosen]-poses[idx];yaw=np.abs(np.arctan2(np.sin(delta[:,2]),np.cos(delta[:,2])))
            if np.any((np.linalg.norm(delta[:,:2],axis=1)<=radius+1e-6)&(yaw<=yaw_radius+1e-6)):continue
        chosen.append(int(idx))
        if len(chosen)==k:break
    return chosen


def support_nms(score,valid_xy,poses,grid_m=.25,yaw_bins=36,radius=1.,yaw_radius=math.pi/6,k=20):
    valid_xy=valid_xy.to(score.device)
    posterior=score.softmax(-1)
    field=rasterize(posterior,valid_xy,yaw_bins).masked_fill(~valid_xy[:,:,None],0).permute(2,0,1)[None]
    extent=math.ceil(radius/grid_m);axis=torch.arange(-extent,extent+1,device=score.device)*grid_m
    yy,xx=torch.meshgrid(axis,axis,indexing='ij');disk=((xx.square()+yy.square())<=radius**2+1e-7).to(score.dtype)
    spatial=F.conv2d(field,disk[None,None].expand(yaw_bins,1,-1,-1),padding=extent,groups=yaw_bins)
    angle_bins=int(math.floor(yaw_radius/(2*math.pi/yaw_bins)+1e-6))
    mass=sum(spatial.roll(offset,1) for offset in range(-angle_bins,angle_bins+1))[0].permute(1,2,0)[valid_xy].flatten()
    scores_field=rasterize(score,valid_xy,yaw_bins).permute(2,0,1)
    padded=torch.cat([scores_field[-1:],scores_field,scores_field[:1]],0)[None,None]
    local_max=F.max_pool3d(padded,kernel_size=3,stride=1,padding=(0,1,1))[0,0]
    is_mode=((scores_field==local_max).permute(1,2,0)[valid_xy]).flatten()
    priority=mass.masked_fill(~is_mode,-torch.inf)
    indices=greedy_nms(priority.detach().cpu().numpy(),poses.detach().cpu().numpy(),k,radius,yaw_radius)
    return torch.tensor(indices,device=score.device,dtype=torch.long),mass

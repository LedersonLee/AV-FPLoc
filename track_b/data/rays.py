"""Non-acoustic footprint boundary rays. Black means blocked/unknown, not wall.

Range truncation and map exits remain censored; never convert these ranges into
specular echo times. Uses FP only, no query observations or pose labels.
"""
import math
import numpy as np


def boundary_rays(free,spacing,origin,xy,yaw_bins=36,ray_bins=16,max_range=20.,ray_step=.1):
    angles_count=math.lcm(yaw_bins,ray_bins)
    angle=np.arange(angles_count)*2*np.pi/angles_count
    direction=np.stack([np.cos(angle),np.sin(angle)],-1)
    shape=(len(xy),angles_count);distance=np.full(shape,max_range,np.float32);hit=np.zeros(shape,bool);active=np.ones(shape,bool)
    for r in np.arange(ray_step,max_range+ray_step*.5,ray_step):
        coords=xy[:,None,:]+r*direction[None]
        ij=np.rint((coords-np.asarray(origin))/spacing).astype(int)
        inside=(ij[...,0]>=0)&(ij[...,0]<free.shape[1])&(ij[...,1]>=0)&(ij[...,1]<free.shape[0])
        blocked=~free[np.clip(ij[...,1],0,free.shape[0]-1),np.clip(ij[...,0],0,free.shape[1]-1)]
        stopped=active&(~inside|blocked)
        distance[stopped]=r;hit[stopped]=inside[stopped];active[stopped]=False
        if not active.any():break
    index=(np.arange(yaw_bins)[:,None]*(angles_count//yaw_bins)+np.arange(ray_bins)[None,:]*(angles_count//ray_bins))%angles_count
    return distance[:,index].reshape(-1,ray_bins),hit[:,index].reshape(-1,ray_bins)


def rays_at_poses(free,spacing,origin,poses,ray_bins=16,max_range=20.,ray_step=.1):
    """Continuous-query geometry supervision; never supplied to model inference."""
    angle=poses[:,2,None]+np.arange(ray_bins)[None]*2*np.pi/ray_bins
    direction=np.stack([np.cos(angle),np.sin(angle)],-1)
    distance=np.full(angle.shape,max_range,np.float32);hit=np.zeros(angle.shape,bool);active=np.ones(angle.shape,bool)
    for r in np.arange(ray_step,max_range+ray_step*.5,ray_step):
        coords=poses[:,None,:2]+r*direction;ij=np.rint((coords-np.asarray(origin))/spacing).astype(int)
        inside=(ij[...,0]>=0)&(ij[...,0]<free.shape[1])&(ij[...,1]>=0)&(ij[...,1]<free.shape[0])
        blocked=~free[np.clip(ij[...,1],0,free.shape[0]-1),np.clip(ij[...,0],0,free.shape[1]-1)]
        stopped=active&(~inside|blocked);distance[stopped]=r;hit[stopped]=inside[stopped];active[stopped]=False
        if not active.any():break
    return distance,hit

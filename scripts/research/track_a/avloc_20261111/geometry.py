"""Independent binary-map DDA ray casting, meters, +X/right and +Z/down.
Pixel centres lie at origin + (column,row)*spacing. No learned/FP-RIR inputs.
"""
import numpy as np
from scipy.ndimage import distance_transform_edt

def cast_rays(free,spacing,origin,poses,ray_count=36):
 poses=np.asarray(poses,dtype=np.float64);a=np.deg2rad(poses[:,2,None]+np.arange(ray_count)[None]*360/ray_count)
 dx=np.cos(a).ravel();dz=np.sin(a).ravel();x=np.broadcast_to((poses[:,0,None]-origin[0])/spacing+.5,a.shape).ravel();z=np.broadcast_to((poses[:,1,None]-origin[1])/spacing+.5,a.shape).ravel()
 col=np.floor(x).astype(int);row=np.floor(z).astype(int)
 inside=(col>=0)&(col<free.shape[1])&(row>=0)&(row<free.shape[0]);assert inside.all() and free[row,col].all(),'Ray origin outside free map'
 sx=np.where(dx>=0,1,-1);sz=np.where(dz>=0,1,-1);ddx=np.divide(spacing,abs(dx),out=np.full_like(dx,np.inf),where=abs(dx)>1e-12);ddz=np.divide(spacing,abs(dz),out=np.full_like(dz,np.inf),where=abs(dz)>1e-12)
 with np.errstate(invalid="ignore"):
  tx=np.where(abs(dx)>1e-12,np.where(sx>0,col+1-x,x-col)*ddx,np.inf);tz=np.where(abs(dz)>1e-12,np.where(sz>0,row+1-z,z-row)*ddz,np.inf)
 out=np.zeros(len(x));known=np.ones(len(x),bool);active=np.arange(len(x))
 for _ in range(free.shape[0]+free.shape[1]+4):
  if not len(active):break
  i=active;v=np.minimum(tx[i],tz[i]);cross_x=tx[i]<=tz[i]+1e-10;cross_z=tz[i]<=tx[i]+1e-10
  ix=i[cross_x];iz=i[cross_z];col[ix]+=sx[ix];row[iz]+=sz[iz];tx[ix]+=ddx[ix];tz[iz]+=ddz[iz]
  inside=(col[i]>=0)&(col[i]<free.shape[1])&(row[i]>=0)&(row[i]<free.shape[0]);hit=~inside;hit[inside]|=~free[row[i[inside]],col[i[inside]]]
  out[i[hit]]=v[hit];known[i[~inside]]=False;active=i[~hit]
 assert len(active)==0,'Ray traversal did not terminate'
 return out.reshape(a.shape).astype(np.float32),known.reshape(a.shape)

def xy_grid(free,spacing,origin,step):
 x,z=np.meshgrid(np.arange(0,(free.shape[1]-1)*spacing+1e-8,step),np.arange(0,(free.shape[0]-1)*spacing+1e-8,step));cc=np.rint(x/spacing).astype(int);rr=np.rint(z/spacing).astype(int);valid=free[rr,cc] & ((distance_transform_edt(free)*spacing)[rr,cc]>=.06)
 return np.stack([x[valid]+origin[0],z[valid]+origin[1]],-1)

def bank_for_yaws(world_rays):
 return world_rays[:,(np.arange(36)[:,None]+np.arange(36)[None])%36]

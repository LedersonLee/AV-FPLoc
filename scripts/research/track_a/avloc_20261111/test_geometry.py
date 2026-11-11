import json,numpy as np
from geometry import cast_rays,xy_grid,bank_for_yaws
# Rectangular room: interior pixel centres .1..1.9, walls at centre0 and2.
f=np.zeros((21,21),bool);f[1:-1,1:-1]=True;p=np.array([[1.,1.,0.],[.5,.7,30.]])
r,known=cast_rays(f,.1,[0,0],p);assert known.all()
a=np.deg2rad(p[:,2,None]+np.arange(36)*10);dx=np.cos(a);dz=np.sin(a)
with np.errstate(divide='ignore'):
 tx=np.where(dx>0,(1.95-p[:,0,None])/dx,(.05-p[:,0,None])/dx);tz=np.where(dz>0,(1.95-p[:,1,None])/dz,(.05-p[:,1,None])/dz)
tx[abs(dx)<1e-12]=np.inf;tz[abs(dz)<1e-12]=np.inf
np.testing.assert_allclose(r,np.minimum(tx,tz),atol=2e-6)
# Rigid quarter-turn of a square map should permute only ray bearings.
r2,_=cast_rays(f,.1,[0,0],np.array([[1.,1.,90.]]));np.testing.assert_allclose(r2[0],np.roll(r[0],-9),atol=2e-6)
xy=xy_grid(f,.1,[0,0],.2);assert len(xy)==81
world,_=cast_rays(f,.1,[0,0],np.column_stack([xy,np.zeros(len(xy))]));bank=bank_for_yaws(world)
idx=17;direct,_=cast_rays(f,.1,[0,0],np.array([[*xy[idx],70.]]));np.testing.assert_array_equal(bank[idx,7],direct[0])
print(json.dumps(dict(status='pass',checks=['analytic rectangular room boundary','yaw sign and90deg equivariance','query-independent XY grid','bank yaw permutation matches direct cast'],note='No distance cap; raster exit carries known=false'),indent=2))

import numpy as np
def angular_rays(depth,scale):
    """Matches UnLoc get_ray_from_depth_uncertainty(V11,F_W3/8)."""
    angles=np.deg2rad(np.arange(-5,6)*10)
    coordinates=np.tan(angles)*len(depth)*3/8+(len(depth)-1)/2
    if coordinates.min()<0 or coordinates.max()>len(depth)-1:raise ValueError('Ray fan exceeds image')
    return np.interp(coordinates,np.arange(len(depth)),depth)/np.cos(angles),np.interp(coordinates,np.arange(len(depth)),scale)

def bank_windows(payload,xy):
    step=payload['desdf_spacing_m'];origin=np.array([payload['origin_x_m'],payload['origin_z_min_m']])
    indices=np.rint((np.asarray(xy)-origin)/step).astype(int)
    reconstructed=origin+indices*step
    if not np.allclose(reconstructed,xy,atol=1e-6,rtol=0):raise ValueError('FP/vision grids misaligned')
    if (indices<0).any() or (indices[:,0]>=payload['desdf'].shape[1]).any() or (indices[:,1]>=payload['desdf'].shape[0]).any():
        raise ValueError('FP bank exceeds vision DESDF')
    field=payload['desdf'][indices[:,1],indices[:,0]]
    if not np.all(np.any(field>0,axis=1)):raise ValueError('FP and vision disagree on valid grid cells')
    bearings=(np.arange(36)[:,None]+np.arange(-5,6))%36
    return field[:,bearings].astype(np.float32)

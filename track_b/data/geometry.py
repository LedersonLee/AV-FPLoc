"""Map-local X/Z compatibility; body forward/left/up uses Y=-world Z.

Map-local yaw increases toward +Z (image down). A right-handed FLU world uses
(x, y)=(X,-Z) and yaw=-logical_yaw. No observation/GT selects map components.
"""
import numpy as np

RING = np.arange(3, 9) % 6 * np.pi / 3


def candidate_bank(free, spacing, origin, step=.25, yaw_bins=36):
    if step <= 0 or spacing <= 0 or yaw_bins < 1 or free.ndim != 2:
        raise ValueError('Invalid grid')
    # Anchor to the map pixel-centre origin, never to a query trajectory.
    xs = np.arange(0, (free.shape[1]-1)*spacing+1e-8, step)
    zs = np.arange(0, (free.shape[0]-1)*spacing+1e-8, step)
    x,z = np.meshgrid(xs,zs)
    rc = np.stack([np.rint(z/spacing),np.rint(x/spacing)],-1).astype(int)
    valid = free[rc[...,0],rc[...,1]]
    xy = np.stack([x+origin[0],z+origin[1]],-1)[valid]
    poses = np.concatenate([np.repeat(xy,yaw_bins,axis=0),
                            np.tile(np.arange(yaw_bins)*2*np.pi/yaw_bins,len(xy))[:,None]],1)
    return poses, valid


def world_to_pixel(xz, spacing, origin):
    return (np.asarray(xz)-origin)/spacing


def canonical_rir(rir, logical_yaw_deg, base_acoustic_yaw_deg):
    a = (-logical_yaw_deg) % 360
    turns = (a-base_acoustic_yaw_deg)/60
    if rir.ndim != 2 or rir.shape[0] != 6 or not np.isclose(turns,round(turns)):
        raise ValueError('RIR/yaw contract mismatch')
    return np.roll(rir,-int(round(turns)),axis=0)


def mic_offsets(logical_yaw_deg, radius=.05):
    a = RING - np.deg2rad(logical_yaw_deg)
    return radius*np.stack([np.cos(a),-np.sin(a)],-1)


def camera_forward(habitat_yaw_deg):
    a=np.deg2rad(habitat_yaw_deg)
    return np.array([-np.sin(a),-np.cos(a)])


def rigid_transform(poses, angle, offset):
    c,s=np.cos(angle),np.sin(angle)
    xy=poses[:,:2]@np.array([[c,s],[-s,c]])+offset
    return np.column_stack([xy,(poses[:,2]+angle)%(2*np.pi)])

"""Ray/segment targets in native X/Z coordinates; missing hits are censored."""
import numpy as np


def intersect_segments(poses, segments, directions=36, max_range=20.):
    poses = np.asarray(poses, dtype=np.float64)
    segments = np.asarray(segments, dtype=np.float64).reshape(-1, 2, 2)
    angle = poses[:, 2, None] + np.arange(directions) * 2*np.pi/directions
    rays = np.stack((np.cos(angle), np.sin(angle)), -1)
    distances = np.full(angle.shape, np.inf)
    def cross(a, b):
        return a[..., 0]*b[..., 1] - a[..., 1]*b[..., 0]
    for start in range(0, len(segments), 1024):
        seg = segments[start:start+1024]
        edge = seg[:, 1]-seg[:, 0]
        offset = seg[None, None, :, 0]-poses[:, None, None, :2]
        denominator = cross(rays[:, :, None], edge)
        safe = np.where(np.abs(denominator)>1e-10, denominator, 1.)
        t = cross(offset, edge)/safe
        u = cross(offset, rays[:, :, None])/safe
        valid = (np.abs(denominator)>1e-10)&(t>1e-5)&(u>=-1e-8)&(u<=1+1e-8)
        distances = np.minimum(distances, np.where(valid, t, np.inf).min(-1))
    mask = np.isfinite(distances)&(distances<=max_range)
    return np.where(mask, distances, max_range).astype(np.float32), mask

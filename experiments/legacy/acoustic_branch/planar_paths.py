"""Deterministic first-order specular horizontal-path proxy, NOT full-wave 2D.

No ceiling/floor in the planar model, no diffraction/scattering/high-order
reverberation. Uses 1/(4*pi*distance) amplitude (3D spreading on planar paths),
not a cylindrical-wave Green function. Intended to isolate timing cues.
"""
import numpy as np


def runs(v):
    d=np.diff(np.r_[False,v,False].astype(np.int8))
    return zip(np.flatnonzero(d==1),np.flatnonzero(d==-1))


def segments_from_mask(free,spacing,xmin,zmin):
    p=np.pad(np.asarray(free,dtype=bool),1)
    edges=[]
    x=lambda i:xmin+(i-.5)*spacing
    z=lambda i:zmin+(i-.5)*spacing
    for boundary,shift in [(free & ~p[:-2,1:-1],0),(free & ~p[2:,1:-1],1)]:
        for r in np.flatnonzero(boundary.any(1)):
            for a,b in runs(boundary[r]):edges.append([[x(a),z(r+shift)],[x(b),z(r+shift)]])
    for boundary,shift in [(free & ~p[1:-1,:-2],0),(free & ~p[1:-1,2:],1)]:
        for c in np.flatnonzero(boundary.any(0)):
            for a,b in runs(boundary[:,c]):edges.append([[x(c+shift),z(a)],[x(c+shift),z(b)]])
    return np.asarray(edges,dtype=float)


def cross(a,b):return a[...,0]*b[...,1]-a[...,1]*b[...,0]


def clear_paths(start,end,walls):
    """Visibility of line segments, ignoring contact at the two ray endpoints."""
    start,end=np.broadcast_arrays(np.atleast_2d(start),np.atleast_2d(end))
    ray=end-start;edge=walls[:,1]-walls[:,0]
    den=cross(ray[:,None],edge[None])
    off=walls[None,:,0]-start[:,None]
    with np.errstate(divide='ignore',invalid='ignore'):
        t=cross(off,edge[None])/den
        u=cross(off,ray[:,None])/den
    hit=(abs(den)>1e-12)&(t>1e-7)&(t<1-1e-7)&(u>=-1e-9)&(u<=1+1e-9)
    return ~hit.any(1)


def first_order(source,target,walls,max_distance=43.904):
    source,target=np.asarray(source),np.asarray(target)
    a,b=walls[:,0],walls[:,1]
    edge=b-a;normal=np.column_stack([-edge[:,1],edge[:,0]])
    normal/=np.linalg.norm(normal,axis=1,keepdims=True)
    signed=((source-a)*normal).sum(1)
    image=source-2*signed[:,None]*normal
    ray=target-image
    den=cross(ray,edge)
    with np.errstate(divide='ignore',invalid='ignore'):
        t=cross(a-image,edge)/den
        u=cross(a-image,ray)/den
    valid=(abs(den)>1e-12)&(t>0)&(t<1)&(u>=-1e-9)&(u<=1+1e-9)
    valid &= signed*((target-a)*normal).sum(1)>0
    ids=np.flatnonzero(valid)
    points=image[ids]+t[ids,None]*ray[ids]
    length=np.linalg.norm(points-source,axis=1)+np.linalg.norm(points-target,axis=1)
    visible=(length<=max_distance)&clear_paths(source,points,walls)&clear_paths(target,points,walls)
    return length[visible],points[visible],ids[visible]


def add_arrival(signal,distance,gain=1.,fs=48000,speed=343.):
    """Windowed-sinc fractional delay, with NO added global filter delay."""
    position=distance*fs/speed
    centre=int(np.floor(position))
    indices=centre+np.arange(-40,41)
    taps=np.sinc(indices-position)*np.hanning(81)
    taps/=max(taps.sum(),1e-12)
    keep=(indices>=0)&(indices<len(signal))
    signal[indices[keep]]+=gain/(4*np.pi*max(distance,1e-9))*taps[keep]


def render(source_xyz,base,walls,floor,ceiling,reflection=.8):
    from .contracts import RING
    source=np.asarray(source_xyz)[[0,2]]
    angles=RING+np.deg2rad(base)
    microphones=source+np.column_stack([.05*np.cos(angles),-.05*np.sin(angles)])
    planar=np.zeros((6,6152),dtype=np.float64)
    with_vertical=np.zeros_like(planar)
    records=[]
    for mic,target in enumerate(microphones):
        if not clear_paths(source,target,walls)[0]:raise ValueError('Array crosses a canonical wall')
        direct=np.linalg.norm(target-source)
        add_arrival(planar[mic],direct)
        lengths,points,ids=first_order(source,target,walls)
        for length in lengths:add_arrival(planar[mic],length,reflection)
        with_vertical[mic]=planar[mic]
        vertical=[]
        for plane in (floor,ceiling):
            distance=np.sqrt(direct**2+(2*(source_xyz[1]-plane))**2)
            vertical.append(float(distance));add_arrival(with_vertical[mic],distance,reflection)
        records.append(dict(mic=mic,horizontal_lengths_m=lengths.tolist(),wall_ids=ids.tolist(),
            reflection_points_xz=points.tolist(),vertical_lengths_m=vertical))
    return planar.astype(np.float32),with_vertical.astype(np.float32),records

import numpy as np
import pytest
from track_b.data.geometry import candidate_bank, canonical_rir, mic_offsets, camera_forward, world_to_pixel, rigid_transform


def test_bank_exhaustive_and_no_gt():
    free=np.ones((9,13),bool); free[3:6,4:7]=False
    p,mask=candidate_bank(free,.25,[-1,2],.25,36)
    assert len(p)==int(free.sum())*36
    assert np.array_equal(mask,free)
    ij=np.rint(world_to_pixel(p[:,:2],.25,[-1,2])).astype(int)
    assert free[ij[:,1],ij[:,0]].all()
    assert len(np.unique(p,axis=0))==len(p)


def test_six_mic_roll_for_all_headings():
    for yaw in range(0,360,10):
        base=(-yaw)%60
        actual=canonical_rir(mic_offsets(-base),yaw,base)
        np.testing.assert_allclose(actual,mic_offsets(yaw),atol=1e-14)
    with pytest.raises(ValueError): canonical_rir(np.zeros((7,50)),0,0)


def test_camera_heading_and_flu_handedness():
    for yaw in range(0,360,45):
        np.testing.assert_allclose(camera_forward(-yaw-90),[np.cos(np.deg2rad(yaw)),np.sin(np.deg2rad(yaw))],atol=1e-14)
        # FLU: forward/left in (X,Yup,Z) must cross to up.
        a=np.deg2rad(yaw); f=np.array([np.cos(a),0,np.sin(a)]); left=np.array([np.sin(a),0,-np.cos(a)])
        np.testing.assert_allclose(np.cross(f,left),[0,1,0],atol=1e-14)


def test_rigid_roundtrip():
    p=np.array([[1.,2.,.3],[3.,4.,5.]])
    a=.9; t=np.array([4.,-2.]); q=rigid_transform(p,a,t)
    c,s=np.cos(a),np.sin(a)
    inv=-t@np.array([[c,-s],[s,c]])
    np.testing.assert_allclose(rigid_transform(q,-a,inv),p,atol=1e-14)

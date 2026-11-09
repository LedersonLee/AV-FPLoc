import numpy as np
from track_b.data.rays import boundary_rays


def test_square_distances_and_heading():
    free=np.ones((101,101),bool);free[[0,-1],:]=False;free[:,[0,-1]]=False
    d,h=boundary_rays(free,.1,[0,0],np.array([[2.,5.]]),yaw_bins=4,ray_bins=4,ray_step=.01)
    np.testing.assert_allclose(d[0],[8,5,2,5],atol=.061)
    np.testing.assert_allclose(d[1],np.roll(d[0],-1),atol=.011)
    assert h.all()


def test_unknown_map_exit_censoring():
    free=np.ones((11,11),bool)
    d,h=boundary_rays(free,.1,[0,0],np.array([[.5,.5]]),yaw_bins=4,ray_bins=4,max_range=2)
    assert not h.any()
    assert np.all(d<2)

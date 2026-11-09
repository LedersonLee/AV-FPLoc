import math
import numpy as np
import torch
from track_b.models.structure import StructureMatcher,rir_spectrum,auxiliary_loss
from track_b.tests.test_dense import fixture
from track_b.data.rays import boundary_rays,rays_at_poses


def test_ray_supervision_matches_grid_without_gt_injection():
    free=np.ones((51,61),bool);free[[0,-1]]=False;free[:,[0,-1]]=False
    xy=np.array([[1.,2.],[3.,2.]])
    d,h=boundary_rays(free,.1,[0,0],xy,12)
    p=np.column_stack([np.repeat(xy,12,axis=0),np.tile(np.arange(12)*math.pi/6,2)])
    d2,h2=rays_at_poses(free,.1,[0,0],p)
    np.testing.assert_allclose(d,d2,atol=.101) # raster tie at floating ray/pixel boundary
    assert h2.all()


def test_spectral_spatial_channels_and_silence():
    r=torch.randn(2,6,8192);s=rir_spectrum(r)
    assert s.shape==(2,18,44,257) and torch.isfinite(s).all()
    assert torch.isfinite(rir_spectrum(torch.zeros_like(r))).all()
    assert not torch.allclose(s,rir_spectrum(r.roll(1,1)))


def test_structure_dense_gradient_chunk_and_labels_absent():
    torch.manual_seed(42);rgb,rir,fp,cal,p=fixture();fp=fp[:1]
    spec=rir_spectrum(rir);p=torch.cat([p,torch.rand(len(p),16),torch.ones(len(p),16)],1)
    m=StructureMatcher();a=m(rgb,spec,fp,cal,p,chunk_size=17);b=m(rgb,spec,fp,cal,p,chunk_size=9999)
    torch.testing.assert_close(a,b,atol=1e-5,rtol=1e-5)
    a.log_softmax(-1)[:,5].neg().mean().backward()
    for branch in (m.visual,m.acoustic,m.floorplan,m.fp_head):assert sum(float(q.grad.abs().sum()) for q in branch.parameters() if q.grad is not None)>0
    assert a.shape==(2,len(p))


def test_match_equals_explicit_squared_distance():
    m=StructureMatcher();o=torch.rand(3,16);fp=torch.rand(41,16);w=torch.rand(41,16)
    expected=-50*((o[:,None]-fp[None]).square()*w).sum(-1)/w.sum(-1)
    torch.testing.assert_close(m.match(o,fp,w),expected,atol=2e-5,rtol=2e-5)

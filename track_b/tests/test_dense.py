import inspect
import numpy as np
import torch
from track_b.models.dense import DenseScorer,dense_loss,topk
from track_b.data.geometry import candidate_bank

torch.set_num_threads(2)

def fixture():
    torch.manual_seed(42)
    free=np.ones((9,13),bool);free[0]=False;free[-1]=False;free[:,0]=False;free[:,-1]=False;free[3:5,4:6]=False
    poses,_=candidate_bank(free,.25,[0,0],.5,12)
    p=torch.tensor(poses,dtype=torch.float32)
    fp=torch.tensor(np.stack([free,~free]),dtype=torch.float32)[None]
    return torch.rand(2,3,24,32),torch.randn(2,6,512),fp.expand(2,-1,-1,-1),{'origin':torch.zeros(2),'spacing':.25},p


def test_full_bank_chunk_and_gradient():
    m=DenseScorer();args=fixture();a=m(*args,chunk_size=17);b=m(*args,chunk_size=999999)
    torch.testing.assert_close(a,b,atol=2e-7,rtol=2e-6)
    assert a.shape==(2,len(args[-1]))
    dense_loss(a,args[-1],args[-1][[1,13]]).backward()
    for branch in (m.visual,m.acoustic,m.floorplan,m.v_attn,m.a_attn,m.joint):
        assert sum(float(x.grad.abs().sum()) for x in branch.parameters() if x.grad is not None)>0


def test_gt_absent_shuffle_new_map():
    m=DenseScorer().eval();args=list(fixture());s=m(*args)
    assert 'gt_pose' not in inspect.signature(m.forward).parameters
    args[1]=args[1].flip(0);t=m(*args);assert (s-t).abs().max()>1e-7
    args[2]=torch.rand(2,2,15,17);assert m(*args).shape==s.shape


def test_extract_and_frozen():
    m=DenseScorer();args=fixture()
    for branch in (m.visual,m.acoustic,m.floorplan):
        for p in branch.parameters(): p.requires_grad_(False)
    s=m(*args);dense_loss(s,args[-1],args[-1][[1,2]]).backward()
    assert all(p.grad is None for branch in (m.visual,m.acoustic,m.floorplan) for p in branch.parameters())
    for radius in (None,.5):
        idx=topk(s[0],args[-1],radius=radius)
        assert len(set(idx.tolist()))==len(idx) and idx.min()>=0 and idx.max()<len(args[-1])


def test_translation_origin_equivariance():
    m=DenseScorer().eval();args=list(fixture());s=m(*args)
    args[-1]=args[-1]+torch.tensor([4.,-2.,0.]);args[-2]={'origin':torch.tensor([4.,-2.]),'spacing':.25}
    torch.testing.assert_close(s,m(*args),atol=2e-7,rtol=2e-6)


def test_rigid_map_frame_equivariance():
    from track_b.data.geometry import rigid_transform
    m=DenseScorer().eval();args=list(fixture());s=m(*args)
    args[-1]=torch.tensor(rigid_transform(args[-1].numpy(),.7,[4.,-2.]),dtype=torch.float32)
    args[-2]={'origin':torch.tensor([4.,-2.]),'spacing':.25,'map_angle':.7}
    torch.testing.assert_close(s,m(*args),atol=3e-7,rtol=3e-6)


def test_masked_raster_full_field():
    from track_b.models.dense import rasterize
    mask=torch.tensor([[True,False],[False,True]])
    scores=torch.arange(24,dtype=torch.float32).reshape(2,12)
    field=rasterize(scores,mask,6)
    assert field.shape==(2,2,2,6)
    assert torch.isneginf(field[:,~mask]).all()
    torch.testing.assert_close(field[:,mask].flatten(1),scores)


def test_ray_dense_chunk_gradient_and_zero_control():
    from track_b.models.ray_dense import RayDenseScorer
    args=list(fixture());args[-1]=torch.cat([args[-1],torch.rand(len(args[-1]),32)],-1)
    m=RayDenseScorer();a=m(*args,chunk_size=17);b=m(*args,chunk_size=99999)
    torch.testing.assert_close(a,b,atol=3e-7,rtol=3e-6)
    dense_loss(a,args[-1][:,:3],args[-1][[1,13],:3]).backward()
    assert sum(float(x.grad.abs().sum()) for x in m.geometry_projection.parameters())>0
    z=RayDenseScorer(geometry='zero').eval();before=z(*args);args[-1][:,3:]=0
    torch.testing.assert_close(before,z(*args))


def test_single_modality_isolation():
    args=list(fixture());v=DenseScorer(mode='visual').eval();a=DenseScorer(mode='acoustic').eval()
    sv=v(*args);sa=a(*args)
    original_rgb=args[0];args[0]=torch.randn_like(args[0]);torch.testing.assert_close(sa,a(*args))
    args[0]=original_rgb;args[1]=torch.randn_like(args[1]);torch.testing.assert_close(sv,v(*args))

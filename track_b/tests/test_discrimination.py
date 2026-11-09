import inspect
import torch
from track_b.models.discrimination import candidate_features, GeometryDiscriminator

def test_candidate_features_rigid_frame_and_no_label_input():
    torch.manual_seed(7)
    c=torch.rand(25,35); v,a=torch.rand(16),torch.rand(16)
    vf,af=torch.rand(25,16),torch.rand(25,16)
    vs,ac=torch.randn(25),torch.randn(25); idx=torch.tensor([2,7,13])
    mask=torch.ones(16)
    x,s=candidate_features(v,a,c,vf,af,vs,ac,idx,mask)
    cc=c.clone(); angle=torch.tensor(.7)
    rotation=torch.stack([torch.stack([angle.cos(),-angle.sin()]),torch.stack([angle.sin(),angle.cos()])])
    cc[:,:2]=c[:,:2]@rotation.T+torch.tensor([4.,-3.]); cc[:,2]+=angle
    xx,ss=candidate_features(v,a,cc,vf,af,vs,ac,idx,mask)
    assert torch.allclose(x,xx) and torch.allclose(s,ss)
    assert not any('gt' in p or 'label' in p for p in inspect.signature(candidate_features).parameters)
    model=GeometryDiscriminator(); assert model(x).shape==(3,)
    model(x).sum().backward();assert all(p.grad is not None for p in model.parameters())

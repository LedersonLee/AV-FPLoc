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


def test_joint_posterior_support_uses_actual_full_field():
    # Separated poses have singleton support, so mass has an exact independent oracle.
    c=torch.zeros(25,35);c[:,0]=torch.arange(25)*3.
    v,a=torch.zeros(16),torch.ones(16)
    vf,af=torch.zeros(25,16),torch.ones(25,16)
    vs,ac=torch.zeros(25),torch.zeros(25)
    field=torch.linspace(-3,3,25);idx=field.topk(3).indices;mask=torch.ones(16)
    x,s=candidate_features(v,a,c,vf,af,vs,ac,idx,mask,score_field=field)
    assert torch.allclose(s,field.log_softmax(0)[idx])
    assert torch.allclose(x[:,-3],field[idx]-field.max())
    expected_entropy=-(field.softmax(0)*field.log_softmax(0)).sum()/torch.tensor(25.).log()
    assert torch.allclose(x[:,-1],expected_entropy.expand(3))
    old,old_support=candidate_features(v,a,c,vf,af,vs,ac,idx,mask)
    explicit,explicit_support=candidate_features(v,a,c,vf,af,vs,ac,idx,mask,score_field=.85*vs+.15*ac)
    assert torch.equal(old,explicit) and torch.equal(old_support,explicit_support)
    assert not torch.allclose(s,old_support)
    import pytest
    with pytest.raises(ValueError,match='full-field'):
        candidate_features(v,a,c,vf,af,vs,ac,idx,mask,score_field=field[idx])
    with pytest.raises(ValueError,match='finite'):
        candidate_features(v,a,c,vf,af,vs,ac,idx,mask,score_field=field*float('nan'))

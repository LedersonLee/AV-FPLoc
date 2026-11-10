import torch
from track_b.acoustic_geometry.features import features
from track_b.acoustic_geometry.models import GeometryNet,masked_loss
from track_b.acoustic_geometry.train import donors


def test_frontend_channel_information_and_loss_mask():
    torch.set_num_threads(2);torch.manual_seed(9)
    x=torch.randn(2,6,4096);e,s=features(x)
    assert e.shape==(2,6,64,128) and s.shape==(2,18,64,128)
    e2,s2=features(x*3)
    torch.testing.assert_close(e,e2,atol=2e-6,rtol=1e-5)
    torch.testing.assert_close(s,s2,atol=2e-5,rtol=1e-4)
    _,sr=features(x.roll(1,1))
    torch.testing.assert_close(s[:,:6].roll(1,1),sr[:,:6])
    for architecture in ['echo','spatial']:
      for head in ['regression','categorical']:
        model=GeometryNet(architecture,head)
        mean,logits=model(e if architecture=='echo' else s)
        assert mean.shape==(2,36) and torch.isfinite(mean).all()
        target=torch.ones_like(mean);mask=torch.ones_like(mean,dtype=torch.bool);mask[:,0]=False
        loss=masked_loss(mean,logits,target,mask)
        target[:,0]=1000
        torch.testing.assert_close(loss,masked_loss(mean,logits,target,mask))
        loss.backward()
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.encoder.parameters())


def test_donors_change_position_not_only_heading():
    poses=torch.tensor([[0.,0.,0.],[0.,0.,1.],[2.,0.,0.],[2.,0.,1.]])
    ids=donors(poses)
    assert ((poses[ids,:2]-poses[:,:2]).norm(dim=1)>.5).all()
    torch.testing.assert_close(poses[ids,2],poses[:,2])


def test_heads_share_exact_encoder_initialization():
    for architecture in ['echo','spatial']:
        torch.manual_seed(42);a=GeometryNet(architecture,'regression')
        torch.manual_seed(42);b=GeometryNet(architecture,'categorical')
        for k,v in a.encoder.state_dict().items():torch.testing.assert_close(v,b.encoder.state_dict()[k])
        pa=sum(p.numel() for p in a.parameters());pb=sum(p.numel() for p in b.parameters())
        assert abs(pa-pb)/pa<.15

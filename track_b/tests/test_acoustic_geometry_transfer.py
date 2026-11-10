import torch
from track_b.acoustic_geometry.models import GeometryNet
from track_b.acoustic_geometry.transfer import make_model,tensor_hash,score
from track_b.models.foundation import FoundationStructureMatcher


def test_transfer_shared_initialization_and_frozen_encoder():
    torch.set_num_threads(2);torch.manual_seed(10)
    ck={'model':GeometryNet('spatial','regression').state_dict()}
    models={arm:make_model(42,arm,'spatial',ck if arm!='scratch' else None) for arm in ['scratch','frozen','finetuned','joint']}
    assert len({tensor_hash(m.acoustic.readout) for m in models.values()})==1
    assert len({tensor_hash(m.floorplan) for m in models.values()})==1
    assert len({tensor_hash(m.fp_head) for m in models.values()})==1
    frozen=models['frozen'];before=tensor_hash(frozen.acoustic.geometry.encoder)
    opt=torch.optim.SGD([p for p in frozen.parameters() if p.requires_grad],lr=.01)
    x=torch.randn(2,18,64,128);frozen.train();y=frozen.acoustic(x);y.sum().backward();opt.step()
    assert y.shape==(2,16) and before==tensor_hash(frozen.acoustic.geometry.encoder)
    for arm in ['finetuned','joint']:
        m=models[arm];m.train();out=m.acoustic(x,auxiliary=arm=='joint')
        loss=sum(z.sum() for z in out) if isinstance(out,tuple) else out.sum();loss.backward()
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.acoustic.geometry.encoder.parameters())


def test_transfer_does_not_change_score_algebra():
    torch.manual_seed(22);base=FoundationStructureMatcher();m=make_model(22,'scratch','spatial')
    observed=torch.rand(3,16);fp=torch.rand(100,16);c=torch.rand(100,35)
    expected=base.match(observed,fp,.25+.75*c[:,19:35])
    torch.testing.assert_close(score(m,observed,fp,c),expected)

import torch
from track_b.models.foundation import FoundationVisualAdapter

def test_frozen_feature_adapter_is_trainable_and_shape_general():
 torch.manual_seed(42);m=FoundationVisualAdapter();x=torch.randn(2,1024,4,16);y=m(x);assert y.shape==(2,16) and torch.isfinite(y).all();(y-.7).square().mean().backward()
 assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
 assert sum(float(p.grad.abs().sum()) for p in m.parameters())>0
 assert m(torch.randn(1,1024,8,12)).shape==(1,16)

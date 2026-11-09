import torch
from track_b.models.foundation import FoundationVisualAdapter

def test_frozen_feature_adapter_is_trainable_and_shape_general():
 torch.manual_seed(42);m=FoundationVisualAdapter();x=torch.randn(2,1024,4,16);y=m(x);assert y.shape==(2,16) and torch.isfinite(y).all();(y-.7).square().mean().backward()
 assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
 assert sum(float(p.grad.abs().sum()) for p in m.parameters())>0
 assert m(torch.randn(1,1024,8,12)).shape==(1,16)

def test_feature_attachment_uses_exact_query_alignment_and_no_pixel_scaling():
 import pytest
 from track_b.scripts.train_structure import attach_visual_features,visual_input
 scene={'scene':'synthetic','query_ids':[3,7],'rgb':torch.zeros(2,3,8,8,dtype=torch.uint8)};x=torch.full((2,1024,4,16),2.,dtype=torch.float16);package={'scene':'synthetic','query_ids':[3,7],'visual_features':x}
 attach_visual_features(scene,package);assert visual_input(scene,[0]).mean()==2
 with pytest.raises(AssertionError):attach_visual_features(scene,{**package,'query_ids':[7,3]})

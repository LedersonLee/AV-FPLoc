import torch
from track_b.models.interaction import InteractionMatcher
from track_b.models.structure import rir_spectrum
from track_b.tests.test_dense import fixture

def test_interaction_full_field_checkpoint_gradients_and_capacity():
 torch.manual_seed(42);rgb,rir,fp,cal,p=fixture();spec=rir_spectrum(rir);fp=fp[:1];p=torch.cat([p,torch.rand(len(p),16),torch.ones(len(p),16)],1);counts=[]
 for mode in ('attention','concat','separable'):
  m=InteractionMatcher(mode);counts.append(sum(v.numel() for v in m.parameters()));one=m(rgb,spec,fp,cal,p,chunk_size=17);two=m(rgb,spec,fp,cal,p,chunk_size=9999,checkpoint_chunks=False)
  torch.testing.assert_close(one,two,atol=2e-5,rtol=1e-5)
  one.log_softmax(-1)[:,5].neg().mean().backward()
  for name,param in m.named_parameters():assert param.grad is not None and torch.isfinite(param.grad).all(),name
  for branch in (m.visual,m.acoustic,m.floorplan,m.v_retrieval,m.a_retrieval,m.joint_head):assert sum(float(x.grad.abs().sum()) for x in branch.parameters())>0
 assert len(set(counts))==1

def test_separable_has_no_cross_modality_score_interaction():
 torch.manual_seed(42);rgb,rir,fp,cal,p=fixture();spec=rir_spectrum(rir);fp=fp[:1];p=torch.cat([p,torch.rand(len(p),16),torch.ones(len(p),16)],1)
 m=InteractionMatcher('separable').eval()
 with torch.no_grad():
  f=m(rgb,spec,fp,cal,p);v=m(rgb.flip(0),spec,fp,cal,p);a=m(rgb,spec.flip(0),fp,cal,p);both=m(rgb.flip(0),spec.flip(0),fp,cal,p)
 torch.testing.assert_close(f+both,v+a,atol=3e-5,rtol=1e-5)

def test_interaction_rigid_frame_and_new_map_without_labels():
 import inspect
 from track_b.data.geometry import rigid_transform
 rgb,rir,fp,cal,p=fixture();spec=rir_spectrum(rir);fp=fp[:1];p=torch.cat([p,torch.rand(len(p),16),torch.ones(len(p),16)],1)
 m=InteractionMatcher().eval();assert 'gt' not in inspect.signature(m.forward).parameters
 with torch.no_grad():
  before=m(rgb,spec,fp,cal,p);moved=p.clone();moved[:,:3]=torch.tensor(rigid_transform(p[:,:3].numpy(),.7,[3.,-2.]),dtype=torch.float32)
  after=m(rgb,spec,fp,{'origin':torch.tensor([3.,-2.]),'spacing':.25,'map_angle':.7},moved)
  torch.testing.assert_close(before,after,atol=3e-5,rtol=1e-5)
  assert m(rgb,spec,torch.rand(1,2,17,23),cal,p).shape==before.shape
  assert not torch.allclose(before,m(rgb,spec.flip(0),fp,cal,p),atol=1e-5,rtol=1e-5)

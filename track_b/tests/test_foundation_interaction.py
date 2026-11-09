import torch
from track_b.models.foundation_interaction import FoundationInteractionMatcher
from track_b.models.foundation import FoundationStructureMatcher
from track_b.models.structure import StructureMatcher,rir_spectrum
from track_b.tests.test_dense import fixture

def test_independent_start_and_cached_scores():
 torch.manual_seed(42);rgb,rir,fp,cal,p=fixture();x=torch.randn(len(rgb),1024,4,16);spec=rir_spectrum(rir);fp=fp[:1];p=torch.cat([p,torch.rand(len(p),16),torch.ones(len(p),16)],1)
 visual=FoundationStructureMatcher().eval();acoustic=StructureMatcher().eval();counts=[]
 with torch.no_grad():
  reference=.85*visual(x,spec,fp,cal,p,mode='visual')+.15*acoustic(rgb,spec,fp,cal,p,mode='acoustic')
  for mode in ['attention','concat','separable']:
   m=FoundationInteractionMatcher(mode).eval();m.initialize_independent(visual,acoustic);counts.append(sum(p.numel() for p in m.parameters()))
   out=m(x,spec,fp,cal,p,chunk_size=17);cached=m.forward_cached(x,spec,m.prepare_map(fp,cal,p),chunk_size=19)
   torch.testing.assert_close(out,reference,atol=3e-5,rtol=1e-5);torch.testing.assert_close(out,cached,atol=3e-5,rtol=1e-5)
 assert len(set(counts))==1

def test_active_branches_and_separability():
 torch.manual_seed(42);rgb,rir,fp,cal,p=fixture();x=torch.randn(len(rgb),1024,4,16);spec=rir_spectrum(rir);fp=fp[:1];p=torch.cat([p,torch.rand(len(p),16),torch.ones(len(p),16)],1)
 for mode in ['attention','concat','separable']:
  m=FoundationInteractionMatcher(mode);field=m(x,spec,fp,cal,p,chunk_size=17);field.log_softmax(-1)[:,5].neg().mean().backward()
  for name,branch in [('visual',m.visual),('acoustic',m.acoustic),('visual_fp',m.floorplan),('audio_fp',m.acoustic_floorplan),('vr',m.v_retrieval),('ar',m.a_retrieval)]:
   assert sum(float(p.grad.abs().sum()) for p in branch.parameters() if p.grad is not None)>0,name
  if mode=='separable':
   m.eval()
   with torch.no_grad():
    f=m(x,spec,fp,cal,p);v=m(x.flip(0),spec,fp,cal,p);a=m(x,spec.flip(0),fp,cal,p);both=m(x.flip(0),spec.flip(0),fp,cal,p)
   torch.testing.assert_close(f+both,v+a,atol=3e-5,rtol=1e-5)

def test_foundation_dual_map_frame_and_new_size_without_labels():
 import inspect
 from track_b.data.geometry import rigid_transform
 torch.manual_seed(43);rgb,rir,fp,cal,p=fixture();x=torch.randn(len(rgb),1024,4,16);spec=rir_spectrum(rir);fp=fp[:1];p=torch.cat([p,torch.rand(len(p),16),torch.ones(len(p),16)],1)
 for mode in ['attention','concat','separable']:
  m=FoundationInteractionMatcher(mode).eval();assert not any(k in inspect.signature(m.forward).parameters for k in ['gt','scene_id','query_id','reference_rir'])
  with torch.no_grad():
   before=m(x,spec,fp,cal,p);moved=p.clone();moved[:,:3]=torch.tensor(rigid_transform(p[:,:3].numpy(),.7,[3.,-2.]),dtype=p.dtype)
   after=m(x,spec,fp,{'origin':torch.tensor([3.,-2.]),'spacing':.25,'map_angle':.7},moved)
   torch.testing.assert_close(before,after,atol=3e-5,rtol=1e-5)
   new=m(x,spec,torch.rand(1,2,17,23),cal,p)
   assert new.shape==before.shape and torch.isfinite(new).all()

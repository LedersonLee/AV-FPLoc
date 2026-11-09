import inspect
import math
import pytest
import torch
from track_b.models.ray_uncertainty import RayScaleHead,frozen_observation,laplace_field,uncertainty_control,scale_loss
from track_b.models.structure import visible_rays,ObservationStructure


def test_scale_bounds_and_frozen_means():
    torch.manual_seed(42);expert=ObservationStructure(18,(4,64)).eval().requires_grad_(False)
    obs=torch.randn(2,18,44,65);hidden,mean=frozen_observation(expert,obs)
    assert torch.equal(mean,expert(obs)) and hidden.shape==(2,128)
    head=RayScaleHead();b=head(hidden)
    assert b.shape==(2,16) and torch.allclose(b,torch.full_like(b,.1))
    loss=scale_loss(mean,b,torch.rand_like(mean),torch.ones(16));loss.backward()
    assert all(p.grad is None for p in expert.parameters())
    assert head.network[-1].weight.grad.abs().sum()>0
    assert sum(p.numel() for p in head.parameters())==4225
    with torch.no_grad():head.network[-1].bias.fill_(100)
    assert torch.equal(head(hidden),torch.ones(2,16))
    with torch.no_grad():head.network[-1].bias.fill_(-100)
    assert (head(hidden)>=.001).all()


def test_constant_l1_ranking_and_chunk_exactness():
    mean=torch.zeros(2,16);fp=torch.stack([torch.zeros(16),torch.ones(16)*.2,torch.ones(16)*.8])
    b=torch.full_like(mean,.1);w=torch.ones_like(fp)
    expected=-fp.abs().mean(-1)/.1-math.log(.2)
    field=laplace_field(mean,b,fp,w)
    assert torch.allclose(field,expected.expand(2,-1)) and field.argmax(-1).tolist()==[0,0]
    assert torch.equal(field,laplace_field(mean,b,fp,w,chunk_size=1))
    # Equidistant candidates keep a tie; do not invent an invariant top-k ordering.
    fp[2]=-.2
    assert torch.equal(laplace_field(mean,b,fp,w)[:,1],laplace_field(mean,b,fp,w)[:,2])


def test_fov_and_observation_only_controls():
    mask=visible_rays();mean=torch.rand(2,16);fp=torch.rand(7,16);w=torch.ones_like(fp)*mask
    b=torch.linspace(.01,.8,16).expand(2,-1).clone()
    score=laplace_field(mean,b,fp,w)
    mm=mean.clone();ff=fp.clone();bb=b.clone();mm[:,mask==0]=100;ff[:,mask==0]=-100;bb[:,mask==0]=.9
    assert torch.equal(score,laplace_field(mm,bb,ff,w))
    perm=uncertainty_control(b,mask,'U4')
    assert torch.equal(perm[:,mask==0],b[:,mask==0])
    assert torch.equal(perm[:,mask.bool()].sort().values,b[:,mask.bool()].sort().values)
    assert not torch.allclose(score,laplace_field(mean,perm,fp,w))
    global_b=uncertainty_control(b,mask,'U2')
    assert torch.allclose(global_b,b[:,mask.bool()].mean(-1,keepdim=True).expand_as(b))
    assert not any('gt' in k or 'target' in k or 'scene' in k or 'query' in k for k in inspect.signature(laplace_field).parameters)
    with pytest.raises(ValueError):laplace_field(mean,b,fp,torch.zeros_like(w))
    with pytest.raises(ValueError):laplace_field(mean,b*0,fp,w)


def test_log_scale_term_candidate_constant_only_with_fixed_weights():
    mean=torch.zeros(1,16);fp=torch.zeros(2,16);b=torch.linspace(.01,1,16)[None]
    w=torch.ones_like(fp);fixed=laplace_field(mean,b,fp,w)
    assert fixed[0,0]==fixed[0,1]
    w[0,0]=.25
    varying=laplace_field(mean,b,fp,w)
    assert varying[0,0]!=varying[0,1]
    expected=-(torch.log(2*b)*w).sum(-1)/w.sum(-1)
    assert torch.allclose(varying[0],expected)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_cpu_gpu_laplace_agreement():
    torch.manual_seed(12);m=torch.rand(3,16);b=torch.rand(3,16)*.5+.001;fp=torch.rand(31,16);w=torch.rand(31,16)+.25
    cpu=laplace_field(m,b,fp,w,chunk_size=7)
    gpu=laplace_field(m.cuda(),b.cuda(),fp.cuda(),w.cuda(),chunk_size=11).cpu()
    assert torch.allclose(cpu,gpu,rtol=1e-5,atol=2e-5)

"""GT-free destructive controls on cached candidate-map evidence."""
import math
import torch

def intervene_map(cache, candidates, condition):
    keys=('fp','acoustic_fp','q','weight')
    n=len(candidates)
    assert all(len(cache[k])==n for k in keys)
    if condition=='normal':return cache
    if condition=='candidate_geometry_removed':
        return {**cache,**{k:cache[k].mean(0,keepdim=True).expand_as(cache[k]) for k in keys}}
    if condition not in ('yaw90','yaw180','yaw_random'):raise ValueError(condition)
    assert n%36==0,'Control requires the frozen36-yaw bank'
    grouped=candidates[:,:3].reshape(-1,36,3)
    assert torch.allclose(grouped[:,:,:2],grouped[:,:1,:2].expand(-1,36,-1))
    step=grouped[:,1:,2]-grouped[:,:-1,2]
    wrapped=torch.atan2(step.sin(),step.cos())
    assert torch.allclose(wrapped,torch.full_like(wrapped,2*math.pi/36),atol=1e-5)
    shift=9 if condition=='yaw90' else 18
    if condition=='yaw_random':
        generator=torch.Generator().manual_seed(20261109)
        shifts=torch.randint(1,36,(n//36,1),generator=generator).to(candidates.device)
        idx=(torch.arange(n//36,device=candidates.device)[:,None]*36+(torch.arange(36,device=candidates.device)[None]+shifts)%36).flatten()
    else:
        idx=torch.arange(n,device=candidates.device).reshape(-1,36).roll(-shift,1).flatten()
    return {**cache,**{k:cache[k][idx] for k in keys}}

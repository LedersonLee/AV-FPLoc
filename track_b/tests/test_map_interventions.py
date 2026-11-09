import math
import pytest
import torch
from track_b.models.map_interventions import intervene_map

def test_map_controls_preserve_positions_and_remove_candidate_variation():
    p=torch.zeros(72,35);p[36:,0]=4;p[:,2]=torch.arange(36).repeat(2)*2*math.pi/36
    cache={k:torch.arange(72).float()[:,None].expand(-1,d) for k,d in [('fp',16),('acoustic_fp',16),('q',32),('weight',16)]};cache['hfov']=106.
    out=intervene_map(cache,p,'yaw90')
    assert out['fp'][0,0]==9 and out['fp'][35,0]==8 and out['fp'][36,0]==45
    assert torch.equal(out['fp'].sort(0).values,cache['fp'])
    random=intervene_map(cache,p,'yaw_random')
    assert torch.equal(random['fp'],intervene_map(cache,p,'yaw_random')['fp'])
    assert torch.equal(random['fp'].sort(0).values,cache['fp'])
    assert (random['fp'][:36]<36).all() and (random['fp'][36:]>=36).all()
    removed=intervene_map(cache,p,'candidate_geometry_removed')
    assert all(torch.equal(removed[k],removed[k][:1].expand_as(removed[k])) for k in ['fp','acoustic_fp','q','weight'])
    assert torch.equal(cache['fp'][:,0],torch.arange(72).float())
    with pytest.raises(AssertionError):intervene_map(cache,p.flip(0),'yaw90')

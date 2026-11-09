import math
import numpy as np
import torch
from track_b.data.geometry import candidate_bank
from track_b.eval.hypotheses import support_nms
from track_b.models.structure import rir_spectrum


def test_support_mass_against_exact_pose_neighborhood():
    free=np.ones((3,3),bool);free[1,1]=False;p,valid=candidate_bank(free,.5,[0,0],.5,12);p=torch.tensor(p,dtype=torch.float32)
    torch.manual_seed(1);score=torch.randn(len(p));idx,mass=support_nms(score,torch.tensor(valid),p,.5,12,radius=.5)
    prob=score.softmax(0);expected=[]
    for pose in p:
        d=p-pose;yaw=torch.atan2(d[:,2].sin(),d[:,2].cos()).abs()
        expected.append(prob[(d[:,:2].norm(dim=-1)<=.5+1e-6)&(yaw<=math.pi/6+1e-6)].sum())
    torch.testing.assert_close(mass,torch.stack(expected),atol=2e-7,rtol=2e-6)
    assert 0<len(idx)<=20 and len(idx.unique())==len(idx)


def test_six_mic_rotation_matches_spectral_group_permutation():
    r=torch.randn(2,6,8192);s=rir_spectrum(r)
    transformed=torch.cat([s[:,offset:offset+6].roll(3,1) for offset in (0,6,12)],1)
    torch.testing.assert_close(transformed,rir_spectrum(r.roll(3,1)),atol=1e-6,rtol=1e-6)

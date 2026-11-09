"""GT-free, body-frame candidate features for a newly learned backend.

No scene/query identifiers, absolute poses, target rays or reference RIR enter
the classifier. Poses are used only for relative posterior neighborhood mass.
"""
import math
import torch
from torch import nn


def candidate_features(v, a, candidates, visual_fp, acoustic_fp, visual_score,
                       acoustic_score, indices, visible):
    fused = .85 * visual_score + .15 * acoustic_score
    logp = fused.log_softmax(-1)
    pose = candidates[:, :3]
    delta = pose[indices, None] - pose[None]
    yaw = torch.atan2(delta[..., 2].sin(), delta[..., 2].cos()).abs()
    near = (delta[..., :2].norm(dim=-1) <= 1 + 1e-6) & (yaw <= math.pi/6 + 1e-6)
    support = torch.logsumexp(logp[None].expand(len(indices), -1).masked_fill(~near, -torch.inf), -1)
    k = len(indices)
    obs_v, obs_a = v.expand(k, -1), a.expand(k, -1)
    vf, af = visual_fp[indices], acoustic_fp[indices]
    entropy = -(logp.exp() * logp).sum() / math.log(max(2, len(fused)))
    local = torch.stack([visual_score[indices]-visual_score.max(),
                         acoustic_score[indices]-acoustic_score.max(),
                         fused[indices]-fused.max(), support,
                         entropy.expand(k)], -1)
    features = torch.cat([obs_v, obs_a, candidates[indices, 3:35], vf, af,
                          (obs_v-vf)*visible, obs_a-af, local], -1)
    assert features.shape == (k, 133) and torch.isfinite(features).all()
    return features, support


class GeometryDiscriminator(nn.Module):
    def __init__(self, features=133):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(features, 64), nn.GELU(),
                                     nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, features):
        return self.network(features).squeeze(-1)

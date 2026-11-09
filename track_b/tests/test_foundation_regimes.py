import torch
from track_b.models.foundation_interaction import FoundationInteractionMatcher
from track_b.models.foundation_regimes import set_foundation_regime

def test_regime_boundaries_and_optimizer_preserves_frozen_weights():
    counts={}
    for regime in ('frozen','partial','joint'):
        counts[regime]=[]
        for mode in ('attention','concat','separable'):
            m=FoundationInteractionMatcher(mode); active=set_foundation_regime(m,regime)
            assert 'log_scale' in active and 'acoustic_log_scale' in active
            assert 'joint_head.2.weight' in active and 'v_projection.weight' in active
            assert ('visual.conv.0.weight' in active)==(regime=='joint')
            assert ('acoustic_floorplan.0.weight' in active)==(regime=='joint')
            assert ('visual.head.2.weight' in active)==(regime!='frozen')
            assert ('acoustic_fp_head.2.weight' in active)==(regime!='frozen')
            before={n:p.detach().clone() for n,p in m.named_parameters() if not p.requires_grad}
            opt=torch.optim.AdamW(m.parameters(),lr=.01,weight_decay=.1)
            sum(p.square().sum() for p in m.parameters() if p.requires_grad).backward();opt.step()
            assert all(torch.equal(before[n],p) for n,p in m.named_parameters() if n in before)
            counts[regime].append(sum(active.values()))
        assert len(set(counts[regime]))==1
    assert counts['frozen'][0]<counts['partial'][0]<counts['joint'][0]

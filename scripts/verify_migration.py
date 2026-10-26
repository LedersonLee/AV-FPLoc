"""Numerical parity against an explicit preserved workspace (developer-only)."""
import argparse
from pathlib import Path
import sys
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--legacy-root',type=Path,required=True)
    p.add_argument('--hssd-root',type=Path,required=True);a=p.parse_args()
    sys.path.insert(0,str(a.legacy_root.resolve()))
    from avfploc.fusion.features import extract
    from avfploc.evaluation.baselines import standardized
    from reliability_fusion.core import features
    from reliability_fusion.input_supervision_audit import engineered,envelope
    rng=np.random.default_rng(42);v,audio=[standardized(rng.normal(size=(120,36))) for _ in range(2)]
    xy=rng.normal(size=(120,2))*5;depth=rng.uniform(.1,10,40);rays=rng.uniform(.1,10,(120,36))
    core=features(v,audio,xy,depth)[[*range(10),13,14,15,16]]
    extra,_,_=engineered(v,audio,xy,depth,rays,envelope(v,audio))
    x,names=extract(v,audio,xy,depth,'processed58',rays)
    np.testing.assert_array_equal(x,np.r_[core,extra])
    import torch
    from acoustic_branch.train import AcousticEncoder as OldEncoder
    from avfploc.acoustic.train import AcousticEncoder
    ck=a.hssd_root/'AcousticContrastive/pilots/scene_subset_v1/input_ablation_v1/runs/all/best.pt'
    state=torch.load(ck,map_location='cpu',weights_only=False)
    old=OldEncoder(**state['config']['model']).eval();new=AcousticEncoder(**state['config']['model']).eval()
    for model in (old,new):model.load_state_dict(state['model'],strict=True)
    torch.set_num_threads(2);torch.manual_seed(42)
    values=[torch.randn(2,6,19,95),torch.randn(2,2,6,2,19),torch.randn(2,6,252)]
    with torch.inference_mode():torch.testing.assert_close(old(*values),new(*values),rtol=0,atol=0)
    from avfploc.acoustic.features import extract as extract_ec
    from avfploc.acoustic.envelope_cache import extract as extract_n
    from acoustic_branch.features import extract as old_ec
    from acoustic_branch.envelope_cache import extract as old_n
    rir=np.load(a.hssd_root/'RIR/Observed/train/102815835/records/p00020/base_acoustic_yaw_000.npy',allow_pickle=False)
    energy,coherence=extract_ec(rir,state['config']['features'])
    oe,oc=old_ec(rir,state['config']['features'])
    env=extract_n(rir).astype(np.float16).astype(np.float32)
    np.testing.assert_array_equal(energy,oe);np.testing.assert_array_equal(coherence,oc)
    np.testing.assert_array_equal(env,old_n(rir).astype(np.float16).astype(np.float32))
    values=[torch.tensor(z)[None] for z in (energy,coherence,env)]
    with torch.inference_mode():torch.testing.assert_close(old(*values),new(*values),rtol=0,atol=0)
    from avfploc.fusion.infer import load_gate
    gate=a.hssd_root/'ReliabilityFusion/pilots/capacity_probe_v1/runs/128_64_seed42/best.pt'
    model,mean,std,stored=load_gate(gate)
    if stored!=names:raise ValueError('Real processed58 checkpoint schema mismatch')
    from reliability_fusion.model import Gate
    original=Gate(58,hidden=(128,64)).eval();original.load_state_dict(model.state_dict())
    xt=torch.as_tensor((x-mean)/std,dtype=torch.float32)[None]
    with torch.inference_mode():torch.testing.assert_close(original(xt),model(xt),rtol=0,atol=0)
    print('PASS: exact processed58 features, synthetic/real-RIR E+C+N checkpoint outputs, 128x64 gate outputs; Test untouched')


if __name__=='__main__':main()

"""Synchronized batch1 downstream fusion latency; score construction excluded."""
import copy
from pathlib import Path
import platform
import time

import numpy as np
import torch

from .core import features
from .model import Gate


def measure(fn,warmup=20,repeats=100,cuda=False):
    for _ in range(warmup):fn()
    if cuda:torch.cuda.synchronize()
    samples=[]
    for _ in range(repeats):
        if cuda:torch.cuda.synchronize()
        start=time.perf_counter_ns();fn()
        if cuda:torch.cuda.synchronize()
        samples.append((time.perf_counter_ns()-start)/1e6)
    return dict(median_ms=float(np.median(samples)),p95_ms=float(np.quantile(samples,.95)),
        min_ms=float(np.min(samples)),repeats=repeats,warmup=warmup)


@torch.inference_mode()
def benchmark(scenes,checkpoint,config):
    state=torch.load(checkpoint,map_location='cpu',weights_only=False);cpu=Gate(14).eval();cpu.load_state_dict(state['model']);gpu=copy.deepcopy(cpu).cuda().eval()
    mean,std=state['mean'],state['std'];old_threads=torch.get_num_threads();torch.set_num_threads(1)
    ordered=sorted(scenes,key=lambda s:len(s['arrays']['xy']));chosen=[ordered[i] for i in sorted(set((0,len(ordered)//2,len(ordered)-1)))]
    results=[]
    try:
        for scene in chosen:
            v,a=np.array(scene['arrays']['scores'][0]);xy=np.array(scene['arrays']['xy']);depth=np.linspace(.5,5,11)
            def extract14():return features(v,a,xy,depth)[config['feature_columns']]
            x=extract14();np.testing.assert_allclose(x,scene['arrays']['features'][0],atol=2e-5,rtol=2e-5)
            normalized=(x-mean)/std;xt=torch.tensor(normalized[None],dtype=torch.float32);xg=xt.cuda()
            weight=float(cpu(xt).item());vg=torch.tensor(v,device='cuda');ag=torch.tensor(a,device='cuda')
            def gate_cpu():return float(cpu(torch.tensor(((x-mean)/std)[None],dtype=torch.float32)).item())
            def map_cpu(w=weight):return int(((1-w)*v+w*a).argmax())
            def full_cpu():
                vector=extract14();w=float(cpu(torch.tensor(((vector-mean)/std)[None],dtype=torch.float32)).item());return map_cpu(w)
            def gpu_roundtrip():
                vv,aa=vg.cpu().numpy(),ag.cpu().numpy()
                vector=features(vv,aa,xy,depth)[config['feature_columns']]
                w=float(cpu(torch.tensor(((vector-mean)/std)[None],dtype=torch.float32)).item())
                return int(((1-w)*vg+w*ag).argmax().item())
            np.testing.assert_equal(map_cpu(),gpu_roundtrip())
            results.append(dict(scene_id=scene['scene_id'],split=scene['split'],cells=len(xy),yaw_bins=36,pose_candidates=len(xy)*36,
                feature14_via_current17=measure(extract14),gate14_cpu=measure(gate_cpu),fusion_MAP_cpu=measure(map_cpu),
                downstream_CPU_resident=measure(full_cpu),gate14_GPU_resident=measure(lambda:gpu(xg),cuda=True),
                downstream_GPU_resident_roundtrip=measure(gpu_roundtrip,cuda=True)))
        cpu_name=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),platform.processor())
        return dict(status='complete',batch_size=1,CPU=cpu_name,GPU=torch.cuda.get_device_name(0),cpu_torch_threads=1,
            gate_parameters=sum(p.numel() for p in cpu.parameters()),gate_input_features=14,legacy_feature_statistics_computed=17,
            CPU_pipeline='existing17-stat extraction then select14 -> train normalization -> CPU MLP -> fused MAP',
            GPU_roundtrip_pipeline='resident GPU score maps -> CPU copy/features/MLP -> GPU fusion/MAP -> synchronized scalar result',
            excluded=['RGB/RIR acquisition','visual/acoustic encoder forward','dense candidate score-map construction',
                'FP RIR generation','disk/network input','model loading'],
            note='Downstream latency only, not end-to-end AV localization FPS. GPU timings synchronized per batch1 query; no throughput amortization.',scenes=results)
    finally:torch.set_num_threads(old_threads)

"""Limited real-checkpoint, GT-free cached scoring audit and timing probe."""
import argparse,builtins,io,time,json
from pathlib import Path
import numpy as np
import torch
from track_b.models.foundation import FoundationStructureMatcher
from track_b.models.structure import StructureMatcher,visible_rays
from track_b.models.ray_uncertainty import RayScaleHead,frozen_observation,uncertainty_control,laplace_field
from track_b.scripts.train_structure import scene_to_gpu,full_descriptors
from track_b.scripts.train_ray_uncertainty import read,sha,write


def main():
    p=argparse.ArgumentParser()
    for k in ['scale-root','study-root','out']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--code-sha',required=True);a=p.parse_args();torch.set_num_threads(2)
    r=a.scale_root;study=a.study_root;seed=42;sid=read(r/'cache_v1/protocol.json')['scenes']['val'][0]
    s=scene_to_gpu(torch.load(r/'cache_v1'/f'{sid}.pt',weights_only=True));f=torch.load(r/'features_v1'/f'{sid}.pt',weights_only=True);assert f['query_ids']==s['query_ids']
    # Remove every supervision/identifier field before the scored inference path.
    for key in ['gt','target','target_hit','query_ids','scene','split']:s.pop(key,None)
    rgb=f['visual_features'][:1].cuda().float();spectrum=s['spectrum'][:1].float();shifted=s['spectrum'][1:2].float()
    models={};heads={};manifest=read(study/'preflight_v1/validation.json');hp_path=study/'heads_v1/heads_42.pt';assert sha(hp_path)==read(study/'heads_v1/heads_42.json')['checkpoint_sha256'];hp=torch.load(hp_path,weights_only=True)
    for mode,cls,stage in [('visual',FoundationStructureMatcher,'visual_joint'),('acoustic',StructureMatcher,'acoustic')]:
        path=r/'scaling_fits_v1'/f'n112_budget1_{stage}_attempt1'/f'{mode}_{seed}.pt';assert sha(path)==manifest['checkpoints'][f'{mode}_{seed}']['sha256']
        models[mode]=cls().cuda().eval().requires_grad_(False);models[mode].load_state_dict(torch.load(path,weights_only=True));heads[mode]=RayScaleHead().cuda().eval();heads[mode].load_state_dict(hp['heads'][mode])
    masks={'visual':visible_rays(s['hfov_deg'],device='cuda'),'acoustic':torch.ones(16,device='cuda')};weights={m:(.25+.75*s['candidates'][:,19:35])*masks[m] for m in models}
    def maps():return {m:full_descriptors(models[m],s,True) for m in models}
    def infer(arm,cache,chunk=2048,audio=spectrum):
        fields={}
        for m,obs in [('visual',rgb),('acoustic',audio)]:
            hidden,mean=frozen_observation(getattr(models[m],m),obs)
            if arm=='U0':fields[m]=models[m].match(mean,cache[m],weights[m])
            else:
                b=uncertainty_control(heads[m](hidden),masks[m],arm,hp['constants'][m]);fields[m]=laplace_field(mean,b,cache[m],weights[m],chunk)
        fields['fixed']=.85*fields['visual']+.15*fields['acoustic'];return fields
    checks=[];timings={};events=[]
    with torch.no_grad():
        torch.cuda.synchronize();t=time.perf_counter();cache=maps();torch.cuda.synchronize();setup=time.perf_counter()-t
        rebuilt=maps();assert all(torch.equal(cache[m],rebuilt[m]) for m in models)
        for arm in ['U0','U1','U2','U3','U4']:
            reference=infer(arm,cache);other=infer(arm,rebuilt,chunk=1024)
            for method in reference:
                x,y=reference[method],other[method];checks.append(dict(arm=arm,method=method,max_abs=float((x-y).abs().max()),allclose=bool(torch.allclose(x,y,rtol=1e-5,atol=2e-5)),argmax_equal=bool(x.argmax()==y.argmax()),top_score_ties=int((x==x.max()).sum())))
            for _ in range(2):infer(arm,cache)
            elapsed=[];torch.cuda.reset_peak_memory_stats()
            for _ in range(5):
                torch.cuda.synchronize();t=time.perf_counter();infer(arm,cache);torch.cuda.synchronize();elapsed.append(time.perf_counter()-t)
            timings[arm]=dict(repetitions=5,warmup=2,milliseconds=[1000*x for x in elapsed],median_ms=1000*float(np.median(elapsed)),peak_allocated_bytes=torch.cuda.max_memory_allocated())
        clean=infer('U3',cache);shuffled=infer('U3',cache,audio=shifted);permuted=infer('U4',cache)
        audio_change=float((clean['fixed']-shuffled['fixed']).abs().max());permutation_change=float((clean['fixed']-permuted['fixed']).abs().max())
        original_open=builtins.open;original_io_open=io.open
        def traced_open(file,*args,**kwargs):events.append(str(file));return original_open(file,*args,**kwargs)
        def traced_io(file,*args,**kwargs):events.append(str(file));return original_io_open(file,*args,**kwargs)
        builtins.open=traced_open;io.open=traced_io
        try:
            for arm in ['U0','U1','U2','U3','U4']:infer(arm,cache)
            torch.cuda.synchronize()
        finally:builtins.open=original_open;io.open=original_io_open
    a.out.mkdir(parents=True,exist_ok=False);result=dict(code_sha=a.code_sha,scene=sid,seed=seed,query_position=0,candidates=len(s['candidates']),checks=checks,python_open_events=events,audio_shuffle_max_field_change=audio_change,scale_permutation_max_field_change=permutation_change,map_setup_seconds=setup,timing=timings,scope='One reused Val query/seed; cached foundation observations, maps and models preloaded. No GT/target/IDs in scoring. Python open hooks only, not native/syscall tracing. Timing includes local encoders, scales and field scoring; excludes raw RGB/RIR, upstream features, loading, Top20/metrics and maps; concurrent GPU evaluations.')
    write(a.out/'validation.json',result)
    assert all(x['allclose'] for x in checks) and not events and audio_change>0 and permutation_change>0
    print('PASS',len(checks),'field comparisons',flush=True)
if __name__=='__main__':main()

"""Study30 clean-Train head-only fitting, with preserved expert states."""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from track_b.models.foundation import FoundationStructureMatcher
from track_b.models.structure import StructureMatcher,visible_rays
from track_b.models.ray_uncertainty import RayScaleHead,frozen_observation,scale_loss

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2))

def main():
    p=argparse.ArgumentParser()
    for name in ['scale-root','study-root','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--code-sha',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
    r=a.scale_root;study=a.study_root;declared=read(study/'protocol.json');audit=read(study/'preflight_v1/validation.json');replay=read(study/'u0_replay_v1/comparison.json')
    assert not replay['mismatches'] and replay['rows']==14592
    assert declared['training']['steps']==1000 and declared['training']['batch']==32
    assert declared['head']['initial_scale']==.1 and declared['head']['units']=='log1p(metres)/log(21)'
    cache=r/'cache_v1';features=r/'features_v1';data=read(cache/'protocol.json');scenes=data['scenes']['train'][:1] if a.smoke else data['scenes']['train'];seeds=[42] if a.smoke else declared['seeds'];steps=2 if a.smoke else 1000
    cm={x['scene']:x for x in read(cache/'manifest.json')};fm={x['scene']:x for x in read(features/'manifest.json')}
    a.out.mkdir(parents=True,exist_ok=False)
    setup=dict(code_sha=a.code_sha,smoke=a.smoke,steps=steps,seeds=seeds,train_scenes=scenes,queries_per_scene=4 if a.smoke else 64,declared_protocol_sha256=sha(study/'protocol.json'),input_audit_sha256=sha(study/'preflight_v1/validation.json'),u0_replay_sha256=sha(study/'u0_replay_v1/comparison.json'),training=declared['training'],head=declared['head'],test_accessed=False)
    write(a.out/'protocol.json',setup);fits=[]
    for seed in seeds:
        torch.manual_seed(seed);models={};before={};paths={}
        for mode,cls,stage in [('visual',FoundationStructureMatcher,'visual_joint'),('acoustic',StructureMatcher,'acoustic')]:
            path=r/'scaling_fits_v1'/f'n112_budget1_{stage}_attempt1'/f'{mode}_{seed}.pt'
            assert sha(path)==audit['checkpoints'][f'{mode}_{seed}']['sha256']
            model=cls().cuda().eval().requires_grad_(False);model.load_state_dict(torch.load(path,weights_only=True))
            models[mode]=model;before[mode]={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};paths[mode]=path
        torch.cuda.synchronize();start=time.perf_counter();packs=[];torch.cuda.reset_peak_memory_stats()
        for sid in scenes:
            path=cache/f'{sid}.pt';fpath=features/f'{sid}.pt';assert sha(path)==cm[sid]['cache_sha256'] and sha(fpath)==fm[sid]['sha256']
            s=torch.load(path,weights_only=True,map_location='cpu');f=torch.load(fpath,weights_only=True,map_location='cpu');assert s['query_ids']==f['query_ids']
            count=4 if a.smoke else len(s['query_ids']);target=s['target'][:count].cuda();assert torch.isfinite(target).all() and (target>=0).all() and (target<=1.000001).all()
            collected={m:[] for m in models};means={m:[] for m in models}
            for ids in torch.arange(count).split(8):
                for mode,predictor,obs in [('visual',models['visual'].visual,f['visual_features'][ids].float().cuda()),('acoustic',models['acoustic'].acoustic,s['spectrum'][ids].float().cuda())]:
                    hidden,mean=frozen_observation(predictor,obs)
                    # Real-checkpoint extraction must reproduce the original predictor.
                    with torch.no_grad():assert torch.equal(mean,predictor(obs))
                    collected[mode].append(hidden);means[mode].append(mean)
            packs.append(dict(hidden={m:torch.cat(collected[m]) for m in models},mean={m:torch.cat(means[m]) for m in models},target=target))
            del s,f
        torch.cuda.synchronize();extract_seconds=time.perf_counter()-start;extract_peak=torch.cuda.max_memory_allocated()
        masks={'visual':visible_rays(device='cuda'),'acoustic':torch.ones(16,device='cuda')}
        constants={m:float(torch.stack([((x['target']-x['mean'][m]).abs()*masks[m]).sum()/len(x['target'])/masks[m].sum() for x in packs]).mean().clamp(.001,1.)) for m in models}
        # Re-seed after frozen-model construction so head initialization is explicit.
        torch.manual_seed(seed);heads={m:RayScaleHead().cuda() for m in models};initial={m:{k:v.detach().cpu().clone() for k,v in h.state_dict().items()} for m,h in heads.items()}
        optimizer=torch.optim.AdamW([p for h in heads.values() for p in h.parameters()],lr=.001,weight_decay=.0001);rng=np.random.default_rng(seed);history=[]
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
        for step in range(steps):
            x=packs[int(rng.integers(len(packs)))];idx=torch.tensor(rng.integers(len(x['target']),size=32),device='cuda');optimizer.zero_grad()
            losses={m:scale_loss(x['mean'][m][idx],heads[m](x['hidden'][m][idx]),x['target'][idx],masks[m]) for m in models}
            loss=sum(losses.values());assert torch.isfinite(loss);loss.backward();optimizer.step()
            if step%100==0 or step==steps-1:
                row=dict(step=step,loss=float(loss),**{m+'_nll':float(v) for m,v in losses.items()});history.append(row);print(seed,row,flush=True)
        torch.cuda.synchronize();seconds=time.perf_counter()-start;peak=torch.cuda.max_memory_allocated()
        for m,model in models.items():
            assert all(torch.equal(v.detach().cpu(),before[m][k]) for k,v in model.state_dict().items())
            assert all(p.grad is None for p in model.parameters()) and sha(paths[m])==audit['checkpoints'][f'{m}_{seed}']['sha256']
            assert any(not torch.equal(v.detach().cpu(),initial[m][k]) for k,v in heads[m].state_dict().items())
        stats={}
        with torch.no_grad():
            for m in models:
                means=torch.cat([x['mean'][m] for x in packs]);targets=torch.cat([x['target'] for x in packs]);scale=torch.cat([heads[m](x['hidden'][m]) for x in packs]);mask=masks[m].bool();error=(targets-means).abs()[:,mask];b=scale[:,mask]
                stats[m]=dict(mae=float(error.mean()),ray_nll=float((error/b+torch.log(2*b)).mean()),fixed_ray_nll=float((error/constants[m]+np.log(2*constants[m])).mean()),mean_scale=float(b.mean()),min_scale=float(b.min()),max_scale=float(b.max()),coverage_b=float((error<=b).float().mean()),coverage_90=float((error<=b*np.log(10)).float().mean()))
        path=a.out/f'heads_{seed}.pt';torch.save(dict(heads={m:h.state_dict() for m,h in heads.items()},constants=constants,seed=seed),path)
        rec=dict(code_sha=a.code_sha,seed=seed,steps=steps,queries=sum(len(x['target']) for x in packs),parameters_per_modality=4225,constant_scales=constants,history=history,train_only_diagnostics=stats,expert_states_bitwise_unchanged=True,expert_checkpoint_files_unchanged=True,expert_gradients_absent=True,head_parameters_changed=True,extraction_seconds=extract_seconds,extraction_peak_allocated=extract_peak,head_training_seconds=seconds,head_peak_allocated=peak,checkpoint_sha256=sha(path),scope='In-sample clean Train diagnostics, not held-out efficacy; allocated peak includes resident extracted features')
        write(a.out/f'heads_{seed}.json',rec);fits.append(rec);write(a.out/'summary.json',fits)
        del models,before,heads,optimizer,packs;torch.cuda.empty_cache()
    print('COMPLETE',flush=True)
if __name__=='__main__':main()

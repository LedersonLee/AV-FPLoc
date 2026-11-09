"""Exhaustive Study30 arms; fixed frozen experts, GT only after scoring."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from track_b.scripts.train_ray_uncertainty import read,sha,write
from track_b.models.foundation import FoundationStructureMatcher
from track_b.models.structure import StructureMatcher,visible_rays
from track_b.models.ray_uncertainty import RayScaleHead,frozen_observation,uncertainty_control,laplace_field
from track_b.scripts.train_structure import scene_to_gpu,full_descriptors
from track_b.eval.extended_metrics import extended_pose_statistics,reliability_statistics


def main():
    p=argparse.ArgumentParser()
    for k in ['scale-root','study-root','out']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--code-sha',required=True);p.add_argument('--condition',choices=['normal','occlusion25','occlusion50','occlusion75','brightness','audio_shuffle','audio_noise20'],required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();torch.set_num_threads(2)
    scale=a.scale_root;study=a.study_root;data=read(scale/'cache_v1/protocol.json');audit=read(study/'preflight_v1/validation.json');fits={x['seed']:x for x in read(study/'heads_v1/summary.json')};reference={ (r['method'],r['seed'],r['scene'],r['query']):r for r in read(scale/'n112_budget1_evaluation_v1/predictions.json')}
    assert set(fits)=={42,43,44} and all(x['steps']==1000 and x['queries']==7168 and x['expert_states_bitwise_unchanged'] for x in fits.values())
    feature_root=scale/'features_v1'
    if a.condition.startswith('occlusion'):feature_root=study/f'features_{a.condition}_v1'
    if a.condition=='brightness':feature_root=scale/'features_lowlight_v1'
    fp=read(feature_root/'protocol.json');assert fp['data']==data and not fp['smoke']
    assert fp['brightness']==(.2 if a.condition=='brightness' else 1.)
    assert fp.get('center_occlusion',0)==(int(a.condition[9:])/100 if a.condition.startswith('occlusion') else 0)
    cleanp=read(scale/'features_v1/protocol.json')
    for k in ['checkpoint_sha256','source_hashes','preprocessing','features']:assert fp[k]==cleanp[k]
    fm={x['scene']:x for x in read(feature_root/'manifest.json')};scenes=data['scenes']['val'][:1] if a.smoke else data['scenes']['val'];seeds=[42] if a.smoke else [42,43,44]
    assert set(data['scenes']['val'])<=set(fm)
    cm={x['scene']:x for x in read(scale/'cache_v1/manifest.json')};noise_root=study/'audio_noise20_v1';nm={}
    if a.condition=='audio_noise20':
        np_=read(noise_root/'protocol.json');assert np_['data']==data and np_['snr_db']==20
        assert read(noise_root/'validation.json')['clean_spectrum_bitwise_replay_mismatches']==0
        nm={x['scene']:x for x in read(noise_root/'manifest.json')};assert set(nm)==set(data['scenes']['val'])
    a.out.mkdir(parents=True,exist_ok=False);write(a.out/'protocol.json',dict(code_sha=a.code_sha,condition=a.condition,smoke=a.smoke,scenes=scenes,seeds=seeds,queries_per_scene=2 if a.smoke else 64,arms=['U0','U1','U2','U3','U4'],methods=['visual','acoustic','fixed'],secondary='original frozen modality scales on Laplace fields, clean only',feature_protocol_sha256=sha(feature_root/'protocol.json'),head_protocol_sha256=sha(study/'heads_v1/protocol.json'),declared_protocol_sha256=sha(study/'protocol.json'),test_accessed=False))
    rows=[];rays=[];costs=[];spreads=[];replayed=0
    with torch.no_grad():
      for seed in seeds:
        models={};heads={};headpath=study/'heads_v1'/f'heads_{seed}.pt';assert sha(headpath)==fits[seed]['checkpoint_sha256'];hp=torch.load(headpath,weights_only=True)
        for mode,cls,stage in [('visual',FoundationStructureMatcher,'visual_joint'),('acoustic',StructureMatcher,'acoustic')]:
            path=scale/'scaling_fits_v1'/f'n112_budget1_{stage}_attempt1'/f'{mode}_{seed}.pt';assert sha(path)==audit['checkpoints'][f'{mode}_{seed}']['sha256']
            m=cls().cuda().eval().requires_grad_(False);m.load_state_dict(torch.load(path,weights_only=True));models[mode]=m
            h=RayScaleHead().cuda().eval();h.load_state_dict(hp['heads'][mode]);heads[mode]=h
        for sid in scenes:
            path=scale/'cache_v1'/f'{sid}.pt';assert sha(path)==cm[sid]['cache_sha256'];s=scene_to_gpu(torch.load(path,weights_only=True));path=feature_root/f'{sid}.pt';assert sha(path)==fm[sid]['sha256'];fv=torch.load(path,weights_only=True);assert fv['query_ids']==s['query_ids'];rgb=fv['visual_features'].cuda().float()
            spectrum=s['spectrum'].float()
            if a.condition=='audio_shuffle':spectrum=torch.roll(spectrum,-1,0)
            if a.condition=='audio_noise20':
                path=noise_root/f'{sid}.pt';assert sha(path)==nm[sid]['sha256'];n=torch.load(path,weights_only=True);assert n['query_ids']==s['query_ids'];spectrum=n['spectrum'].cuda().float()
            torch.cuda.synchronize();t=time.perf_counter();maps={m:full_descriptors(models[m],s,True) for m in models};torch.cuda.synchronize();costs.append(dict(seed=seed,scene=sid,component='map_descriptors',seconds=time.perf_counter()-t))
            visibility={'visual':visible_rays(s['hfov_deg'],device='cuda'),'acoustic':torch.ones(16,device='cuda')};weights={m:(.25+.75*s['candidates'][:,19:35])*visibility[m] for m in models}
            count=2 if a.smoke else len(s['query_ids'])
            for ids in torch.arange(count,device='cuda').split(8):
                mean={};bs={};hidden={};torch.cuda.synchronize();t=time.perf_counter()
                for m,obs in [('visual',rgb[ids]),('acoustic',spectrum[ids])]:hidden[m],mean[m]=frozen_observation(getattr(models[m],m),obs);bs[m]=heads[m](hidden[m])
                torch.cuda.synchronize();costs.append(dict(seed=seed,scene=sid,component='observation_and_scale_heads',queries=len(ids),seconds=time.perf_counter()-t))
                for arm in ['U0','U1','U2','U3','U4']:
                    fields={};torch.cuda.synchronize();t=time.perf_counter()
                    for m in models:
                        if arm=='U0':fields[m]=models[m].match(mean[m],maps[m],weights[m])
                        else:
                            b=uncertainty_control(bs[m],visibility[m],arm,hp['constants'][m]);fields[m]=laplace_field(mean[m],b,maps[m],weights[m])
                            # Labels first enter after means/scales/fields were computed.
                            for j,q in enumerate(ids.tolist()):rays.append(dict(seed=seed,scene=sid,query=s['query_ids'][q],arm=arm,modality=m,mean=mean[m][j].cpu(),scale=b[j].cpu(),target=s['target'][q].cpu(),target_hit=s['target_hit'][q].cpu(),visible=visibility[m].cpu()))
                        for j,q in enumerate(ids.tolist()):spreads.append(dict(seed=seed,scene=sid,query=s['query_ids'][q],arm=arm,modality=m,std=float(fields[m][j].std()),minimum=float(fields[m][j].min()),maximum=float(fields[m][j].max())))
                    torch.cuda.synchronize();costs.append(dict(seed=seed,scene=sid,arm=arm,component='fields_plus_ray_record_transfer',queries=len(ids),seconds=time.perf_counter()-t))
                    for policy in (['primary','original_scales'] if a.condition=='normal' and arm!='U0' else ['primary']):
                        ff={m:field*(models[m].log_scale.clamp(0,8).exp() if policy=='original_scales' else 1.) for m,field in fields.items()};ff['fixed']=.85*ff['visual']+.15*ff['acoustic']
                        for method,field in ff.items():
                            for j,q in enumerate(ids.tolist()):
                                rec=dict(seed=seed,scene=sid,query=s['query_ids'][q],arm=arm,method=method,policy=policy,**extended_pose_statistics(field[j],s['candidates'][:,:3],s['gt'][q],diverse=True));rows.append(rec)
                                if arm=='U0' and a.condition=='normal':
                                    old=reference[method,seed,sid,rec['query']]
                                    assert all(np.isclose(old[k],rec[k],rtol=1e-5,atol=2e-5) for k in ['joint','r1','fine','raw20','diverse20','nll']), 'U0 replay mismatch'
                                    replayed+=1
            print('DONE',seed,sid,flush=True)
      summary=[]
      for key in sorted({(x['arm'],x['method'],x['policy']) for x in rows}):
        rr=[x for x in rows if (x['arm'],x['method'],x['policy'])==key];summary.append(dict(arm=key[0],method=key[1],policy=key[2],queries=len(rr),mean={k:float(np.mean([x[k] for x in rr])) for k in ['joint','r1','fine','strict_05_15','strict_02_10','raw5','raw10','raw20','diverse20','nll','normalized_entropy']},reliability=reliability_statistics(rr)))
      torch.save(rays,a.out/'ray_records.pt')
      for name,value in [('predictions',rows),('summary',summary),('timing',costs),('field_spreads',spreads),('validation',dict(records=len(rows),ray_records=len(rays),u0_replay_checks=replayed,smoke=a.smoke))]:write(a.out/f'{name}.json',value)
      print('COMPLETE',flush=True)
if __name__=='__main__':main()

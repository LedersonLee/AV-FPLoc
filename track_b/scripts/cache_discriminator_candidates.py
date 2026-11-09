"""Generate actual OOF/full-Train F1 candidates; labels are appended afterward."""
import argparse, json, math
from pathlib import Path
import torch
from track_b.models.foundation import FoundationStructureMatcher
from track_b.models.foundation_interaction import FoundationInteractionMatcher
from track_b.models.structure import StructureMatcher, visible_rays
from track_b.models.discrimination import candidate_features
from track_b.data.folds import validate_training_fold
from track_b.scripts.train_structure import (read, sha, scene_to_gpu, full_descriptors,
    attach_visual_features, visual_input, diverse_cpu)


def main():
    p = argparse.ArgumentParser()
    for name in ('cache', 'features', 'visual_run', 'acoustic_run', 'out'):
        p.add_argument('--'+name.replace('_','-'), type=Path, required=True)
    p.add_argument('--interaction-run', type=Path)
    p.add_argument('--interaction-mode', choices=['attention','concat','separable'], default='attention')
    p.add_argument('--role', choices=['oof', 'development'], required=True)
    args = p.parse_args(); torch.set_num_threads(2)
    base = read(args.cache/'protocol.json')
    vp, ap = [read(r/'protocol.json') for r in (args.visual_run, args.acoustic_run)]
    assert vp['data']['scenes'] == ap['data']['scenes']
    assert vp['cache_protocol_sha256'] == ap['cache_protocol_sha256'] == sha(args.cache/'protocol.json')
    assert vp['visual_cache_protocol_sha256'] == sha(args.features/'protocol.json')
    assert vp['seeds'] == ap['seeds'] == [42, 43, 44]
    if args.role == 'oof':
        selected = validate_training_fold(base['scenes'], vp['scene_fold'], sha(args.cache/'protocol.json'))
        assert vp['scene_fold'] == ap['scene_fold'] and selected == vp['data']['scenes']
    else:
        assert vp['data']['scenes'] == base['scenes'] and 'scene_fold' not in vp
        selected = base['scenes']
    ip=None
    if args.interaction_run:
        ip=read(args.interaction_run/'protocol.json')
        assert ip['data']['scenes']==selected and ip['seeds']==vp['seeds']
        assert args.interaction_mode in ip['modes']
        assert ip['visual_cache_protocol_sha256']==sha(args.features/'protocol.json')
        assert ip['cache_manifest_sha256']==sha(args.cache/'manifest.json')
        if args.role=='oof':
            assert ip.get('scene_fold')==vp['scene_fold'], 'Joint model must be trained within the same OOF fold'
        for seed in vp['seeds']:
            for mode,run in [('visual',args.visual_run),('acoustic',args.acoustic_run)]:
                assert ip['pretrained_sha256'][f'{mode}_{seed}']==sha(run/f'{mode}_{seed}.pt')
            rec=read(args.interaction_run/f'{args.interaction_mode}_{seed}.json')
            assert rec['history'][-1]['step']==ip['steps']-1, 'Incomplete joint training'
    args.out.mkdir(parents=True, exist_ok=False)
    protocol = dict(role=args.role, scorer='independent fixed V.85/A.15',
        training_scenes=selected['train'], candidate_scenes=selected['val'],
        visual_protocol_sha256=sha(args.visual_run/'protocol.json'),
        acoustic_protocol_sha256=sha(args.acoustic_run/'protocol.json'),
        feature_protocol_sha256=sha(args.features/'protocol.json'),
        candidates='raw20 torch.topk and diverse20 score-ordered 1m/30deg suppression',
        feature_dimension=133, gt='labels/metrics only, never feature or proposal input',
        upstream_exposure='not ruled out; reused development method choices', test_accessed=False)
    if ip is not None:
        protocol.update(scorer='foundation interaction '+args.interaction_mode, interaction_protocol_sha256=sha(args.interaction_run/'protocol.json'), posterior='actual joint full field for proposals, support and posterior features')
    (args.out/'protocol.json').write_text(json.dumps(protocol, indent=2))
    manifest = {r['scene']:r['sha256'] for r in read(args.features/'manifest.json')}
    rows=[]; hashes={}
    for seed in vp['seeds']:
        vm, am = FoundationStructureMatcher().cuda().eval(), StructureMatcher().cuda().eval()
        for m, path in [(vm,args.visual_run/f'visual_{seed}.pt'), (am,args.acoustic_run/f'acoustic_{seed}.pt')]:
            m.load_state_dict(torch.load(path,weights_only=True)); hashes[str(path)] = sha(path)
        jm=None
        if ip is not None:
            jm=FoundationInteractionMatcher(args.interaction_mode).cuda().eval()
            path=args.interaction_run/f'{args.interaction_mode}_{seed}.pt'
            jm.load_state_dict(torch.load(path,weights_only=True));hashes[str(path)]=sha(path)
        with torch.no_grad():
            for sid in selected['val']:
                s = scene_to_gpu(torch.load(args.cache/f'{sid}.pt',weights_only=True))
                fpath=args.features/f'{sid}.pt'; assert sha(fpath)==manifest[sid]
                attach_visual_features(s,torch.load(fpath,weights_only=True))
                vf, af = full_descriptors(vm,s,True), full_descriptors(am,s,True)
                joint_cache=None
                if jm is not None:
                    joint_cache=jm.prepare_map(s['floorplan'],{'origin':s['origin'],'spacing':s['spacing'],'hfov_deg':s['hfov_deg']},s['candidates'])
                    vf,af=joint_cache['fp'],joint_cache['acoustic_fp']
                c=s['candidates']; mask=visible_rays(s['hfov_deg'],device='cuda'); weight=.25+.75*c[:,19:35]
                packs=[]
                for ids in torch.arange(len(s['query_ids']),device='cuda').split(8):
                    v=vm.visual(visual_input(s,ids)); a=am.acoustic(s['spectrum'][ids].float())
                    vs=vm.match(v,vf,weight*mask); ac=am.match(a,af,weight); fused=.85*vs+.15*ac
                    if jm is not None:
                        v,a,_,_=jm.encode_observations(visual_input(s,ids),s['spectrum'][ids].float())
                        vs=jm.match(v,vf,weight*mask);ac=jm.match(a,af,weight,log_scale=jm.acoustic_log_scale)
                        fused=jm.forward_cached(visual_input(s,ids),s['spectrum'][ids].float(),joint_cache)
                    for j,q in enumerate(ids.tolist()):
                        policies={'raw20':fused[j].topk(20).indices,
                            'diverse20':torch.tensor(diverse_cpu(fused[j].cpu().numpy(),c[:,:3].cpu().numpy()),device='cuda')}
                        for policy,idx in policies.items():
                            x,support=candidate_features(v[j],a[j],c,vf,af,vs[j],ac[j],idx,mask,score_field=fused[j] if jm is not None else None)
                            # GT is first accessed here, after all proposals/features are fixed.
                            delta=c[idx,:3]-s['gt'][q]; distance=delta[:,:2].norm(dim=-1)
                            yaw=torch.atan2(delta[:,2].sin(),delta[:,2].cos()).abs()
                            label=(distance<=1+1e-6)&(yaw<=math.pi/6+1e-6)
                            record=dict(scene=sid,query=s['query_ids'][q],seed=seed,policy=policy,
                                features=x.cpu(),candidate_indices=idx.cpu(),labels=label.cpu(),
                                distance=distance.cpu(),yaw=yaw.cpu(),support=support.cpu())
                            packs.append(record)
                            rows.append(dict(scene=sid,seed=seed,policy=policy,query=s['query_ids'][q],
                                oracle=float(label.any()),map=float(label[0]),support=float(label[support.argmax()])))
                path=args.out/f'{sid}_{seed}.pt';torch.save(packs,path)
                print('DONE',seed,sid,flush=True)
        del vm,am,jm
    summary={policy:{k:sum(r[k] for r in rows if r['policy']==policy)/sum(r['policy']==policy for r in rows)
        for k in ['oracle','map','support']} for policy in ['raw20','diverse20']}
    for name,obj in [('summary',summary),('rows',rows),('checkpoint_hashes',hashes)]:
        (args.out/f'{name}.json').write_text(json.dumps(obj,indent=2))
    print(json.dumps(summary),flush=True); print('COMPLETE',flush=True)

if __name__=='__main__':main()

"""Full-field structural pretraining/fine-tuning with explicit development controls."""
import argparse,hashlib,json,math,time
from pathlib import Path
import numpy as np
import torch
from track_b.models.structure import StructureMatcher,auxiliary_loss,visible_rays
from track_b.models.dense import dense_loss
from track_b.train_regimes import set_regime
from track_b.models.foundation import FoundationStructureMatcher
from track_b.data.folds import validate_training_fold
from track_b.data.scene_residency import SceneSequence
from track_b.data.training_subset import restrict_training_scenes


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def scene_to_gpu(scene):
    return {k:v.cuda() if isinstance(v,torch.Tensor) and k!='valid' else v for k,v in scene.items()}


def visual_input(scene,ids):
    return scene['visual_features'][ids].float() if 'visual_features' in scene else scene['rgb'][ids].float()/255


def attach_visual_features(scene,package):
    assert package['scene']==scene['scene'] and package['query_ids']==scene['query_ids'], 'Visual features/query IDs do not align'
    x=package['visual_features'];assert x.shape==(len(scene['query_ids']),1024,4,16) and torch.isfinite(x).all()
    scene['visual_features']=x.to(scene['rgb'].device)


def observe(model,scene,ids,mode):
    rgb=visual_input(scene,ids);spectrum=scene['spectrum'][ids].float()
    return model.observations(rgb,spectrum)


def full_descriptors(model,scene,residual):
    if not residual:return scene['candidates'][:,3:19]
    feature=model.floorplan(scene['floorplan']);cal={'origin':scene['origin'],'spacing':scene['spacing']}
    return torch.cat([model.fp_descriptors(feature,p,cal) for p in scene['candidates'].split(8192)])


def scores(model,v,a,fp,scene,mode):
    weight=.25+.75*scene['candidates'][:,19:35];mask=visible_rays(scene['hfov_deg'],device='cuda')
    sv=model.match(v,fp,weight*mask);sa=model.match(a,fp,weight)
    return sv if mode=='visual' else sa if mode=='acoustic' else .85*sv+.15*sa


def diverse_cpu(score,poses,k=20):
    order=np.argsort(-score,kind='stable');chosen=[]
    for i in order:
        if chosen:
            d=poses[chosen]-poses[i];angle=np.abs(np.arctan2(np.sin(d[:,2]),np.cos(d[:,2])))
            if np.any((np.linalg.norm(d[:,:2],axis=1)<=1)&(angle<=np.pi/6+1e-6)):continue
        chosen.append(int(i))
        if len(chosen)==k:break
    return chosen


def evaluate(model,scenes,mode,residual,interventions=True,diversity=True):
    rows=[];cost=[];model.eval()
    with torch.no_grad():
      for scene in scenes:
        torch.cuda.synchronize();start=time.perf_counter();fp=full_descriptors(model,scene,residual);torch.cuda.synchronize()
        cost.append(dict(scene=scene['scene'],fp_encoding_ms=(time.perf_counter()-start)*1000,fp_cache_bytes=fp.numel()*fp.element_size(),candidates=len(fp)))
        poses=scene['candidates'][:,:3];cpu_poses=poses.cpu().numpy();qcount=len(scene['gt'])
        for ids in torch.arange(qcount,device='cuda').split(8):
          rgb=visual_input(scene,ids);spec=scene['spectrum'][ids].float();v,a=model.observations(rgb,spec)
          pred=v if mode=='visual' else a;mask=visible_rays(scene['hfov_deg'],device='cuda') if mode=='visual' else torch.ones(16,device='cuda')
          gt=scene['gt'][ids];dxy=(poses[None,:,:2]-gt[:,None,:2]).square().sum(-1).sqrt()
          dyaw=torch.atan2(torch.sin(poses[None,:,2]-gt[:,None,2]),torch.cos(poses[None,:,2]-gt[:,None,2])).abs()
          ok=(dxy<=1+1e-6)&(dyaw<=math.pi/6+1e-6);nearest=(dxy.square()/.35**2+dyaw.square()/math.radians(15)**2).argmin(-1)
          arms=['normal','audio_shuffle','audio_zero','rgb_zero'] if interventions else ['normal']
          for arm in arms:
            va,aa=v,a
            if arm=='audio_shuffle':aa=model.acoustic(scene['spectrum'][(ids+1)%qcount].float())
            elif arm=='audio_zero':aa=model.acoustic(torch.zeros_like(spec))
            elif arm=='rgb_zero':va=model.visual(torch.zeros_like(rgb))
            pred=va if mode=='visual' else aa
            s=scores(model,va,aa,fp,scene,mode);idx=s.argmax(-1);raw=s.topk(20,dim=-1).indices;logp=s.log_softmax(-1)
            for j,q in enumerate(ids.tolist()):
                div=diverse_cpu(s[j].cpu().numpy(),cpu_poses) if diversity and arm=='normal' else []
                metric=dict(scene=scene['scene'],query=scene['query_ids'][q],arm=arm,joint=float(ok[j,idx[j]]),r1=float(dxy[j,idx[j]]<=1+1e-6),fine=float(dxy[j,idx[j]]<=.1+1e-6),
                            raw20=float(ok[j,raw[j]].any()),diverse20=float(ok[j,div].any()) if div else None,nll=float(-logp[j,nearest[j]]),entropy=float(-(logp[j].exp()*logp[j]).sum()),
                            oracle_joint=float(ok[j].any()),oracle_fine=float((dxy[j]<=.1+1e-6).any()),ray_mse=float(((pred[j]-scene['target'][q]).square()*mask).sum()/mask.sum()),
                            ray_mae_m=float(((torch.expm1(pred[j]*math.log(21))-torch.expm1(scene['target'][q]*math.log(21))).abs()*mask).sum()/mask.sum()))
                rows.append(metric)
    summary={}
    for arm in sorted({r['arm'] for r in rows}):
        rr=[r for r in rows if r['arm']==arm];summary[arm]={k:float(np.mean([r[k] for r in rr])) for k in ('joint','r1','fine','raw20','nll','entropy','ray_mse','ray_mae_m','oracle_joint','oracle_fine')}
        summary[arm]['diverse20']=float(np.mean([r['diverse20'] for r in rr])) if diversity and arm=='normal' else None
    return dict(summary=summary,rows=rows,cost=cost)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--modes',nargs='+',default=['visual','acoustic','fusion'],choices=['visual','acoustic','fusion']);ap.add_argument('--seeds',nargs='+',type=int,default=[42,43,44]);ap.add_argument('--arm',choices=['supervised','shuffled','dense_only'],default='supervised')
    ap.add_argument('--regime',choices=['frozen','partial','joint'],default='joint');ap.add_argument('--aux-steps',type=int,default=1500);ap.add_argument('--dense-steps',type=int,default=1000);ap.add_argument('--batch-size',type=int,default=8);ap.add_argument('--pretrained-dir',type=Path);ap.add_argument('--skip-sampling-steps',type=int,default=0);ap.add_argument('--visual-cache',type=Path);ap.add_argument('--aux-weight',type=float,default=10.);ap.add_argument('--fixed-fp',action='store_true');ap.add_argument('--scene-fold',type=Path);ap.add_argument('--train-scene-count',type=int);ap.add_argument('--save-initial-state',action='store_true');ap.add_argument('--code-sha',default='UNRECORDED');ap.add_argument('--scene-residency',choices=['gpu','cpu'],default='gpu');ap.add_argument('--fp-anchor-weight',type=float,default=0.);args=ap.parse_args()
    assert args.fp_anchor_weight>=0 and not (args.fixed_fp and args.fp_anchor_weight), 'FP anchor requires learned descriptors'
    if args.visual_cache and args.pretrained_dir:
        prior_protocol=read(args.pretrained_dir/'protocol.json')
        assert prior_protocol.get('visual_cache_protocol_sha256')==sha(args.visual_cache/'protocol.json'), 'Pretrained adapter must use the identical foundation cache'
    if args.pretrained_dir:assert args.aux_steps==0 and args.arm=='supervised', 'Matched continuation requires zero warmup and supervised loss'
    args.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.4);torch.backends.cudnn.benchmark=False
    data_protocol=read(args.cache/'protocol.json');manifest=read(args.cache/'manifest.json');assert len(manifest)==sum(map(len,data_protocol['scenes'].values())),'Cache incomplete'
    effective_scenes=validate_training_fold(data_protocol['scenes'],read(args.scene_fold),sha(args.cache/'protocol.json')) if args.scene_fold else data_protocol['scenes']
    if args.train_scene_count is not None:
        assert not args.scene_fold, 'Do not mix OOF folds and scaling prefixes'
        effective_scenes=restrict_training_scenes(effective_scenes,args.train_scene_count)
    if args.pretrained_dir:
        assert read(args.pretrained_dir/'protocol.json')['data']['scenes']==effective_scenes, 'Pretrained weights saw a different fold'
    protocol=dict(arm=args.arm,encoder_regime=args.regime,modes=args.modes,seeds=args.seeds,aux_steps=args.aux_steps,dense_steps=args.dense_steps,batch_size=args.batch_size,
        cache_protocol_sha256=sha(args.cache/'protocol.json'),cache_manifest_sha256=sha(args.cache/'manifest.json'),data=data_protocol,aux_weight=args.aux_weight,fixed_fp=args.fixed_fp,fp_anchor_weight=args.fp_anchor_weight,lr=.0003,
        objective='continuous FP boundary ray MSE + exact full-grid soft CE; labels are training-only',
        controls='shuffled permutes target geometry within scene; dense_only uses all updates for dense loss without auxiliary term',
        metric='NLL at nearest valid SE2 grid candidate; fine tolerance1e-6m',evidence='Exploratory development; no test use',init='scratch, same seed for every mode/arm')
    if args.train_scene_count is not None:protocol.update(data={**data_protocol,'scenes':effective_scenes},training_scene_limit=args.train_scene_count,training_scene_selection='prefix of frozen SHA256-ordered Train cohort; full Val unchanged')
    if args.scene_fold:protocol.update(data={**data_protocol,'scenes':effective_scenes},scene_fold_sha256=sha(args.scene_fold),scene_fold=read(args.scene_fold))
    if args.pretrained_dir:
        protocol.update(init='archived common pretrained weights, fresh AdamW for every regime',pretrained_sha256={f'{mode}_{seed}':sha(args.pretrained_dir/f'{mode}_{seed}_pretrained.pt') for mode in args.modes for seed in args.seeds},skip_sampling_steps=args.skip_sampling_steps)
    if args.visual_cache:
        foundation_protocol=read(args.visual_cache/'protocol.json');foundation_manifest=read(args.visual_cache/'manifest.json')
        assert foundation_protocol['data']==data_protocol and not foundation_protocol['smoke'] and len(foundation_manifest)==len(manifest)
        protocol.update(init=('archived foundation adapter, fresh AdamW' if args.pretrained_dir else 'fresh Track B adapter/FP weights, frozen separately sourced upstream RGB encoder'),visual_representation='frozen upstream foundation features + fresh trainable adapter; no photometric intervention in feature space',visual_cache_protocol_sha256=sha(args.visual_cache/'protocol.json'),upstream=foundation_protocol,foundation_cost_included_in_training_seconds=False)
    protocol['scene_residency']=args.scene_residency
    (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2));scenes={}
    for part,ids in effective_scenes.items():
        scenes[part]=[torch.load(args.cache/f'{s}.pt',weights_only=True,map_location='cpu') for s in ids]
        if args.scene_residency=='gpu':scenes[part]=[scene_to_gpu(s) for s in scenes[part]]
    if args.visual_cache:
        feature_hashes={r['scene']:r['sha256'] for r in foundation_manifest}
        for part in scenes:
            for scene in scenes[part]:
                path=args.visual_cache/f"{scene['scene']}.pt";assert sha(path)==feature_hashes[scene['scene']]
                attach_visual_features(scene,torch.load(path,weights_only=True,map_location='cpu'))
    if args.scene_residency=='cpu':scenes={part:SceneSequence(items) for part,items in scenes.items()}
    print('DATA_READY',flush=True);mean_target=torch.cat([s['target'] for s in scenes['train']]).mean(0)
    prior=[]
    for s in scenes['val']:
        error=(s['target']-mean_target).square();mask=visible_rays(s['hfov_deg'],device='cuda')
        prior.append(dict(scene=s['scene'],acoustic_ray_mse=float(error.mean()),visual_ray_mse=float((error*mask).sum(-1).mean()/mask.sum())))
    (args.out/'constant_structure_prior.json').write_text(json.dumps(prior,indent=2));all_results=[]
    for seed in args.seeds:
      for mode in args.modes:
        torch.manual_seed(seed);rng=np.random.default_rng(seed);model=(FoundationStructureMatcher() if args.visual_cache else StructureMatcher()).cuda();opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001)
        if args.pretrained_dir:model.load_state_dict(torch.load(args.pretrained_dir/f'{mode}_{seed}_pretrained.pt',weights_only=True))
        initial_hash=None
        if args.save_initial_state:
            initial_path=args.out/f'{mode}_{seed}_initial.pt';torch.save(model.state_dict(),initial_path);initial_hash=sha(initial_path)
        permutations={s['scene']:torch.tensor(rng.permutation(len(s['gt'])),device='cuda') for s in scenes['train']}
        rng=np.random.default_rng(seed);history=[];aux_eval_seconds=0.;start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        for _ in range(args.skip_sampling_steps):
            skipped_scene=scenes['train'][int(rng.integers(len(scenes['train'])))];rng.integers(len(skipped_scene['gt']),size=args.batch_size)
        total=args.aux_steps+args.dense_steps
        for step in range(total):
            if (args.arm=='dense_only' and step==0) or (args.arm!='dense_only' and step==args.aux_steps):
                active=set_regime(model,args.regime)
                print(json.dumps(dict(mode=mode,seed=seed,regime=args.regime,trainable_parameters=active)),flush=True)
            model.train();s=scenes['train'][int(rng.integers(len(scenes['train'])))];ids=torch.tensor(rng.integers(len(s['gt']),size=args.batch_size),device='cuda')
            v,a=observe(model,s,ids,mode);target_ids=permutations[s['scene']][ids] if args.arm=='shuffled' else ids
            aux=auxiliary_loss(v,a,s['target'][target_ids],mode);opt.zero_grad()
            dense_phase=args.arm=='dense_only' or step>=args.aux_steps
            if dense_phase:
                fp=full_descriptors(model,s,not args.fixed_fp);score=scores(model,v,a,fp,s,mode);ce=dense_loss(score,s['candidates'][:,:3],s['gt'][ids])
                fp_anchor=(fp-s['candidates'][:,3:19]).square().mean() if args.fp_anchor_weight else aux*0
                loss=ce+(args.aux_weight*aux if args.arm!='dense_only' else 0)+args.fp_anchor_weight*fp_anchor
            else:ce=aux*0;fp_anchor=aux*0;loss=aux
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
            if step%100==0 or step==total-1:
                rec=dict(step=step,loss=float(loss),aux=float(aux),fp_anchor=float(fp_anchor),dense_ce=float(ce),elapsed_s=time.perf_counter()-start,gradient_l1={name:sum(float(p.grad.abs().sum()) for p in branch.parameters() if p.grad is not None) for name,branch in [('visual',model.visual),('acoustic',model.acoustic),('floorplan',model.floorplan),('fp_head',model.fp_head)]});history.append(rec);print(json.dumps(dict(seed=seed,mode=mode,**rec)),flush=True)
            if step+1==args.aux_steps and args.arm!='dense_only':
                torch.save(model.state_dict(),args.out/f'{mode}_{seed}_pretrained.pt')
                torch.cuda.synchronize();aux_start=time.perf_counter()
                aux_eval=evaluate(model,scenes['val'],mode,False,interventions=False,diversity=False)
                torch.cuda.synchronize();aux_eval_seconds+=time.perf_counter()-aux_start
                (args.out/f'{mode}_{seed}_aux_eval.json').write_text(json.dumps(aux_eval,indent=2))
        torch.cuda.synchronize();train_seconds=time.perf_counter()-start;peak=torch.cuda.max_memory_allocated();torch.save(model.state_dict(),args.out/f'{mode}_{seed}.pt')
        train_eval=evaluate(model,scenes['train'],mode,not args.fixed_fp,interventions=False,diversity=False)
        val_eval=evaluate(model,scenes['val'],mode,not args.fixed_fp,interventions=not bool(args.visual_cache),diversity=True)
        record=dict(execution_code_sha=args.code_sha,initial_state_sha256=initial_hash,seed=seed,mode=mode,arm=args.arm,history=history,training_and_aux_eval_seconds=train_seconds,training_seconds=train_seconds-aux_eval_seconds,aux_eval_seconds=aux_eval_seconds,peak_vram_bytes=peak,parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),encoder_regime=args.regime,train=train_eval,val=val_eval)
        (args.out/f'{mode}_{seed}.json').write_text(json.dumps(record,indent=2));aggregate={k:v for k,v in record.items() if k not in ('train','val','history')};aggregate.update(train=train_eval['summary'],val=val_eval['summary'])
        all_results.append(aggregate);(args.out/'summary.json').write_text(json.dumps(all_results,indent=2));print(json.dumps(aggregate),flush=True)
        del model,opt
    print('COMPLETE',flush=True)

if __name__=='__main__':main()

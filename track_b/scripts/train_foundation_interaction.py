"""Common-initialization full-bank attention/concat/separable training comparison."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from track_b.models.foundation_interaction import FoundationInteractionMatcher
from track_b.models.foundation import FoundationStructureMatcher
from track_b.models.foundation_regimes import set_foundation_regime
from track_b.models.structure import StructureMatcher,visible_rays
from track_b.models.structure import auxiliary_loss
from track_b.models.dense import dense_loss
from track_b.scripts.train_structure import scene_to_gpu,attach_visual_features,visual_input
from track_b.eval.metrics import pose_statistics
from track_b.data.scene_residency import SceneSequence
from track_b.data.folds import validate_training_fold

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def predict(m,s,ids,arm='normal',aux=False):
 rgb=visual_input(s,ids);spec=s['spectrum'][ids].float()
 if arm=='audio_shuffle':spec=s['spectrum'][(ids+1)%len(s['gt'])].float()
 elif arm=='low_light':rgb=s['lowlight_features'][ids].float()
 return m(rgb,spec,s['floorplan'],{'origin':s['origin'],'spacing':s['spacing'],'hfov_deg':s['hfov_deg']},s['candidates'],return_aux=aux)
def evaluate(m,scenes):
 m.eval();rows=[];cost=[]
 with torch.no_grad():
  for s in scenes:
   for ids in torch.arange(len(s['gt']),device='cuda').split(8):
    for arm in ('normal','audio_shuffle','low_light'):
     torch.cuda.synchronize();start=time.perf_counter();field=predict(m,s,ids,arm);torch.cuda.synchronize();cost.append(dict(scene=s['scene'],arm=arm,queries=len(ids),seconds=time.perf_counter()-start))
     for j,q in enumerate(ids.tolist()):rows.append(dict(scene=s['scene'],query=s['query_ids'][q],arm=arm,**pose_statistics(field[j],s['candidates'][:,:3],s['gt'][q],diverse=arm=='normal')))
 summary={arm:{k:float(np.mean([r[k] for r in rows if r['arm']==arm])) for k in ('joint','r1','fine','raw20','nll','entropy')} for arm in ('normal','audio_shuffle','low_light')};summary['normal']['diverse20']=float(np.mean([r['diverse20'] for r in rows if r['arm']=='normal']))
 return dict(summary=summary,rows=rows,cost=cost)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--visual-run',type=Path,required=True);ap.add_argument('--acoustic-run',type=Path,required=True);ap.add_argument('--visual-cache',type=Path,required=True);ap.add_argument('--lowlight-cache',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--scene-fold',type=Path);ap.add_argument('--scene-residency',choices=['gpu','cpu'],default='gpu');ap.add_argument('--resume',action='store_true');ap.add_argument('--regime',choices=['frozen','partial','joint'],default='joint');ap.add_argument('--initial-reference',type=Path);ap.add_argument('--code-sha',default='UNRECORDED');ap.add_argument('--steps',type=int,default=1000);ap.add_argument('--seeds',type=int,nargs='+',default=[42,43,44]);ap.add_argument('--modes',nargs='+',default=['attention','concat','separable']);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=a.resume);torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.cuda.set_per_process_memory_fraction(.5)
 base_data=read(a.cache/'protocol.json');data=dict(base_data);fold=None
 if a.scene_fold:
  fold=read(a.scene_fold);data['scenes']=validate_training_fold(base_data['scenes'],fold,sha(a.cache/'protocol.json'))
  assert read(a.visual_run/'protocol.json').get('scene_fold')==read(a.acoustic_run/'protocol.json').get('scene_fold')==fold, 'Precursor fold mismatch'
 protocol=dict(data=data,steps=a.steps,seeds=a.seeds,modes=a.modes,batch=8,lr=.0003,weight_decay=.0001,visual_aux_weight=1000,acoustic_aux_weight=10,map_anchor_weight=1000,skip_sampling_steps=1500,initialization='independent anchored visual and acoustic checkpoints; separate FP paths; zero final interaction outputs preserve exact F1; same seeded new modules and fresh AdamW',pretrained_sha256={f'{mode}_{seed}':sha(run/f'{mode}_{seed}.pt') for mode,run in [('visual',a.visual_run),('acoustic',a.acoustic_run)] for seed in a.seeds},visual_cache_protocol_sha256=sha(a.visual_cache/'protocol.json'),lowlight_cache_protocol_sha256=sha(a.lowlight_cache/'protocol.json'),cache_manifest_sha256=sha(a.cache/'manifest.json'),input_controls='within-scene cyclic audio shift; separately re-extracted RGB brightness0.2 features; no feature scaling',selection='final scheduled update only; no epoch selection',evidence='Exploratory reused development scenes',cost='equal parameter count and updates, not equal measured FLOPs; each mode uses all head parameters')
 if fold is not None:protocol['scene_fold']=fold
 if a.scene_residency=='cpu':protocol['scene_residency']='cpu'
 assert read(a.visual_run/'protocol.json')['data']['scenes']==read(a.acoustic_run/'protocol.json')['data']['scenes']==data['scenes'], 'Precursor effective cohort mismatch'
 assert read(a.visual_run/'protocol.json')['cache_protocol_sha256']==sha(a.cache/'protocol.json')
 assert read(a.acoustic_run/'protocol.json')['cache_protocol_sha256']==sha(a.cache/'protocol.json')
 assert read(a.visual_run/'protocol.json')['visual_cache_protocol_sha256']==sha(a.visual_cache/'protocol.json')
 assert read(a.lowlight_cache/'protocol.json')['data']==base_data and read(a.lowlight_cache/'protocol.json')['brightness']==.2
 if a.regime!='joint':protocol.update(encoder_regime=a.regime,regime_definition='frozen: precursor visual/acoustic/dual-FP frozen; partial: final visual/acoustic descriptor Linear and final dual-FP residual Linear train; both scales and all new interaction heads train in every regime')
 if a.initial_reference:protocol['initial_reference_sha256']={f'{mode}_{seed}':sha(a.initial_reference/f'{mode}_{seed}_initial.pt') for mode in a.modes for seed in a.seeds}
 if a.resume:assert read(a.out/'protocol.json')==json.loads(json.dumps(protocol)), 'Resume protocol differs'
 else:(a.out/'protocol.json').write_text(json.dumps(protocol,indent=2))
 scenes={part:[(scene_to_gpu(torch.load(a.cache/f'{sid}.pt',weights_only=True)) if a.scene_residency=='gpu' else torch.load(a.cache/f'{sid}.pt',weights_only=True,map_location='cpu')) for sid in ids] for part,ids in data['scenes'].items()};results=read(a.out/'summary.json') if a.resume and (a.out/'summary.json').exists() else []
 for part in scenes:
  for scene in scenes[part]:
   path=a.visual_cache/f"{scene['scene']}.pt";manifest={x['scene']:x['sha256'] for x in read(a.visual_cache/'manifest.json')};assert sha(path)==manifest[scene['scene']];attach_visual_features(scene,torch.load(path,weights_only=True))
   if part=='val':
    path=a.lowlight_cache/f"{scene['scene']}.pt";manifest={x['scene']:x['sha256'] for x in read(a.lowlight_cache/'manifest.json')};assert sha(path)==manifest[scene['scene']];package=torch.load(path,weights_only=True);assert package['query_ids']==scene['query_ids'];scene['lowlight_features']=package['visual_features'].to('cuda' if a.scene_residency=='gpu' else 'cpu')
 training_scenes=SceneSequence(scenes['train']) if a.scene_residency=='cpu' else scenes['train']
 validation_scenes=SceneSequence(scenes['val']) if a.scene_residency=='cpu' else scenes['val']
 for seed in a.seeds:
  for mode in a.modes:
   if a.resume and (a.out/f'{mode}_{seed}.json').exists():
    done=read(a.out/f'{mode}_{seed}.json');assert done['history'][-1]['step']==a.steps-1 and (a.out/f'{mode}_{seed}.pt').exists() and (a.out/f'{mode}_{seed}_initial.pt').exists();assert sum(x['mode']==mode and x['seed']==seed for x in results)==1
    print('RESUME_SKIP_COMPLETED',mode,seed,flush=True);continue
   torch.manual_seed(seed);m=FoundationInteractionMatcher(mode).cuda()
   vm=FoundationStructureMatcher().cuda();vm.load_state_dict(torch.load(a.visual_run/f'visual_{seed}.pt',weights_only=True));am=StructureMatcher().cuda();am.load_state_dict(torch.load(a.acoustic_run/f'acoustic_{seed}.pt',weights_only=True));m.initialize_independent(vm,am);del vm,am
   with torch.no_grad():
    for stored_scene in scenes['train']:
     scene=scene_to_gpu(stored_scene) if a.scene_residency=='cpu' else stored_scene
     cal={'origin':scene['origin'],'spacing':scene['spacing']};_,anchor=m.map_descriptors(scene['floorplan'],cal,scene['candidates']);stored_scene['acoustic_anchor']=anchor.detach().cpu() if a.scene_residency=='cpu' else anchor.detach()
   initial={k:v.detach().cpu() for k,v in m.state_dict().items()}
   initial_path=a.out/f'{a.modes[0]}_{seed}_initial.pt'
   if initial_path.exists():
    previous=torch.load(initial_path,weights_only=True,map_location='cpu');assert all(torch.equal(initial[k],previous[k]) for k in initial)
   if a.initial_reference:
    reference=torch.load(a.initial_reference/f'{mode}_{seed}_initial.pt',weights_only=True,map_location='cpu');assert set(initial)==set(reference) and all(torch.equal(initial[k],reference[k]) for k in initial), 'Regime comparison initial state differs';del reference
   active=set_foundation_regime(m,a.regime);frozen_before={n:p.detach().cpu().clone() for n,p in m.named_parameters() if not p.requires_grad}
   del initial
   opt=torch.optim.AdamW(m.parameters(),lr=.0003,weight_decay=.0001);rng=np.random.default_rng(seed)
   for _ in range(1500):s=scenes['train'][int(rng.integers(len(scenes['train'])))];rng.integers(len(s['gt']),size=8)
   # Serialized initial state records identical initialization across modes.
   torch.save(m.state_dict(),a.out/f'{mode}_{seed}_initial.pt');history=[];torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
   for step in range(a.steps):
    m.train();s=training_scenes[int(rng.integers(len(training_scenes)))];ids=torch.tensor(rng.integers(len(s['gt']),size=8),device='cuda');opt.zero_grad();field,v,au=predict(m,s,ids,aux=True);ce=dense_loss(field,s['candidates'][:,:3],s['gt'][ids]);mask=visible_rays(s['hfov_deg'],device='cuda');vl=((v-s['target'][ids]).square()*mask).sum(-1).mean()/mask.sum();al=(au-s['target'][ids]).square().mean();aux=1000*vl+10*al
    vf,af=m.map_descriptors(s['floorplan'],{'origin':s['origin'],'spacing':s['spacing']},s['candidates']);anchor=(vf-s['candidates'][:,3:19]).square().mean()+(af-s['acoustic_anchor']).square().mean();loss=ce+aux+1000*anchor;loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5.);opt.step()
    if step%100==0 or step==a.steps-1:
     row=dict(step=step,loss=float(loss),dense_ce=float(ce),aux=float(aux),visual_aux=float(vl),acoustic_aux=float(al),map_anchor=float(anchor),gradient_l1={name:sum(float(p.grad.abs().sum()) for p in branch.parameters() if p.grad is not None) for name,branch in [('visual',m.visual),('acoustic',m.acoustic),('visual_fp',m.floorplan),('acoustic_fp',m.acoustic_floorplan),('visual_attention',m.v_retrieval),('acoustic_attention',m.a_retrieval)]},elapsed_seconds=time.perf_counter()-start);history.append(row);print(json.dumps(dict(mode=mode,seed=seed,**row)),flush=True)
   assert all(torch.equal(frozen_before[n],p.detach().cpu()) for n,p in m.named_parameters() if n in frozen_before), 'Frozen parameter changed'
   torch.cuda.synchronize();seconds=time.perf_counter()-start;peak=torch.cuda.max_memory_allocated();torch.save(m.state_dict(),a.out/f'{mode}_{seed}.pt');evaluation=evaluate(m,validation_scenes);rec=dict(execution_code_sha=a.code_sha,encoder_regime=a.regime,trainable_parameters=sum(active.values()),active_parameters=active,frozen_parameters_bitwise_unchanged=True,mode=mode,seed=seed,parameters=sum(p.numel() for p in m.parameters()),training_seconds=seconds,peak_vram_bytes=peak,history=history,val=evaluation);(a.out/f'{mode}_{seed}.json').write_text(json.dumps(rec,indent=2));results.append({k:v for k,v in rec.items() if k not in ('history','val')}|{'val':evaluation['summary']});(a.out/'summary.json').write_text(json.dumps(results,indent=2));print(json.dumps(results[-1]),flush=True);del m,opt,field,v,au,loss,ce,aux,vf,af,anchor,vl,al;torch.cuda.empty_cache()
 print('COMPLETE',flush=True)
if __name__=='__main__':main()

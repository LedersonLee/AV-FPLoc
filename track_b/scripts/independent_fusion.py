"""Independent V/A full-field fusion with leave-one-development-scene-out calibration."""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
import torch
from track_b.models.structure import StructureMatcher
from track_b.models.foundation import FoundationStructureMatcher
from track_b.scripts.train_structure import scene_to_gpu,full_descriptors,scores,attach_visual_features,visual_input
from track_b.eval.metrics import pose_statistics

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--run',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--visual-run',type=Path);ap.add_argument('--visual-cache',type=Path);ap.add_argument('--eval-visual-cache',type=Path);ap.add_argument('--selection-reference',type=Path);ap.add_argument('--fixed-only',action='store_true',help='Evaluate declared V/A/fixed fusion and shuffle only; do not select calibration parameters');a=ap.parse_args()
 assert not (a.fixed_only and a.selection_reference), 'Fixed-only evaluation cannot use a calibration selection'
 assert bool(a.eval_visual_cache)==bool(a.selection_reference)
 if a.eval_visual_cache:assert a.visual_cache and read(a.eval_visual_cache/'protocol.json')['data']==read(a.visual_cache/'protocol.json')['data']
 assert bool(a.visual_run)==bool(a.visual_cache)
 a.out.mkdir(exist_ok=False,parents=True);torch.set_num_threads(2)
 protocol=read(a.run/'protocol.json');scene_ids=protocol['data']['scenes']['val'];seeds=protocol['seeds'];weights=[0.,.85,1.] if a.fixed_only else [0.,.15,.5,.85,1.];temps=[1.] if a.fixed_only else [.25,1.,4.]
 visual_run=a.visual_run or a.run
 if a.visual_run:
  vp=read(visual_run/'protocol.json');assert vp['data']['scenes']==protocol['data']['scenes'] and vp['seeds']==protocol['seeds'], 'Independent modality models must share the effective training/evaluation cohort and seeds';assert vp['cache_protocol_sha256']==protocol['cache_protocol_sha256'];assert vp['visual_cache_protocol_sha256']==sha(a.visual_cache/'protocol.json');assert not vp.get('fixed_fp',False)
 setup=dict(evaluation_visual_cache_protocol_sha256=sha((a.eval_visual_cache or a.visual_cache)/'protocol.json') if a.visual_cache else None,calibration_reference_sha256=sha(a.selection_reference/'selection.json') if a.selection_reference else None,visual_weights=weights,temperatures=temps,selection='none: prespecified fixed weights and temperature' if a.fixed_only else 'minimum nearest-grid NLL on remaining development scenes, pooled seeds; tie in declared parameter order',fixed_only=a.fixed_only,fixed=dict(visual_weight=.85,temperature=1.),scene_ids=scene_ids,seeds=seeds,evidence='Exploratory: reused development scenes, not full-pipeline independence',checkpoint_hashes={f'{m}_{s}':sha((visual_run if m=='visual' else a.run)/f'{m}_{s}.pt') for m in ('visual','acoustic') for s in seeds},cache_manifest_sha256=sha(a.cache/'manifest.json'))
 (a.out/'protocol.json').write_text(json.dumps(setup,indent=2));models={}
 for seed in seeds:
  models[seed]={}
  for mode in ('visual','acoustic'):
   m=(FoundationStructureMatcher() if mode=='visual' and a.visual_run else StructureMatcher()).cuda().eval();m.load_state_dict(torch.load((visual_run if mode=='visual' else a.run)/f'{mode}_{seed}.pt',weights_only=True));models[seed][mode]=m
 def load_scene(sid):
  scene=scene_to_gpu(torch.load(a.cache/f'{sid}.pt',weights_only=True))
  if a.visual_cache:
   feature_cache=a.eval_visual_cache or a.visual_cache;path=feature_cache/f'{sid}.pt';manifest={r['scene']:r['sha256'] for r in read(feature_cache/'manifest.json')};assert sha(path)==manifest[sid];attach_visual_features(scene,torch.load(path,weights_only=True))
  return scene
 rows=[];replay=[]
 with torch.no_grad():
  for sid in scene_ids:
   scene=load_scene(sid);poses=scene['candidates'][:,:3]
   for seed,mm in models.items():
    fp={m:full_descriptors(mm[m],scene,True) for m in mm};reference={m:{r['query']:r for r in read((visual_run if m=='visual' else a.run)/f'{m}_{seed}.json')['val']['rows'] if r['scene']==sid and r['arm']=='normal'} for m in mm}
    for ids in torch.arange(len(scene['gt']),device='cuda').split(8):
     rgb=scene['rgb'][ids].float()/255;spec=scene['spectrum'][ids].float();fields={}
     for mode,m in mm.items():
      v=m.visual(visual_input(scene,ids)) if mode=='visual' else m.visual(rgb);au=m.acoustic(spec);fields[mode]=scores(m,v,au,fp[mode],scene,mode)
     gt=scene['gt'][ids];dist=(poses[None,:,:2]-gt[:,None,:2]).norm(dim=-1);yaw=torch.atan2(torch.sin(poses[None,:,2]-gt[:,None,2]),torch.cos(poses[None,:,2]-gt[:,None,2])).abs();nearest=(dist.square()/.35**2+yaw.square()/math.radians(15)**2).argmin(-1);ok=(dist<=1+1e-6)&(yaw<=math.pi/6+1e-6)
     for weight in weights:
      combined=weight*fields['visual']+(1-weight)*fields['acoustic'];top=combined.argmax(-1);raw=combined.topk(20,dim=-1).indices
      for temp in temps:
       logits=combined/temp;logp=logits.log_softmax(-1);nll=-logp.gather(1,nearest[:,None])[:,0];entropy=-(logp.exp()*logp).sum(-1)
       for j,q in enumerate(ids.tolist()):
        rec=dict(scene=sid,seed=seed,query=scene['query_ids'][q],weight=weight,temperature=temp,joint=float(ok[j,top[j]]),r1=float(dist[j,top[j]]<=1+1e-6),fine=float(dist[j,top[j]]<=.1+1e-6),raw20=float(ok[j,raw[j]].any()),nll=float(nll[j]),entropy=float(entropy[j]));rows.append(rec)
        if temp==1 and weight in (0,1) and not (weight==1 and a.eval_visual_cache):
         old=reference['visual' if weight==1 else 'acoustic'][rec['query']]
         assert all(np.isclose(old[k],rec[k],atol=1e-5,rtol=1e-5) for k in ('joint','r1','fine','raw20','nll'))
         replay.append((sid,seed,q,weight))
   print('SWEEP',sid,flush=True)
 selected={}
 for sid in ([] if a.fixed_only else scene_ids):
  options=[]
  for weight in weights:
   for temp in temps:
    rr=[r for r in rows if r['scene']!=sid and r['weight']==weight and r['temperature']==temp];options.append(dict(weight=weight,temperature=temp,calibration_nll=float(np.mean([r['nll'] for r in rr]))))
  selected[sid]=min(options,key=lambda r:r['calibration_nll'])
 if a.selection_reference:
  ref=read(a.selection_reference/'protocol.json');assert ref['checkpoint_hashes']==setup['checkpoint_hashes'] and ref['scene_ids']==scene_ids
  selected=read(a.selection_reference/'selection.json')
 (a.out/'selection.json').write_text(json.dumps(selected,indent=2));(a.out/'sweep.json').write_text(json.dumps(rows));final=[]
 with torch.no_grad():
  for sid in scene_ids:
   scene=load_scene(sid)
   for seed,mm in models.items():
    fp={m:full_descriptors(mm[m],scene,True) for m in mm}
    for ids in torch.arange(len(scene['gt']),device='cuda').split(8):
     rgb=scene['rgb'][ids].float()/255;spec=scene['spectrum'][ids].float();v=mm['visual'].visual(visual_input(scene,ids));au=mm['visual'].acoustic(spec);vf=scores(mm['visual'],v,au,fp['visual'],scene,'visual');v,au=mm['acoustic'].observations(rgb,spec);af=scores(mm['acoustic'],v,au,fp['acoustic'],scene,'acoustic');sh=mm['acoustic'].acoustic(scene['spectrum'][(ids+1)%len(scene['gt'])].float());shf=scores(mm['acoustic'],v,sh,fp['acoustic'],scene,'acoustic')
     arms=[('visual',1.,1.,af),('acoustic',0.,1.,af),('fixed_audio_shuffle',.85,1.,shf),('fixed',.85,1.,af)]
     if not a.fixed_only:arms += [('calibrated',selected[sid]['weight'],selected[sid]['temperature'],af),('calibrated_audio_shuffle',selected[sid]['weight'],selected[sid]['temperature'],shf)]
     for method,weight,temp,field in arms:
      combined=(weight*vf+(1-weight)*field)/temp
      for j,q in enumerate(ids.tolist()):final.append(dict(scene=sid,seed=seed,query=scene['query_ids'][q],method=method,**pose_statistics(combined[j],scene['candidates'][:,:3],scene['gt'][q],diverse=True)))
   print('FINAL',sid,flush=True)
 summary={method:{k:float(np.mean([r[k] for r in final if r['method']==method])) for k in ('joint','r1','fine','raw20','diverse20','nll','entropy')} for method in dict.fromkeys(r['method'] for r in final)}
 rng=np.random.default_rng(20261109);paired=[]
 for left,right in [('calibrated','fixed'),('calibrated','calibrated_audio_shuffle'),('fixed','visual'),('calibrated','visual'),('fixed','fixed_audio_shuffle')]:
  if left not in summary or right not in summary:continue
  d=np.array([np.mean([r['joint'] for r in final if r['scene']==sid and r['method']==left])-np.mean([r['joint'] for r in final if r['scene']==sid and r['method']==right]) for sid in scene_ids]);boot=d[rng.integers(0,len(d),(10000,len(d)))].mean(1)*100;paired.append(dict(left=left,right=right,delta_pp=100*d.mean(),ci95_pp=np.quantile(boot,[.025,.975]).tolist(),unit='scene, seeds averaged'))
 (a.out/'predictions.json').write_text(json.dumps(final));result=dict(summary=summary,paired=paired,unimodal_replay_checks=len(replay),selection=selected);(a.out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True);print('COMPLETE',flush=True)
if __name__=='__main__':main()

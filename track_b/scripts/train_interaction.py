"""Common-initialization full-bank attention/concat/separable training comparison."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from track_b.models.interaction import InteractionMatcher
from track_b.models.structure import auxiliary_loss
from track_b.models.dense import dense_loss
from track_b.scripts.train_structure import scene_to_gpu
from track_b.eval.metrics import pose_statistics

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def predict(m,s,ids,arm='normal',aux=False):
 rgb=s['rgb'][ids].float()/255;spec=s['spectrum'][ids].float()
 if arm=='audio_shuffle':spec=s['spectrum'][(ids+1)%len(s['gt'])].float()
 elif arm=='low_light':rgb=rgb*.2
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
 ap=argparse.ArgumentParser();ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--pretrained',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--steps',type=int,default=1000);ap.add_argument('--seeds',type=int,nargs='+',default=[42,43,44]);ap.add_argument('--modes',nargs='+',default=['attention','concat','separable']);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.cuda.set_per_process_memory_fraction(.5)
 data=read(a.cache/'protocol.json');protocol=dict(data=data,steps=a.steps,seeds=a.seeds,modes=a.modes,batch=8,lr=.0003,weight_decay=.0001,aux_weight=10,skip_sampling_steps=1500,initialization='same archived geometry-pretrained backbone per seed; same seeded new modules; fresh AdamW',pretrained_sha256={s:sha(a.pretrained/f'fusion_{s}_pretrained.pt') for s in a.seeds},cache_manifest_sha256=sha(a.cache/'manifest.json'),input_controls='within-scene cyclic audio shift; RGB brightness times0.2',selection='final scheduled update only; no epoch selection',evidence='Exploratory reused development scenes',cost='equal parameter count and updates, not equal measured FLOPs; each mode uses all head parameters')
 (a.out/'protocol.json').write_text(json.dumps(protocol,indent=2));scenes={part:[scene_to_gpu(torch.load(a.cache/f'{sid}.pt',weights_only=True)) for sid in ids] for part,ids in data['scenes'].items()};results=[]
 for seed in a.seeds:
  for mode in a.modes:
   torch.manual_seed(seed);m=InteractionMatcher(mode).cuda();base=torch.load(a.pretrained/f'fusion_{seed}_pretrained.pt',weights_only=True);loaded=m.load_state_dict(base,strict=False);assert not loaded.unexpected_keys and all(k.startswith(('v_projection.','a_projection.','q_projection.','v_retrieval.','a_retrieval.','gate.','joint_head.')) for k in loaded.missing_keys)
   opt=torch.optim.AdamW(m.parameters(),lr=.0003,weight_decay=.0001);rng=np.random.default_rng(seed)
   for _ in range(1500):s=scenes['train'][int(rng.integers(len(scenes['train'])))];rng.integers(len(s['gt']),size=8)
   # Serialized initial state records identical initialization across modes.
   torch.save(m.state_dict(),a.out/f'{mode}_{seed}_initial.pt');history=[];torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
   for step in range(a.steps):
    m.train();s=scenes['train'][int(rng.integers(len(scenes['train'])))];ids=torch.tensor(rng.integers(len(s['gt']),size=8),device='cuda');opt.zero_grad();field,v,au=predict(m,s,ids,aux=True);ce=dense_loss(field,s['candidates'][:,:3],s['gt'][ids]);aux=auxiliary_loss(v,au,s['target'][ids]);loss=ce+10*aux;loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5.);opt.step()
    if step%100==0 or step==a.steps-1:
     row=dict(step=step,loss=float(loss),dense_ce=float(ce),aux=float(aux),elapsed_seconds=time.perf_counter()-start);history.append(row);print(json.dumps(dict(mode=mode,seed=seed,**row)),flush=True)
   torch.cuda.synchronize();seconds=time.perf_counter()-start;peak=torch.cuda.max_memory_allocated();torch.save(m.state_dict(),a.out/f'{mode}_{seed}.pt');evaluation=evaluate(m,scenes['val']);rec=dict(mode=mode,seed=seed,parameters=sum(p.numel() for p in m.parameters()),training_seconds=seconds,peak_vram_bytes=peak,history=history,val=evaluation);(a.out/f'{mode}_{seed}.json').write_text(json.dumps(rec,indent=2));results.append({k:v for k,v in rec.items() if k not in ('history','val')}|{'val':evaluation['summary']});(a.out/'summary.json').write_text(json.dumps(results,indent=2));print(json.dumps(results[-1]),flush=True);del m,opt,field,v,au,loss,ce,aux;torch.cuda.empty_cache()
 print('COMPLETE',flush=True)
if __name__=='__main__':main()

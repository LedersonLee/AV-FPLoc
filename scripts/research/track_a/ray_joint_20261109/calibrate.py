"""Inner-scene held residual calibration; never fits to outer-held labels."""
from pathlib import Path
import json,time,hashlib,subprocess
import numpy as np,torch
from torch import nn
from train_acoustic import RayModel
from joint_models import JointRay
from oracle_capacity import represent
P=Path('/output');OLD=Path('/workspace/experiments/directional_tribranch_20261109');PR=json.loads((P/'PROTOCOL.json').read_text())
def t(x):return torch.as_tensor(x,dtype=torch.float32,device='cuda')
def main():
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.cuda.set_per_process_memory_fraction(.2);data={s:dict(np.load(P/'audio'/s/'features.npz')) for s in PR['training_scenes']};obs={s:dict(np.load(OLD/'data'/s/'observations.npz')) for s in data};records=[]
 for seed in PR['seeds']:
  for fold,held in enumerate(PR['outer_folds']):
   ids=[s for s in data if s not in held]
   for group,kind in [('acoustic','raw'),('acoustic','chirp'),('acoustic','embedding'),('joint','attention')]:
    root=P/'calibration'/f'{group}_{kind}_seed{seed}_fold{fold}';root.mkdir(parents=True,exist_ok=True)
    if (root/'complete.json').exists():records+=json.loads((root/'complete.json').read_text())['held_records'];continue
    errs=[];receipts=[]
    for inner in range(2):
     val=ids[inner::2];fit=[s for s in ids if s not in val];key='raw' if group=='joint' else kind;x=np.concatenate([data[s][key].astype('float32') for s in fit]);axes=(0,) if x.ndim==2 else (0,2,3);mu=x.mean(axes,keepdims=True);std=x.std(axes,keepdims=True).clip(.01);xx=t(np.clip((x-mu)/std,-10,10));v=np.concatenate([obs[s]['visual'] for s in fit]);vm=v.mean((0,1),keepdims=True);vs=v.std((0,1),keepdims=True).clip(.01);vv=t(np.clip((v-vm)/vs,-10,10));yy=t(np.concatenate([np.log1p(represent(obs[s]['target'],'raw36')) for s in fit]).astype('float32'));torch.manual_seed(seed);model=(JointRay(kind) if group=='joint' else RayModel(kind,36)).cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001);rng=np.random.default_rng(seed);losses=[];start=time.time()
     for epoch in range(80):
      for idx in np.array_split(rng.permutation(len(x)),4):
       pred=model(xx[idx],vv[idx]) if group=='joint' else model(xx[idx]);loss=(pred-yy[idx]).square().mean();opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();losses.append(loss.item())
     model.eval();cp=root/f'inner{inner}.pt';torch.save(dict(state=model.cpu().state_dict(),group=group,kind=kind,mu=mu,std=std,vm=vm,vs=vs,fit_scenes=fit,validation_scenes=val),cp);model.cuda()
     with torch.inference_mode():
      for sid in val:
       a=t(np.clip((data[sid][key]-mu)/std,-10,10));v=t(np.clip((obs[sid]['visual']-vm)/vs,-10,10));pred=(model(a,v) if group=='joint' else model(a)).cpu().numpy();err=abs(pred-np.log1p(represent(obs[sid]['target'],'raw36')));errs.append(err);np.savez_compressed(root/f'{sid}_oof.npz',prediction=pred,absolute_error=err,query_ids=obs[sid]['query_ids'])
     receipts.append(dict(inner=inner,fit_scenes=fit,validation_scenes=val,updates=320,seconds=time.time()-start,final_loss=losses[-1],sha256=hashlib.sha256(cp.read_bytes()).hexdigest()));del model,opt
    err=np.concatenate(errs);sigma=err.mean(0).clip(.02);q90=np.quantile(err,.9,axis=0);heldrows=[]
    for sid in held:
     folder=P/group/(f'{kind}_k36_seed{seed}_fold{fold}' if group=='acoustic' else f'{kind}_seed{seed}_fold{fold}');pred=np.load(folder/f'{sid}.npz')['prediction'];e=abs(pred-np.log1p(represent(obs[sid]['target'],'raw36')));heldrows.append(dict(scene=sid,group=group,kind=kind,seed=seed,fold=fold,interval90_coverage=float((e<=q90).mean()),mean_absolute_log_error=float(e.mean()),predicted_mean_absolute_log_error=float(sigma.mean()),direction_coverage=(e<=q90).mean(0).tolist(),direction_actual_error=e.mean(0).tolist()))
    np.savez_compressed(root/'scales.npz',sigma=sigma,q90=q90);(root/'complete.json').write_text(json.dumps(dict(inner_fits=receipts,held_records=heldrows,outer_fit_scenes=ids,outer_held_scenes=held,code_commit=subprocess.check_output(['git','-C','/workspace/AV-FPLoc','rev-parse','HEAD'],text=True).strip()),indent=2));records+=heldrows;print('CALIBRATION',group,kind,seed,fold,flush=True)
 (P/'calibration_results.json').write_text(json.dumps(dict(records=records,scope='48 innerfits, fit-sceneOOF directionglobalscale;heldcoverage measured, notquery-specificconfidence orguaranteedcalibration',test_accessed=False),indent=2))
if __name__=='__main__':main()

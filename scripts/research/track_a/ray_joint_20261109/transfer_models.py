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
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.cuda.set_per_process_memory_fraction(.2);ids=PR['training_scenes'];held=PR['transfer_scenes'];assert not set(ids)&set(held);data={s:dict(np.load(P/'audio'/s/'features.npz')) for s in ids+held};obs={s:dict(np.load(OLD/'data'/s/'observations.npz')) for s in ids+held};receipts=[];models=[]
 for seed in PR['seeds']:
  for group,kinds in [('acoustic',['ecn','embedding','raw','chirp']),('joint',['vision','acoustic','concat','attention','disabled'])]:
   for kind in kinds:
    root=P/'transfer_models'/f'{group}_{kind}_seed{seed}';root.mkdir(parents=True,exist_ok=True);cp=root/'model.pt';models.append((group,kind,seed,root))
    if (root/'receipt.json').exists():receipts.append(json.loads((root/'receipt.json').read_text()));continue
    key='raw' if group=='joint' else kind;x=np.concatenate([data[s][key].astype('float32') for s in ids]);dims=(0,) if x.ndim==2 else (0,2,3);mu=x.mean(dims,keepdims=True);std=x.std(dims,keepdims=True).clip(.01);xx=t(np.clip((x-mu)/std,-10,10));v=np.concatenate([obs[s]['visual'] for s in ids]);vm=v.mean((0,1),keepdims=True);vs=v.std((0,1),keepdims=True).clip(.01);vv=t(np.clip((v-vm)/vs,-10,10));yy=t(np.concatenate([np.log1p(represent(obs[s]['target'],'raw36')) for s in ids]).astype('float32'));torch.manual_seed(seed);m=(JointRay(kind) if group=='joint' else RayModel(kind,36)).cuda();opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.001);rng=np.random.default_rng(seed);losses=[];start=time.time()
    for epoch in range(40):
     for ix in np.array_split(rng.permutation(len(x)),12):
      p=m(xx[ix],vv[ix]) if group=='joint' else m(xx[ix]);loss=(p-yy[ix]).square().mean();opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1);opt.step();losses.append(loss.item())
    torch.save(dict(state=m.cpu().state_dict(),mu=mu,std=std,vm=vm,vs=vs,group=group,kind=kind,seed=seed,k=36,fit_scenes=ids),cp);r=dict(group=group,kind=kind,seed=seed,fit_scenes=ids,held_scenes=held,updates=480,epochs=40,loss_final=losses[-1],seconds=time.time()-start,checkpoint_sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),code_commit=subprocess.check_output(['git','-C','/workspace/AV-FPLoc','rev-parse','HEAD'],text=True).strip());(root/'receipt.json').write_text(json.dumps(r,indent=2));receipts.append(r);del m,opt;print('TRANSFER FIT',group,kind,seed,flush=True)
 freeze=P/'transfer_frozen.json'
 if not freeze.exists():freeze.write_text(json.dumps(dict(models=receipts,fit_scenes=ids,transfer_scenes=held,coefficient=.15,created_before_predictions=True,scope='AllprimaryK36arms; noTransferselection'),indent=2))
 for group,kind,seed,root in models:
  ck=torch.load(root/'model.pt',map_location='cpu',weights_only=False);m=(JointRay(kind) if group=='joint' else RayModel(kind,36)).cuda().eval();m.load_state_dict(ck['state']);key='raw' if group=='joint' else kind
  with torch.inference_mode():
   for sid in held:
    x=t(np.clip((data[sid][key]-ck['mu'])/ck['std'],-10,10));v=t(np.clip((obs[sid]['visual']-ck['vm'])/ck['vs'],-10,10));pred=(m(x,v) if group=='joint' else m(x)).cpu().numpy();np.savez_compressed(root/f'{sid}.npz',prediction=pred,query_ids=obs[sid]['query_ids'])
  del m
 print('TRANSFER PREDICTIONS FROZEN AND COMPLETE',len(models),flush=True)
if __name__=='__main__':main()

from pathlib import Path
import json,time,hashlib,subprocess
import numpy as np,torch
from torch import nn
from joint_models import JointRay
from oracle_capacity import represent
P=Path('/output');OLD=Path('/workspace/experiments/directional_tribranch_20261109');PR=json.loads((P/'PROTOCOL.json').read_text())
def t(x):return torch.as_tensor(x,dtype=torch.float32,device='cuda')
def main():
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.cuda.set_per_process_memory_fraction(.2);scenes=PR['training_scenes'];data={s:np.load(P/'audio'/s/'features.npz')['raw'].astype('float32') for s in scenes};o={s:dict(np.load(OLD/'data'/s/'observations.npz')) for s in scenes};receipts=[]
 for seed in PR['seeds']:
  for fold,held in enumerate(PR['outer_folds']):
   fit=[s for s in scenes if s not in held];a=np.concatenate([data[s] for s in fit]);v=np.concatenate([o[s]['visual'] for s in fit]);am=a.mean((0,2,3),keepdims=True);ast=a.std((0,2,3),keepdims=True).clip(.01);vm=v.mean((0,1),keepdims=True);vst=v.std((0,1),keepdims=True).clip(.01);aa=t(np.clip((a-am)/ast,-10,10));vv=t(np.clip((v-vm)/vst,-10,10));yy=t(np.concatenate([np.log1p(represent(o[s]['target'],'raw36')) for s in fit]).astype('float32'))
   for kind in ['vision','acoustic','concat','attention','disabled']:
    dest=P/'joint'/f'{kind}_seed{seed}_fold{fold}';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'receipt.json').exists():receipts.append(json.loads((dest/'receipt.json').read_text()));continue
    torch.manual_seed(seed);m=JointRay(kind).cuda();opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.001);rng=np.random.default_rng(seed);losses=[];start=time.time()
    for epoch in range(40):
     m.train();err=[]
     for ids in np.array_split(rng.permutation(len(a)),8):
      loss=(m(aa[ids],vv[ids])-yy[ids]).square().mean();opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1);opt.step();err.append(loss.item())
     losses.append(float(np.mean(err)))
    m.eval();cp=dest/'model.pt';torch.save(dict(state=m.cpu().state_dict(),kind=kind,k=36,am=am,ast=ast,vm=vm,vst=vst,fit_scenes=fit,held_scenes=held),cp);m.cuda()
    with torch.inference_mode():
     for sid in held:
      aq=t(np.clip((data[sid]-am)/ast,-10,10));vq=t(np.clip((o[sid]['visual']-vm)/vst,-10,10));pred=m(aq,vq).cpu().numpy();donor=m(torch.roll(aq,17,0),vq).cpu().numpy();np.savez_compressed(dest/f'{sid}.npz',prediction=pred,donor=donor,query_ids=o[sid]['query_ids'])
    r=dict(kind=kind,seed=seed,fold=fold,fit_scenes=fit,held_scenes=held,epochs=40,updates=320,parameters=sum(p.numel() for p in m.parameters()),loss_curve=losses,seconds=time.time()-start,checkpoint_sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),code_commit=subprocess.check_output(['git','-C','/workspace/AV-FPLoc','rev-parse','HEAD'],text=True).strip());(dest/'receipt.json').write_text(json.dumps(r,indent=2));receipts.append(r);print('JOINT',kind,seed,fold,round(losses[-1],4),round(r['seconds'],1),flush=True);del m,opt
 (P/'joint_fits.json').write_text(json.dumps(dict(fits=receipts,test_accessed=False),indent=2))
if __name__=='__main__':main()

"""Fixed-budget supervised ray learning, all12 scene-held folds; no model selection on held."""
from pathlib import Path
import json,time,hashlib,subprocess
import numpy as np,torch
from torch import nn
from oracle_capacity import represent
P=Path('/output');OLD=Path('/workspace/experiments/directional_tribranch_20261109');PR=json.loads((P/'PROTOCOL.json').read_text())
class RayModel(nn.Module):
 def __init__(self,kind,k):
  super().__init__();self.kind=kind
  if kind in ['ecn','embedding']:
   self.encoder=nn.Sequential(nn.Linear(552 if kind=='ecn' else 128,128),nn.GELU(),nn.Linear(128,64),nn.GELU())
  else:self.encoder=nn.Sequential(nn.Conv2d(18,16,3,padding=1),nn.GroupNorm(4,16),nn.GELU(),nn.Conv2d(16,32,3,stride=2,padding=1),nn.GroupNorm(4,32),nn.GELU(),nn.AvgPool2d(4,4),nn.Flatten(),nn.Linear(512,64),nn.GELU())
  self.head=nn.Linear(64,k)
 def forward(self,x):return self.head(self.encoder(x))
def tensor(x):return torch.as_tensor(x,dtype=torch.float32,device='cuda')
def main():
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--controls',action='store_true');args=ap.parse_args();torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True);torch.cuda.set_per_process_memory_fraction(.20);scenes=PR['training_scenes'];data={s:dict(np.load(P/'audio'/s/'features.npz')) for s in scenes};obs={s:dict(np.load(OLD/'data'/s/'observations.npz')) for s in scenes};receipts=[]
 for seed in PR['seeds']:
  for fold,held in enumerate(PR['outer_folds']):
   fit=[s for s in scenes if s not in held]
   for kind in (['sorted','isotropic'] if args.controls else ['ecn','embedding','raw','chirp']):
    x=np.concatenate([data[s][kind].astype('float32') for s in fit]);dims=(0,) if x.ndim==2 else (0,2,3);mean=x.mean(dims,keepdims=True);std=x.std(dims,keepdims=True).clip(.01);xx=tensor(np.clip((x-mean)/std,-10,10));test={s:tensor(np.clip((data[s][kind]-mean)/std,-10,10)) for s in held}
    for k in ([36] if args.controls else [4,8,12,36]):
     dest=P/'acoustic'/f'{kind}_k{k}_seed{seed}_fold{fold}';dest.mkdir(parents=True,exist_ok=True)
     if (dest/'receipt.json').exists():receipts.append(json.loads((dest/'receipt.json').read_text()));continue
     torch.manual_seed(seed);np.random.seed(seed);model=RayModel(kind,k).cuda();y=tensor(np.concatenate([np.log1p(represent(obs[s]['target'],f'raw{k}')) for s in fit]).astype('float32'));optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001);rng=np.random.default_rng(seed);start=time.time();losses=[]
     for epoch in range(40):
      model.train();errs=[]
      for inds in np.array_split(rng.permutation(len(x)),8):
       pred=model(xx[inds]);loss=(pred-y[inds]).square().mean();optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();errs.append(loss.item())
      losses.append(float(np.mean(errs)))
     model.eval();cp=dest/'model.pt';torch.save(dict(state=model.cpu().state_dict(),mean=mean,std=std,kind=kind,k=k,seed=seed,fit_scenes=fit,held_scenes=held),cp);model.cuda()
     with torch.inference_mode():
      for sid in held:
       preds={kind:model(test[sid]).cpu().numpy()}
       if kind in ['raw','chirp']:
        for control in ['permuted','isotropic','sorted']:
         # Controls in this initial pass are raw RIR transforms; chirp-specific perturbation is separate.
         if kind=='raw':preds[control]=model(tensor(np.clip((data[sid][control]-mean)/std,-10,10))).cpu().numpy()
       np.savez_compressed(dest/f'{sid}.npz',prediction=preds[kind],**{n:v for n,v in preds.items() if n!=kind},query_ids=obs[sid]['query_ids'])
     receipt=dict(kind=kind,k=k,seed=seed,fold=fold,fit_scenes=fit,held_scenes=held,epochs=40,updates=320,batch=128,parameters=sum(v.numel() for v in model.parameters()),loss='mean squared log1p ray error, averaged acrossK',loss_curve=losses,seconds=time.time()-start,checkpoint_sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),code_commit=subprocess.check_output(['git','-C','/workspace/AV-FPLoc','rev-parse','HEAD'],text=True).strip());(dest/'receipt.json').write_text(json.dumps(receipt,indent=2));receipts.append(receipt);print('FIT',kind,k,seed,fold,round(losses[-1],4),round(time.time()-start,1),flush=True)
     del model,optimizer
 (P/('control_fits.json' if args.controls else 'acoustic_fits.json')).write_text(json.dumps(dict(fits=receipts,test_accessed=False),indent=2))
if __name__=='__main__':main()

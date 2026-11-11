"""Full-cohort AVLoc training. No Test access; strict cache-ready and A3 stage gates."""
import argparse,hashlib,json,math,os,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from model import AVLoc,FOV,objective
from dataset import Observations
from augment import prepare,configure_noise

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic_save(obj,path):
 tmp=path.with_suffix(path.suffix+'.tmp');torch.save(obj,tmp);os.replace(tmp,path)
def seed_all(seed):
 torch.manual_seed(seed);np.random.seed(seed);torch.cuda.manual_seed_all(seed)
def init_model(arm,weights,seed):
 # One common full-model draw produces identical V/A/AV initial weights across arms.
 seed_all(seed);full=AVLoc('A3',weights);state=full.state_dict();model=AVLoc(arm);model.load_state_dict({k:state[k] for k in model.state_dict()},strict=True);del full;return model

def evaluate(model,loader,device):
 model.eval();total=0;mae=0.;nll=0.;per_scene={}
 with torch.inference_mode():
  for batch in loader:
   rgb,echo,target=prepare(batch,device,False);out=model(rgb,echo);key={'A0':'V','A1':'A','A2':'AV','A2_disabled':'AV','A3':'F'}[model.arm];mu,b=out[key];truth=target[:,list(FOV)] if key=='V' else target;err=abs(mu-truth).mean(-1);nl=(abs(mu-truth)/b+(2*b).log()).mean(-1);mae+=float(err.sum());nll+=float(nl.sum());total+=len(err)
   for sid,e,n in zip(batch['scene'],err.cpu().tolist(),nl.cpu().tolist()):
    row=per_scene.setdefault(sid,dict(count=0,mae_sum=0.,nll_sum=0.));row['count']+=1;row['mae_sum']+=e;row['nll_sum']+=n
 return dict(queries=total,ray_mae=mae/total,ray_nll=nll/total,per_scene={k:dict(queries=v['count'],ray_mae=v['mae_sum']/v['count'],ray_nll=v['nll_sum']/v['count']) for k,v in per_scene.items()})

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--weights',type=Path,required=True);ap.add_argument('--noise',type=Path,required=True);ap.add_argument('--arm',choices=['A0','A1','A2','A3','A2_disabled'],required=True);ap.add_argument('--seed',type=int,required=True);ap.add_argument('--batch',type=int,default=8);ap.add_argument('--workers',type=int,default=4);ap.add_argument('--epochs',type=int,default=100);ap.add_argument('--patience',type=int,default=20);ap.add_argument('--code-sha',required=True);ap.add_argument('--resume',action='store_true');ap.add_argument('--expert-gate',type=Path);a=ap.parse_args()
 if a.arm=='A3':
  assert a.expert_gate,'A3 needs completed A0/A1/A2 development evaluation gate';gate=json.loads(a.expert_gate.read_text());assert all(gate.get(k,{}).get('validated') for k in ['A0','A1','A2'])
 configure_noise(a.noise);assert 32%a.batch==0;accum=32//a.batch;torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.60);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;seed_all(a.seed)
 train=Observations(a.data,'train');val=Observations(a.data,'val');assert len(train.records)==112 and len(val.records)==19;assert len(train)==58894 and len(val)==9965,'Review integrity exclusions before overriding full-cohort counts'
 if not a.resume:a.out.mkdir(parents=True,exist_ok=False)
 else:
  assert (a.out/'latest.pt').is_file()
  assert not (a.out/'complete.json').exists(),'Completed runs must not be resumed or overwritten'
 protocol=dict(arm=a.arm,seed=a.seed,train_queries=len(train),val_queries=len(val),train_scenes=[r['scene'] for r in train.records],val_scenes=[r['scene'] for r in val.records],code_commit=a.code_sha,source_sha256={name:sha(Path(__file__).parent/name) for name in ['train.py','model.py','dataset.py','augment.py','audio.py']},input_manifest_sha256=sha(a.data/'manifest.json'),imagenet_sha256=sha(a.weights),batch=a.batch,effective_batch=32,max_epochs=a.epochs,patience=a.patience,optimizer='AdamW lr5e-4 wd.01 cosine1e-6',training_precision='bfloat16 autocast, float32 NLL',validation_precision='float32',selection='minimum clean Val19 query-mean ray MAE; patience20',noise_manifest_sha256=sha(a.noise/'manifest.json'),augmentation='paper probabilities; Gaussian or ESC50 train40 categories; explicit endpoints/spatial assumptions in augment.py; independent held10 category split',initialization='common full A3 draw subset by state key for all arms',test_accessed=False)
 if a.resume:assert json.loads((a.out/'protocol.json').read_text())==protocol,'Resume protocol mismatch'
 else:(a.out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
 generator=torch.Generator().manual_seed(a.seed+1000);tl=DataLoader(train,batch_size=a.batch,shuffle=True,generator=generator,num_workers=a.workers,pin_memory=True,persistent_workers=a.workers>0);vl=DataLoader(val,batch_size=a.batch,shuffle=False,num_workers=a.workers,pin_memory=True,persistent_workers=a.workers>0)
 model=init_model(a.arm,a.weights,a.seed).cuda();opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=.01);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs,eta_min=1e-6);start_epoch=1;best=float('inf');bad=0
 if a.resume:
  ck=torch.load(a.out/'latest.pt',weights_only=False,map_location='cpu');model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler']);start_epoch=ck['epoch']+1;best=ck['best'];bad=ck['bad'];torch.set_rng_state(ck['rng_cpu']);torch.cuda.set_rng_state_all(ck['rng_cuda']);np.random.set_state(ck['rng_numpy']);generator.set_state(ck['rng_loader'])
 else:
  initial=evaluate(model,vl,'cuda');(a.out/'initial_validation.json').write_text(json.dumps(initial,indent=2)+'\n');print('INITIAL',a.arm,a.seed,initial['ray_mae'],flush=True)
 for epoch in range(start_epoch,a.epochs+1):
  if bad>=a.patience:break
  start=time.time();model.train();opt.zero_grad(set_to_none=True);sum_loss=0.;n=0;updates=0
  for step,batch in enumerate(tl):
   with torch.random.fork_rng(devices=[0]):
    torch.manual_seed(a.seed+epoch*100000+step);torch.cuda.manual_seed_all(a.seed+epoch*100000+step);rgb,echo,target=prepare(batch,'cuda',True)
   with torch.autocast('cuda',dtype=torch.bfloat16):out=model(rgb,echo)
   out={k:tuple(x.float() for x in v) if k!='weights' else v.float() for k,v in out.items()};loss,_=objective(out,target);assert torch.isfinite(loss),'Nonfinite loss'
   # Last accumulation group uses its actual microbatch count.
   group_start=(step//accum)*accum;group_samples=min(32,len(train)-group_start*a.batch);(loss*len(rgb)/group_samples).backward();sum_loss+=float(loss.detach())*len(rgb);n+=len(rgb)
   if (step+1)%accum==0 or step+1==len(tl):
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();opt.zero_grad(set_to_none=True);updates+=1
   if step%200==0:print('STEP',epoch,step,'/',len(tl),'loss',float(loss.detach()),flush=True)
  assert n==len(train);evaluation=evaluate(model,vl,'cuda');scheduler.step();improved=evaluation['ray_mae']<best
  if improved:
   best=evaluation['ray_mae'];bad=0;atomic_save(dict(model=model.state_dict(),protocol=protocol,epoch=epoch,validation=evaluation),a.out/f'best_epoch_{epoch:03d}.pt');(a.out/'best.json').write_text(json.dumps(dict(epoch=epoch,path=f'best_epoch_{epoch:03d}.pt',ray_mae=best),indent=2)+'\n')
  else:bad+=1
  row=dict(epoch=epoch,train_loss=sum_loss/n,train_queries=n,optimizer_updates=updates,validation=evaluation,best=best,patience_used=bad,seconds=time.time()-start)
  atomic_save(dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),epoch=epoch,best=best,bad=bad,rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all(),rng_numpy=np.random.get_state(),rng_loader=generator.get_state(),metrics=row),a.out/'latest.pt')
  (a.out/f'epoch_{epoch:03d}.json').write_text(json.dumps(row,indent=2)+'\n');print('EPOCH',epoch,'VAL_MAE',evaluation['ray_mae'],'BEST',best,'SECONDS',row['seconds'],flush=True)
 (a.out/'complete.json').write_text(json.dumps(dict(status='training_complete',last_epoch=epoch if 'epoch' in locals() else start_epoch-1,best=best,reason='patience' if bad>=a.patience else 'max_epochs',benchmark_complete=False),indent=2)+'\n')
if __name__=='__main__':main()

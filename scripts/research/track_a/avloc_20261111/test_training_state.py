"""Disposable real-input optimizer/resume check, never a scientific evaluation."""
import argparse, json, tempfile
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from train import init_model, atomic_save, evaluate
from dataset import Observations
from augment import configure_noise, prepare, ambient
from model import objective

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--noise',type=Path,required=True);p.add_argument('--weights',type=Path,required=True);a=p.parse_args()
 torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.60);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 configure_noise(a.noise);ds=Observations(a.data,'train');val=Observations(a.data,'val');assert len(ds)==58894 and len(val)==9965
 for r in ds.records+val.records:
  q=np.load(a.data/r['scene']/'labels.npz');assert len(np.unique(q['query_ids']))==r['queries'];assert q['known'].all() and np.isfinite(q['target']).all()
 batch=next(iter(DataLoader(Subset(ds,range(4)),batch_size=4,num_workers=2)))
 def prepared(seed):
  with torch.random.fork_rng(devices=[0]):
   torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);return prepare(batch,'cuda',True)
 x=prepared(77);y=prepared(77);assert all(torch.equal(u,v) for u,v in zip(x,y));del x,y
 # Exercise actual ESC clips, identifiable by their shared six-channel spatial model.
 shared=independent=0
 for seed in range(30):
  torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);noise=ambient((6,4800),'cuda');assert torch.isfinite(noise).all();shared+=int(torch.equal(noise[0],noise[1]));independent+=int(not torch.equal(noise[0],noise[1]))
 assert shared and independent
 model=init_model('A2',a.weights,42).cuda();opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=.01);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,100,eta_min=1e-6)
 def step(seed):
  model.train();opt.zero_grad(set_to_none=True);rgb,spec,target=prepared(seed)
  with torch.autocast('cuda',dtype=torch.bfloat16):out=model(rgb,spec)
  out={k:tuple(t.float() for t in v) for k,v in out.items()};loss,_=objective(out,target);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();scheduler.step();return float(loss.detach())
 first=step(100)
 with tempfile.TemporaryDirectory() as tmp:
  path=Path(tmp)/'state.pt';atomic_save(dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all()),path)
  expected_loss=step(101);expected={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
  ck=torch.load(path,map_location='cpu',weights_only=False);model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler']);torch.set_rng_state(ck['cpu']);torch.cuda.set_rng_state_all(ck['cuda']);actual_loss=step(101)
  assert actual_loss==expected_loss;assert all(torch.equal(v.cpu(),expected[k]) for k,v in model.state_dict().items())
 result=evaluate(model,DataLoader(Subset(val,range(4)),batch_size=4),'cuda');assert result['queries']==4 and np.isfinite(result['ray_mae'])
 print(json.dumps(dict(status='pass',scope='disposable 4-query integration check; no benchmark or production checkpoint',train_queries=len(ds),val_queries=len(val),all_scene_labels_valid=True,augmentation_seed_replay_exact=True,esc_shared_draws=shared,gaussian_draws=independent,optimizer_resume_byte_exact=True,first_loss=first,resumed_loss=actual_loss,validation_path_finite=True),indent=2))
if __name__=='__main__':main()

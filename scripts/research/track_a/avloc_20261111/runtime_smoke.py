"""Real audited train inputs, disposable optimizer step; not a benchmark/trained model."""
import argparse,json,time
from pathlib import Path
import numpy as np,torch
from PIL import Image
from train import init_model
from augment import prepare
from model import objective

def main():
 p=argparse.ArgumentParser();p.add_argument('--scene',type=Path,required=True);p.add_argument('--weights',type=Path,required=True);p.add_argument('--batch',type=int,default=8);a=p.parse_args();torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.60);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 q=dict(np.load(a.scene/'labels.npz'));images=json.loads((a.scene/'images.json').read_text());n=a.batch;rgb=np.stack([np.array(Image.open(a.scene.parent/s).convert('RGB')).transpose(2,0,1) for s in images[:n]]);waves=np.array(np.load(a.scene/'rir.npy',mmap_mode='r')[:n]);batch=dict(rgb=torch.from_numpy(rgb),rir=torch.from_numpy(waves),target=torch.from_numpy(q['target'][:n]));records=[]
 for arm in ['A0','A1','A2','A3','A2_disabled']:
  torch.manual_seed(42);torch.cuda.manual_seed_all(42);model=init_model(arm,a.weights,42).cuda().train();opt=torch.optim.AdamW(model.parameters(),lr=5e-4);torch.cuda.reset_peak_memory_stats();t=time.perf_counter();v,echo,target=prepare(batch,'cuda',True)
  with torch.autocast('cuda',dtype=torch.bfloat16):out=model(v,echo)
  out={k:tuple(x.float() for x in vv) if k!='weights' else vv.float() for k,vv in out.items()};loss,_=objective(out,target);assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();torch.cuda.synchronize();records.append(dict(arm=arm,batch=n,loss=float(loss),gradient_norm=float(norm),seconds=time.perf_counter()-t,peak_allocated_bytes=torch.cuda.max_memory_allocated()));del out,model,opt,v,echo,target,loss;torch.cuda.empty_cache()
 print(json.dumps(dict(status='pass',scope='one disposable code-check update per arm; no production checkpoint, no model selection',records=records),indent=2))
if __name__=='__main__':main()

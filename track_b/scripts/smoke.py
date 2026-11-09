"""Synthetic overfit and actual GPU profiling; no real-data scientific claim."""
import argparse,json,time
from pathlib import Path
import torch
from track_b.models.dense import DenseScorer,dense_loss
from track_b.tests.test_dense import fixture


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--device',default='cuda');args=ap.parse_args()
    torch.manual_seed(42);torch.set_num_threads(2);dev=args.device
    rgb,rir,fp,cal,p=fixture();rgb=rgb[:1].repeat(8,1,1,1);rgb+=torch.randn_like(rgb)*.2
    rir=torch.randn(8,6,512);fp=fp[:1].expand(8,-1,-1,-1)
    rgb,rir,fp,p=[x.to(dev) for x in (rgb,rir,fp,p)];cal={k:v.to(dev) if isinstance(v,torch.Tensor) else v for k,v in cal.items()}
    gt=p[torch.linspace(0,len(p)-1,8).long()];m=DenseScorer().to(dev);opt=torch.optim.Adam(m.parameters(),lr=.003)
    if dev=='cuda':torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
    start=time.perf_counter();history=[]
    for step in range(301):
        opt.zero_grad();s=m(rgb,rir,fp,cal,p,chunk_size=512);loss=dense_loss(s,p,gt,sigma_xy=.15,sigma_yaw=.1)
        if step%50==0:history.append(dict(step=step,loss=float(loss),exact_map=float((s.argmax(-1)==torch.linspace(0,len(p)-1,8,device=dev).long()).float().mean())))
        if step<300:loss.backward();opt.step()
    if dev=='cuda':torch.cuda.synchronize()
    result=dict(kind='synthetic_8_query_overfit',seed=42,history=history,seconds=time.perf_counter()-start,params=sum(p.numel() for p in m.parameters()),
        peak_vram_bytes=torch.cuda.max_memory_allocated() if dev=='cuda' else None,device=dev,torch=torch.__version__,bank_size=len(p))
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()

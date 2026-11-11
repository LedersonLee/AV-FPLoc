"""Structural/scientific invariants; synthetic inputs are code checks, not results."""
import json,torch
from model import AVLoc,RayRouter,moment_match,objective

def main():
 torch.set_num_threads(2);torch.manual_seed(42);checks=[]
 # Closed-form mixture variance: equal Laplace(1,2), Laplace(5,2) -> mu3, b=sqrt6.
 mu,b=moment_match(torch.tensor([[1.,5.]]),torch.tensor([[2.,2.]]),torch.tensor([[.5,.5]]));torch.testing.assert_close(mu,torch.tensor([3.]));torch.testing.assert_close(b,torch.tensor([6.]).sqrt());checks.append('mixture total variance')
 router=RayRouter();v=(torch.ones(2,11),torch.ones(2,11,requires_grad=True));a=(torch.ones(2,36)*2,torch.ones(2,36,requires_grad=True));j=(torch.ones(2,36)*3,torch.ones(2,36,requires_grad=True));f,w=router(v,a,j);assert not w[:,6:31,0].any();torch.testing.assert_close(w.sum(-1),torch.ones(2,36));w.square().sum().backward();assert all(e[1].grad is None for e in [v,a,j]);checks+=['FoV router mask','router normalization','uncertainty routing input detached']
 rgb=torch.randn(2,3,480,640,device='cuda');audio=torch.randn(2,18,257,301,device='cuda');target=torch.rand(2,36,device='cuda')*10
 for arm in ['A0','A1','A2','A3','A2_disabled']:
  model=AVLoc(arm).cuda().eval()
  with torch.no_grad():
   out=model(rgb,audio);loss,_=objective(out,target);assert torch.isfinite(loss)
   if arm=='A3':
    changed=model(rgb+10,audio);torch.testing.assert_close(out['A'][0],changed['A'][0],rtol=0,atol=0)
    changed=model(rgb,audio+10);torch.testing.assert_close(out['V'][0],changed['V'][0],rtol=0,atol=0)
  checks.append(dict(arm=arm,parameters=sum(p.numel() for p in model.parameters()),shapes={k:[list(t.shape) for t in v] for k,v in out.items() if k!='weights'},loss=float(loss)))
  del model,out;torch.cuda.empty_cache()
 print(json.dumps(dict(status='pass',checks=checks,scope='architecture only, random untrained weights; not checkpoint evaluation'),indent=2))
if __name__=='__main__':main()

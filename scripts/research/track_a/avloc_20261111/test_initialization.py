import argparse,json,torch
from model import AVLoc,Expert,FOV
from train import init_model

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--weights',required=True);a=ap.parse_args();torch.set_num_threads(2);base=init_model('A3',a.weights,42);state=base.state_dict()
 for arm in ['A0','A1','A2','A2_disabled']:
  m=init_model(arm,a.weights,42)
  for k,v in m.state_dict().items():assert torch.equal(v,state[k]),(arm,k)
  del m
 im=torch.load(a.weights,weights_only=True,map_location='cpu');assert torch.equal(state['visual.backbone.0.weight'],im['conv1.weight'])
 expert=Expert('joint').eval();capture=[];hook=expert.encoder.register_forward_pre_hook(lambda module,args:capture.append(args[0].detach()))
 vmap=torch.arange(20,dtype=torch.float32)[None,None,None].expand(1,128,15,20);tokens=torch.zeros(1,5,128)
 with torch.no_grad():expert(tokens,vmap)
 seeded=capture[0][:,:36]-(expert.rays+expert.direction);coords=torch.tan(torch.arange(-5,6)*torch.pi/18)*20*3/8+9.5
 torch.testing.assert_close(seeded[0,list(FOV),0],coords,atol=2e-6,rtol=1e-6);assert seeded[0,6:31].abs().max()==0;hook.remove()
 print(json.dumps(dict(status='pass',checks=['common initialization exact across A0/A1/A2/disabled and A3','ImageNet conv1 byte equality','perspective image-column seeding at feature centres','no visual seeding outside FoV']),indent=2))
if __name__=='__main__':main()

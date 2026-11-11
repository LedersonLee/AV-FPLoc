"""Independent paper-based AVLoc-style 6ch adaptation. No FP-RIR dependency.
Assumptions (not author specifications): pre-norm, GELU, FFN512, dropout .1,
separable learned 2D positions, vertical-mean visual seeding, softplus heads.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50,resnet18
FOV=tuple(x%36 for x in range(-5,6))

def transformer(layers,d=128):
 layer=nn.TransformerEncoderLayer(d,8,dim_feedforward=512,dropout=.1,activation='gelu',batch_first=True,norm_first=True)
 return nn.TransformerEncoder(layer,layers,norm=nn.LayerNorm(d),enable_nested_tensor=False)

class Tokens(nn.Module):
 def __init__(self,kind,imagenet_path=None):
  super().__init__();self.kind=kind
  backbone=resnet50(weights=None) if kind=='vision' else resnet18(weights=None)
  if kind=='vision' and imagenet_path is not None:backbone.load_state_dict(torch.load(imagenet_path,map_location='cpu',weights_only=True))
  if kind=='audio':backbone.conv1=nn.Conv2d(18,64,7,2,3,bias=False)
  self.backbone=nn.Sequential(*list(backbone.children())[:-2]);self.project=nn.Conv2d(2048 if kind=='vision' else 512,128,1)
  h,w=(15,20) if kind=='vision' else (9,10)
  self.row=nn.Parameter(torch.randn(1,128,h,1)*.02);self.col=nn.Parameter(torch.randn(1,128,1,w)*.02);self.modality=nn.Parameter(torch.randn(1,1,128)*.02);self.intra=transformer(2)
 def forward(self,x):
  x=self.project(self.backbone(x));h,w=x.shape[-2:];assert h==self.row.shape[2] and w==self.col.shape[3], 'Expected full-resolution input';pos=self.row+self.col
  tokens=self.intra((x+pos).flatten(2).transpose(1,2)+self.modality)
  return tokens,tokens.transpose(1,2).reshape(len(x),128,h,w)

class Expert(nn.Module):
 def __init__(self,kind):
  super().__init__();self.kind=kind;self.k=11 if kind=='vision' else 36
  self.rays=nn.Parameter(torch.randn(1,self.k,128)*.02);self.direction=nn.Parameter(torch.randn(1,self.k,128)*.02)
  self.encoder=transformer(2 if kind=='vision' else 4);self.head=nn.Linear(128,2)
 def forward(self,tokens,visual_map=None):
  rays=(self.rays+self.direction).expand(len(tokens),-1,-1)
  if visual_map is not None:
   # Image right maps to increasing logical yaw; RGB horizontal FOV=106.2602deg.
   # Match focal/width=3/8 and feature-column centres; deterministic matmul,
   # avoiding CUDA grid_sample backward and its corner-vs-centre mismatch.
   width=visual_map.shape[-1];angles=torch.arange(-5,6,device=tokens.device,dtype=torch.float32)*torch.pi/18
   coords=torch.tan(angles)*width*3/8+(width-1)/2
   assert bool((coords>=0).all() and (coords<=width-1).all())
   columns=torch.arange(width,device=tokens.device,dtype=torch.float32)
   weights=(1-abs(columns[:,None]-coords[None,:])).clamp_min(0).to(visual_map.dtype)
   cols=(visual_map.mean(-2)@weights).transpose(1,2)
   seed=torch.zeros_like(rays);seed[:,list(FOV) if self.k==36 else list(range(11))]=cols.to(rays.dtype);rays=rays+seed
  result=self.encoder(torch.cat([rays,tokens],1))[:,:self.k];mu,scale=F.softplus(self.head(result)).unbind(-1)
  return mu+1e-4,scale+1e-3

def moment_match(means,scales,weights):
 mu=(weights*means).sum(-1);variance=(weights*(2*scales.square()+(means-mu[...,None]).square())).sum(-1)
 return mu,(variance.clamp_min(1e-12)/2).sqrt()

class RayRouter(nn.Module):
 def __init__(self):
  super().__init__();self.mlps=nn.ModuleList([nn.Sequential(nn.Linear(3 if k in FOV else 2,16),nn.GELU(),nn.Linear(16,3 if k in FOV else 2)) for k in range(36)])
 def forward(self,vision,audio,joint,uniform=False):
  # Ordering of experts is V/A/AV inside FOV, A/AV elsewhere.
  mus=[];bs=[];weights=[]
  for k,mlp in enumerate(self.mlps):
   experts=[audio,joint];positions=[k,k]
   if k in FOV:experts.insert(0,vision);positions.insert(0,FOV.index(k))
   m=torch.stack([e[0][:,j] for e,j in zip(experts,positions)],-1);b=torch.stack([e[1][:,j] for e,j in zip(experts,positions)],-1)
   w=torch.ones_like(b)/b.shape[-1] if uniform else mlp(b.detach()).softmax(-1)
   mu,scale=moment_match(m,b,w);mus.append(mu);bs.append(scale);weights.append(w if k in FOV else F.pad(w,(1,0)))
  return (torch.stack(mus,1),torch.stack(bs,1)),torch.stack(weights,1)

class AVLoc(nn.Module):
 def __init__(self,arm,imagenet_path=None):
  super().__init__();assert arm in ['A0','A1','A2','A3','A2_disabled'];self.arm=arm
  if arm!='A1':self.visual=Tokens('vision',imagenet_path)
  if arm!='A0':self.audio=Tokens('audio')
  if arm in ['A0','A3']:self.vision_expert=Expert('vision')
  if arm in ['A1','A3']:self.audio_expert=Expert('audio')
  if arm in ['A2','A3','A2_disabled']:self.joint_expert=Expert('joint')
  if arm=='A3':self.router=RayRouter()
 def forward(self,rgb=None,echo=None,uniform=False):
  out={}
  if self.arm!='A1':v,vm=self.visual(rgb)
  if self.arm!='A0':a,_=self.audio(echo)
  if self.arm in ['A0','A3']:out['V']=self.vision_expert(v,vm)
  if self.arm in ['A1','A3']:out['A']=self.audio_expert(a)
  if self.arm in ['A2','A3']:out['AV']=self.joint_expert(torch.cat([v,a],1),vm)
  if self.arm=='A2_disabled':
   # Same expert parameters, two isolated passes; no cross-modal activations.
   pv=self.joint_expert(v,vm);pa=self.joint_expert(a);mu=torch.stack([pv[0],pa[0]],-1);b=torch.stack([pv[1],pa[1]],-1);out['AV']=moment_match(mu,b,torch.full_like(mu,.5))
  if self.arm=='A3':out['F'],out['weights']=self.router(out['V'],out['A'],out['AV'],uniform)
  return out

def laplace_nll(pred,target):
 mu,b=pred;return ((target-mu).abs()/b+(2*b).log()).mean()
def objective(outputs,target):
 terms={k:laplace_nll(v,target[:,list(FOV)] if k=='V' else target) for k,v in outputs.items() if k!='weights'}
 weights={'F':1.,'AV':.1,'V':.5,'A':.5} if 'F' in terms else {k:1. for k in terms}
 return sum(weights[k]*v for k,v in terms.items()),terms

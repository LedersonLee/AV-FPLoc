"""Shared tokens/ray decoder for joint-learning comparisons; no GT features."""
import torch
from torch import nn
class JointRay(nn.Module):
 def __init__(self,kind,k=36):
  super().__init__();self.kind=kind;self.k=k;self.audio=nn.Sequential(nn.Conv2d(18,16,3,padding=1),nn.GroupNorm(4,16),nn.GELU(),nn.Conv2d(16,32,3,stride=2,padding=1),nn.GroupNorm(4,32),nn.GELU(),nn.AvgPool2d(4,4));self.visual=nn.Linear(128,32);self.pos=nn.Parameter(torch.randn(1,56,32)*.02);self.ray=nn.Parameter(torch.randn(1,k,32)*.02);self.decoder=nn.Sequential(nn.Linear(32,64),nn.GELU(),nn.Linear(64,1))
  # Keep common token encoders and identical per-ray decoder. Concat bottleneck parameter budget approximates two transformer blocks.
  if kind=='concat':self.mixer=nn.Sequential(nn.Flatten(),nn.Linear(56*32,14),nn.GELU(),nn.Linear(14,32))
  else:self.mixer=nn.TransformerEncoder(nn.TransformerEncoderLayer(32,4,128,dropout=0,batch_first=True,norm_first=True),2,enable_nested_tensor=False)
 def forward(self,a,v):
  at=self.audio(a).flatten(2).transpose(1,2);vt=self.visual(v);tok=torch.cat([vt,at],1)+self.pos;r=self.ray.expand(len(a),-1,-1)
  if self.kind=='concat':return self.decoder(r+self.mixer(tok)[:,None]).squeeze(-1)
  seq=torch.cat([r,tok],1)
  def branch(which):
   mask=torch.zeros(len(a),self.k+56,dtype=torch.bool,device=a.device)
   if which=='vision':mask[:,self.k+40:]=True
   if which=='acoustic':mask[:,self.k:self.k+40]=True
   return self.decoder(self.mixer(seq,src_key_padding_mask=mask)[:,:self.k]).squeeze(-1)
  if self.kind=='disabled':return .5*(branch('vision')+branch('acoustic'))
  return branch(self.kind)

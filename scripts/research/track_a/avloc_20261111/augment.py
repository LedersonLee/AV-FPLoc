"""Explicit paper-probability augmentation; endpoints are implementation assumptions.
Configured ESC50 training categories or Gaussian ambient noise; category split and
spatial-noise assumptions are recorded separately. Binaural reflection generalized to the6ch ring.
"""
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from audio import echo_wave,spectrogram,mirror

NOISE=None
def configure_noise(path):
 global NOISE
 p=Path(path);m=json.loads((p/'manifest.json').read_text());assert len(m['train_categories'])==40 and len(m['held_noise_categories'])==10;assert not set(m['train_categories'])&set(m['held_noise_categories']);NOISE=np.load(p/'train.npy',mmap_mode='r');assert NOISE.shape==(1600,240000)
def ambient(shape,device):
 if NOISE is not None and torch.rand((),device=device)<.5:
  for _ in range(8):
   row=int(torch.randint(len(NOISE),()).item());start=int(torch.randint(NOISE.shape[1]-shape[-1]+1,()).item());clip=np.array(NOISE[row,start:start+shape[-1]],copy=True)
   if float(np.mean(clip**2))>1e-12:return torch.from_numpy(clip).to(device)[None].expand(shape)
 return torch.randn(shape,device=device)

SNR_VALUES=list(range(-15,10,2))+[10,20,30,60]
def prepare(batch,device,training=False):
 rgb=batch['rgb'].to(device).float()/255;rir=batch['rir'].to(device);target=batch['target'].to(device)
 if training:
  for i in range(len(rgb)):
   if torch.rand((),device=device)<.5:
    v,r,t=mirror(rgb[i:i+1],rir[i:i+1],target[i:i+1]);rgb[i]=v[0];rir[i]=r[0];target[i]=t[0]
   if torch.rand((),device=device)<.5:
    if torch.rand((),device=device)<.6:rgb[i]*=1-.95*torch.rand((),device=device)
    if torch.rand((),device=device)<.6:rgb[i]+=torch.randn_like(rgb[i])*(.2*torch.rand((),device=device))
    if torch.rand((),device=device)<.3:
     h,w=rgb.shape[-2:];frac=.75*torch.rand(()).item();rh=max(1,int(h*frac**.5));rw=max(1,int(w*frac**.5));y=int(torch.randint(h-rh+1,()).item());x=int(torch.randint(w-rw+1,()).item());rgb[i,:,y:y+rh,x:x+rw]=0
    if torch.rand((),device=device)<.3:
     k=1+2*int(15*torch.rand(()).item());horizontal=bool(torch.rand(())<.5);kernel=torch.ones(3,1,1 if horizontal else k,k if horizontal else 1,device=device)/k
     rgb[i:i+1]=F.conv2d(rgb[i:i+1],kernel,padding=(0,k//2) if horizontal else (k//2,0),groups=3)
 rgb=rgb.clamp(0,1);rgb=(rgb-torch.tensor([.485,.456,.406],device=device)[None,:,None,None])/torch.tensor([.229,.224,.225],device=device)[None,:,None,None]
 echo=echo_wave(rir)
 if training:
  for i in range(len(echo)):
   if torch.rand((),device=device)<.5:
    snr=SNR_VALUES[int(torch.randint(len(SNR_VALUES),()).item())];noise=ambient(echo[i].shape,device);noise=noise*((echo[i].square().mean()/10**(snr/10))/noise.square().mean().clamp_min(1e-20)).sqrt();echo[i]+=noise
 spec=spectrogram(echo)
 if training:
  for i in range(len(spec)):
   if torch.rand((),device=device)<.7:
    for _ in range(2):
     for dim in [-2,-1]:
      width=int(torch.randint(13,()).item());start=int(torch.randint(spec.shape[dim]-width+1,()).item())
      if dim==-2:spec[i,:,start:start+width,:]=0
      else:spec[i,:,:,start:start+width]=0
 return rgb,spec,target

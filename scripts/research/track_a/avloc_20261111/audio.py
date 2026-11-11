"""Paper STFT timing; explicit 6ch adjacent-phase adaptation, direct path retained."""
import math
import torch
from torch.nn import functional as F

def chirp(device,dtype):
 t=torch.arange(2400,device=device,dtype=dtype)/48000
 ratio=20000/20.;phase=2*math.pi*20*.05/math.log(ratio)*(torch.exp(t*math.log(ratio)/.05)-1)
 return torch.cos(phase)

def echo_wave(rir):
 assert rir.ndim==3 and rir.shape[1]==6
 rir=F.pad(rir[...,:4800],(0,max(0,4800-rir.shape[-1])))
 signal=chirp(rir.device,rir.dtype);nfft=8192
 return torch.fft.irfft(torch.fft.rfft(rir,nfft)*torch.fft.rfft(signal,nfft),nfft)[...,:4800]

def spectrogram(echo):
 b,c,t=echo.shape;assert c==6 and t==4800
 z=torch.stft(echo.reshape(b*c,t),n_fft=512,hop_length=16,win_length=64,window=torch.hann_window(64,device=echo.device,dtype=echo.dtype),center=True,pad_mode='reflect',normalized=False,return_complex=True).reshape(b,c,257,301)
 mag=z.abs();cross=z*torch.roll(z,-1,1).conj();phase=cross/cross.abs().clamp_min(1e-12)
 return torch.cat([mag.clamp_min(1e-6).log(),phase.imag,phase.real],1)

def mirror(rgb,rir,target):
 # Body ring angles (index+3)*60deg; reflection sends index j to -j mod6.
 return rgb.flip(-1),rir[:,[0,5,4,3,2,1]],target[:,(-torch.arange(36,device=target.device))%36]

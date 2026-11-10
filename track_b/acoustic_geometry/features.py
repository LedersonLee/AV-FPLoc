"""Two explicit frontends; shared-channel RMS preserves interchannel levels."""
import torch
from torch.nn import functional as F


def features(rir):
    if rir.ndim!=3 or rir.shape[1]!=6 or not torch.isfinite(rir).all():
        raise ValueError('Expected finite B,6,T body-aligned RIR')
    x=F.pad(rir[...,:16384],(0,max(0,16384-rir.shape[-1])))
    x=x/x.square().mean((1,2),keepdim=True).sqrt().clamp_min(1e-8)
    def stft(wave,fft,hop,win):
        z=torch.stft(wave.flatten(0,1),n_fft=fft,hop_length=hop,win_length=win,
                     window=torch.hann_window(win,device=x.device),return_complex=True)
        return z.reshape(len(x),6,*z.shape[-2:])
    z=stft(x,256,128,256)
    cross=z*torch.roll(z,-1,1).conj()
    phase=cross/cross.abs().clamp_min(1e-8)
    spatial=F.adaptive_avg_pool2d(torch.cat((z.abs().log1p(),phase.real,phase.imag),1),(64,128))
    # Original-style 60ms crop, native48kHz RIR (no chirp convolution).
    echo=F.adaptive_avg_pool2d(stft(x[...,:2880],512,16,64).abs().log1p(),(64,128))
    return echo,spatial

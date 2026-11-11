import json,torch,numpy as np
from scipy.signal import chirp as scipy_chirp,fftconvolve
from audio import chirp,echo_wave,spectrogram,mirror

torch.set_num_threads(2);torch.manual_seed(42)
x=torch.randn(2,6,359,dtype=torch.float64);c=chirp(x.device,x.dtype);ref=scipy_chirp(np.arange(2400)/48000,20,.05,20000,method='logarithmic');np.testing.assert_allclose(c.numpy(),ref,atol=2e-10,rtol=1e-8)
y=echo_wave(x);expected=fftconvolve(np.pad(x.numpy(),((0,0),(0,0),(0,4800-359))),ref[None,None],axes=-1)[...,:4800];np.testing.assert_allclose(y.numpy(),expected,atol=2e-9,rtol=1e-7)
s=spectrogram(y.float());assert s.shape==(2,18,257,301) and torch.isfinite(s).all()
rgb=torch.randn(2,3,4,8);rays=torch.randn(2,36);m=mirror(*mirror(rgb,x,rays));assert all(torch.equal(a,b) for a,b in zip(m,(rgb,x,rays)))
# Impulse probe checks the direct path is not silently gated.
imp=torch.zeros(1,6,4800);imp[:,:,0]=1;out=echo_wave(imp);torch.testing.assert_close(out[:,:,:2400],chirp(imp.device,imp.dtype)[None,None].expand(1,6,-1),atol=2e-5,rtol=1e-4)
print(json.dumps(dict(status='pass',checks=['chirp equals independent SciPy log sweep','causal FFT convolution equals SciPy','full 18x257x301 STFT finite','joint reflection involution','direct path impulse retained'],training=False),indent=2))

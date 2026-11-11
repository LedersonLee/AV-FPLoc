"""Fixed shared-clock evidence metrics and scene bootstrap."""
import numpy as np
FS=48000

def measure(raw,times_ms,half_ms=.5):
 centers=np.rint(np.asarray(times_ms)*48).astype(int);h=round(half_ms*48);inds=centers[:,None]+np.arange(-h,h+1)
 if inds.min()<0 or inds.max()>=raw.shape[1]:return None
 x=raw[np.arange(6)[:,None],inds].astype(float);power=x*x;early=raw[:,48:min(3841,raw.shape[1])].astype(float);scale=max(np.mean(early*early),1e-30)
 energy=float(power.mean()/scale);peak=float(power.max(-1).mean()/scale);sharp=float(np.mean(power.max(-1)/np.maximum(power.mean(-1),1e-30)))
 norm=np.linalg.norm(x,axis=-1);corr=x@x.T/np.maximum(norm[:,None]*norm[None,:],1e-30);coh=float(np.mean(abs(corr[np.triu_indices(6,1)])))
 residual=(power.argmax(-1)-h)/48;timing=float(np.std(residual-residual.mean()))
 return dict(power_ratio=energy,peak_ratio=peak,sharpness=sharp,coherence=coh,timing_residual_ms=timing,peak_error_ms=float(np.mean(abs(residual))))
def average(records):
 records=[r for r in records if r is not None]
 return {k:float(np.mean([r[k] for r in records])) for k in records[0]} if records else None

def bootstrap(values,seed=20261111):
 values=np.asarray(values,float);rng=np.random.default_rng(seed);means=values[rng.integers(0,len(values),(10000,len(values)))].mean(-1)
 return dict(mean=float(values.mean()),ci95=np.quantile(means,[.025,.975]).tolist(),scenes=len(values))

"""Fixed feature definitions; wall-state observability is not acoustic path identity."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from scipy.signal import lfilter,find_peaks
FS=48000
BANDS=np.array([20.,176.020,774.604,3408.674,20000.])
def analytic(source,wall=0):
 s=np.array(source);t=np.pi+np.arange(6)*np.pi/3;r=s+np.array([.05*np.cos(t),np.zeros(6),-.05*np.sin(t)]).T;image=s.copy();image[2]=2*wall-s[2];return np.linalg.norm(r-image,axis=1)/343*1000

def features(channels,expected,matched):
 n=min(len(x) for x in channels);x=np.stack([x[:n] for x in channels]).astype(float);power=x*x;direct=power[:,:48].sum(1);ref=power[:,:48].max(1);early=power[:,48:min(n,3840)];late=power[:,3840:min(n,19200)];e=early.sum(1);latee=late.sum(1);pk=early.max(1);ix=early.argmax(1)+48;peakt=ix/48;env=lfilter(np.ones(6)/6,[1],power,axis=1);onsets=[];first=[];sharp=[]
 for c in range(6):
  ids=np.flatnonzero(env[c,48:min(n,3840)]>ref[c]*1e-6);onsets.append(float((ids[0]+48)/48) if len(ids) else None)
  found,_=find_peaks(power[c,48:min(n,3840)],height=ref[c]*1e-6);first.append(float((found[0]+48)/48) if len(found) else None);sharp.append(float(pk[c]/max(power[c,max(48,ix[c]-24):min(3840,ix[c]+25)].mean(),1e-30)))
 times=np.arange(48,48+early.shape[1])/48;cent=(early*times).sum(1)/np.maximum(e,1e-30);spread=np.sqrt((early*(times-cent[:,None])**2).sum(1)/np.maximum(e,1e-30));det=pk>ref*1e-6
 def spectrum(vals):
  length=vals.shape[1];nf=max(4096,2**int(np.ceil(np.log2(max(length,2)))));f=np.fft.rfftfreq(nf,1/FS);z=np.fft.rfft(vals,n=nf,axis=1);ps=abs(z)**2/nf;ps[:,1:-1]*=2
  return np.stack([ps[:,(f>=lo)&(f<hi)].sum(1) for lo,hi in zip(BANDS[:-1],BANDS[1:])],1),z,f
 be,z,f=spectrum(x[:,48:min(n,3840)]);bd,_,_=spectrum(x[:,:48]);bl,_,_=spectrum(x[:,3840:min(n,19200)]) if n>3840 else (np.zeros((6,4)),None,None)
 normalized=be/np.maximum(be.sum(1,keepdims=True),1e-30);center=normalized.mean(0);cos=np.sum(normalized*center,1)/np.maximum(np.linalg.norm(normalized,axis=1)*np.linalg.norm(center),1e-30)
 phase=[]
 for c in range(6):
  for j in range(c):
   cross=z[c]*z[j].conj();valid=(f>=200)&(f<8000)&(abs(cross)>max(abs(cross).max()*1e-6,1e-30));phase.append(float(abs(np.mean(cross[valid]/abs(cross[valid])))) if valid.any() else 0.)
 def window(ts):return [float(power[c,max(48,int((t-.5)*48)):min(n,int((t+.5)*48)+1)].sum()) for c,t in enumerate(ts)]
 def avg(v):return float(np.mean(v))
 # Estimate an extrapolated RT60 only with20dB usable interval and low tail, never for short/truncated signals.
 rt=None;rt_reason='insufficient length/dynamic range or fit quality';po=power.mean(0);dec=np.cumsum(po[::-1])[::-1];db=10*np.log10(np.maximum(dec/max(dec[0],1e-30),1e-30));mask=(db<=-5)&(db>=-25);tail=po[-max(1,n//10):].sum()/max(po.sum(),1e-30)
 if n>=FS*.5 and mask.sum()>=FS*.05 and tail<1e-5:
  tt=np.arange(n)[mask]/FS;yy=db[mask];coef=np.polyfit(tt,yy,1);fit=np.polyval(coef,tt);r2=1-float(((yy-fit)**2).sum()/max(((yy-yy.mean())**2).sum(),1e-30))
  if coef[0]<0 and r2>=.98:rt=float(-60/coef[0]);rt_reason='T20-derived extrapolation; slope fit R2>=.98, tail<−50dB, length>=.5s'
 peaknorm=pk/np.maximum(ref,1e-30);timing=float(np.std(peakt-expected));scores=dict(peak=float(np.log10(max(np.median(peaknorm),1e-30))),sharp=float(np.log10(max(np.mean(sharp),1e-30))),time=-float(np.median(abs(peakt-expected))) if det.any() else -80.,band=avg(cos) if det.any() else -1.,channel=-timing if det.any() else -80.)
 return dict(length_samples=n,channel_lengths=[len(v) for v in channels],direct_energy=direct.tolist(),early_energy=e.tolist(),late_energy_common400=latee.tolist(),total_energy=power.sum(1).tolist(),direct_normalized_early=(e/np.maximum(direct,1e-30)).tolist(),early_late_ratio=(e/np.maximum(latee,1e-30)).tolist(),peak_energy=pk.tolist(),peak_normalized=peaknorm.tolist(),peak_ms=[float(v) if yes else None for v,yes in zip(peakt,det)],onset_ms=onsets,first_peak_ms=first,first_strongest_gap_ms=[float(peakt[c]-v) if v is not None else None for c,v in enumerate(first)],sharpness=sharp,centroid_ms=cent.tolist(),spread_ms=spread.tolist(),expected_ms=expected.tolist(),peak_error_ms=[float(v) if yes else None for v,yes in zip(peakt-expected,det)],band_early=be.tolist(),band_direct=bd.tolist(),band_late=bl.tolist(),band_direct_ratio=(be/np.maximum(bd,1e-30)).tolist(),band_shape=normalized.tolist(),spectral_tilt_db=float(10*np.log10(max(be[:,-1].mean(),1e-30)/max(be[:,0].mean(),1e-30))),channel_relative_energy=(e/max(e.sum(),1e-30)).tolist(),channel_timing_residual_sd_ms=timing,channel_band_cosine=avg(cos),phase_resultant_mean=avg(phase),aligned_energy=window(expected),shifted_energy=window(expected+5),matched_time_energy=window(matched),scores=scores,rt60_extrapolated_s=rt,rt60_status=rt_reason,late_horizon_censored=n<19200)

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();folder=a.root/'controlled';manifest=json.loads((folder/'manifest.json').read_text());assert len(manifest)==1980
 poses={v['pose']['id']:v for v in manifest};ordered=sorted(poses);rng=np.random.default_rng(4211)
 while True:
  permutation=rng.permutation(ordered)
  if all(abs(poses[x]['pose']['z']-poses[y]['pose']['z'])>1 for x,y in zip(ordered,permutation)):break
 donor=dict(zip(ordered,permutation));rows=[];start=time.perf_counter()
 for i,v in enumerate(manifest):
  raw=folder/v['name']/'rir.npz';assert hashlib.sha256(raw.read_bytes()).hexdigest()==v['rir_sha256']
  with np.load(raw) as z:channels=[z[f'ch{c}'] for c in range(6)]
  for c,x in enumerate(channels):assert hashlib.sha256(x.tobytes()).hexdigest()==v['channel_bytes_sha256'][f'ch{c}']
  d=poses[donor[v['pose']['id']]];exp=analytic(v['geometry']['source'],-1 if v['state']=='moved' else 0);matched=analytic(d['geometry']['source'],-1 if v['state']=='moved' else 0)
  rec={k:v[k] for k in ['name','pose','state','absorption','scattering','profile','context','mode']};rec['features']=features(channels,exp,matched);rec['matched_pose']=donor[v['pose']['id']];rows.append(rec)
  if (i+1)%100==0:print('features',i+1,len(manifest),flush=True)
 (a.root/'features.json').write_text(json.dumps(dict(rows=rows,band_edges_hz=BANDS.tolist(),seconds=time.perf_counter()-start,matched_pose_mapping=donor),separators=(',',':')))
if __name__=='__main__':main()

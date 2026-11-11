"""New arrival-estimator audit on preserved and documented-C-API signals."""
import argparse,json,hashlib,sys,time
from pathlib import Path
import numpy as np
from scipy import signal
RING=np.pi+np.arange(6)*np.pi/3
FS=48000
METHODS=['wave_peak','energy_onset','short_increase','band_low_onset','band_high_onset','energy_centroid','envelope_peak','relative_1pct','relative_10pct','removed_difference_onset','band_low_difference_onset','band_high_difference_onset']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def analytic(d,moved=False,direct=False):
 s=np.array([.17,1.5,d]);r=s+np.stack([.05*np.cos(RING),np.zeros(6),-.05*np.sin(RING)],-1)
 if direct:r[:,2]+=2*d;return np.linalg.norm(r-s,axis=-1)/343*1000
 z=-1 if moved else 0;image=s.copy();image[2]=2*z-s[2];frac=(z-r[:,2])/(image[2]-r[:,2]);q=r+frac[:,None]*(image-r);length=np.linalg.norm(s-q,axis=-1)+np.linalg.norm(r-q,axis=-1)
 assert np.all(abs(q[:,0])<=12) and np.all((q[:,1]>=-8)&(q[:,1]<=12))
 return length/343*1000

def estimates(x,fs,reference,removed=None,threshold=1e-6):
 x=x.astype(float);power=x*x/max(reference,1e-30);lo=int(fs*.001);hi=min(len(x),int(fs*.080));window=power[lo:hi];gate=bool(window.max()>threshold);n=max(1,round(fs*.000125));env=signal.lfilter(np.ones(n)/n,[1],power)
 def onset(v,level):
  ids=np.flatnonzero(v[lo:hi]>level);return float((lo+ids[0])/fs*1000) if len(ids) else None
 def peak(v):return float((lo+np.argmax(v[lo:hi]))/fs*1000) if gate else None
 result=dict(wave_peak=peak(power),energy_onset=onset(env,threshold),short_increase=onset(env-np.r_[np.zeros(2*n),env[:-2*n]],threshold),envelope_peak=peak(env))
 for name,band in [('band_low_onset',[500,2000]),('band_high_onset',[2000,8000])]:
  sos=signal.butter(4,band,btype='bandpass',fs=fs,output='sos');f=signal.sosfilt(sos,x);e=signal.lfilter(np.ones(n)/n,[1],f*f/max(reference,1e-30));result[name]=onset(e,threshold)
 weights=np.maximum(window-threshold,0);result['energy_centroid']=float(np.sum(np.arange(lo,hi)*weights)/weights.sum()/fs*1000) if weights.sum()>0 else None
 result['relative_1pct']=onset(env,.01*env[lo:hi].max()) if gate else None;result['relative_10pct']=onset(env,.1*env[lo:hi].max()) if gate else None
 if removed is not None:
  length=min(len(x),len(removed));e=np.maximum(x[:length]**2-removed[:length].astype(float)**2,0)/max(reference,1e-30);result['removed_difference_onset']=onset(signal.lfilter(np.ones(n)/n,[1],e),threshold)
 else:result['removed_difference_onset']=None
 for name,band in [('band_low_difference_onset',[500,2000]),('band_high_difference_onset',[2000,8000])]:
  if removed is None:result[name]=None;continue
  length=min(len(x),len(removed));sos=signal.butter(4,band,btype='bandpass',fs=fs,output='sos');f=signal.sosfilt(sos,x[:length]-removed[:length].astype(float));e=signal.lfilter(np.ones(n)/n,[1],f*f/max(reference,1e-30));result[name]=onset(e,threshold)
 assert set(METHODS)<=set(result)
 result['exact_nonzero_diagnostic']=onset((x!=0).astype(float),0)
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root;old=root.parent/'reflection_observability_20261111';start=time.perf_counter();records=[];inputs=[];references={};removeds={}
 for d in [.3,2.,6.]:
  removed=np.load(old/f'controlled/single_d{d}_removed/rir.npy');removeds[d]=removed;references[d]=(removed[:,:48]**2).max(-1)
 # All96 previous files verified before any old result is trusted.
 for f in sorted((old/'controlled').glob('*/receipt.json'))+sorted((old/'diagnostic').glob('*.json')):
  r=json.loads(f.read_text());file=f.parent/'rir.npy' if f.name=='receipt.json' else f.with_suffix('.npy');assert sha(file)==r.get('rir_sha256',r.get('sha256'));raw=np.load(file);assert raw.shape[0]==6 and np.isfinite(raw).all();inputs.append(dict(path=str(file.relative_to(old)),sha256=sha(file)))
  primary='kind' in r;d=r['distance'];fs=r['settings'].get('sample_rate',48000);variant=r['variant'] if primary else r['label'];expected=analytic(d,moved=variant=='moved',direct=variant=='free_direct');kind=r.get('kind','single');ref=references[d]
  for ch,x in enumerate(raw):
   # Source-ring direct reference preserved; sample-rate-specific relative unit is diagnostic.
   rm=removeds[d][ch] if fs==48000 else None
   for threshold in [1e-8,1e-6,1e-4]:
    methods=estimates(x,fs,float(ref[ch]),rm,threshold)
    records.append(dict(source='previous',name=r.get('name',f.stem),kind=kind,distance=d,variant=variant,channel=ch,repeat=r.get('repeat',0),threshold=threshold,expected_ms=float(expected[ch]),methods=methods))
 assert len(inputs)==96
 # Reproduce saved dominant peaks with original search ranges for every60 diagnostics.
 prior=json.loads((old/'timing_diagnosis.json').read_text());max_error=0
 for r in prior['diagnostic']:
  f=old/'diagnostic'/f"d{r['distance']}_{r['label']}_r{r['repeat']}.npy";raw=np.load(f);fs=r['settings']['sample_rate'];peak=(np.argmax(raw[:,int(.003*fs):int(.080*fs)]**2,axis=-1)+int(.003*fs))/fs*1000;max_error=max(max_error,float(abs(peak-np.array(r['observed_peak_ms'])).max()))
 assert max_error<1e-10
 backend=json.loads((root/'backend_v2/manifest.json').read_text());backend_rows=[];geometry=[]
 for r in backend:
  for f in r['files']:assert sha(root/'backend_v2'/f['name'])==f['sha256']
  for rec in r['records']:
   ch=rec['channel'];x=np.fromfile(root/'backend_v2'/f"{r['file_prefix']}_r{rec['repeat']}_ch{ch}.f32",dtype=np.float32);d=r['distance'];assert len(x)==rec['samples']
   methods=estimates(x,48000,float(references[d][ch]),removeds[d][ch]);backend_rows.append(dict(mode=r['mode'],distance=d,context=r['context'],repeat=rec['repeat'],channel=ch,expected_ms=float(analytic(d)[ch]),methods=methods))
   receiver=np.array(rec['receiver']);expectedr=np.array([.17+.05*np.cos(RING[ch]),1.5,d-.05*np.sin(RING[ch])]);assert np.max(abs(receiver-expectedr))<1e-6;assert rec['ray_hit']==1 and abs(rec['ray_distance']-d)<1e-5
   geometry.append(dict(mode=r['mode'],distance=d,hit_distance=rec['ray_distance'],receiver_error_m=float(np.max(abs(receiver-expectedr))),normal=rec['normal']))
 # Analytic renderer reference, clearly separate from SoundSpaces evidence.
 refs=[]
 for d in [.3,2.,6.]:
  for moved in [False,True]:
   expected=analytic(d,moved=moved);x=np.zeros((6,3840))
   for ch,t in enumerate(expected):
    sample=t/1000*FS;n=int(np.floor(sample));fraction=sample-n;x[ch,n]=1-fraction;x[ch,n+1]=fraction
   refs.append(dict(distance=d,moved=moved,expected_ms=expected.tolist(),peak_ms=(x.argmax(-1)/FS*1000).tolist(),max_peak_error_ms=float(abs(x.argmax(-1)/FS*1000-expected).max()),method='Explicit analytic two-tap fractional-delay image-source reference, NOT SoundSpaces'))
   file=root/f'analytic_reference_d{d}_moved{int(moved)}.npy'
   if not file.exists():np.save(file,x)
   else:np.testing.assert_array_equal(np.load(file),x)
 assert max(x['max_peak_error_ms'] for x in refs)<=1000/FS
 # Primary B statistics: no geometry-centered search or fitted correction.
 summary=[]
 for d in [.3,2.,6.]:
  for method in METHODS:
   for threshold in [1e-8,1e-6,1e-4]:
    base=next(r for r in records if r['name']==f'single_d{d}_base' and r['channel']==0 and r['threshold']==threshold)
    relevant=[r for r in records if r['kind']=='single' and r['distance']==d and r['threshold']==threshold and r['name'].startswith(f'single_d{d}_')]
    vals={v:[r for r in relevant if r['variant']==v] for v in ['base','repeat','moved','removed']}
    errors=[r['methods'][method]-r['expected_ms'] for r in vals['base'] if r['methods'][method] is not None];shift=[]
    for b,m in zip(vals['base'],vals['moved']):
     if b['methods'][method] is not None and m['methods'][method] is not None:shift.append((m['methods'][method]-b['methods'][method])-(m['expected_ms']-b['expected_ms']))
    detected=[r['methods'][method] is not None for r in vals['removed']]
    repeats=[r for r in records if r['kind']=='single' and r['distance']==d and r['threshold']==threshold and r['variant']=='baseline']
    sds=[]
    for ch in range(6):
     v=[r['methods'][method] for r in repeats if r['channel']==ch and r['methods'][method] is not None]
     if len(v)>1:sds.append(float(np.std(v,ddof=1)))
    summary.append(dict(distance=d,method=method,threshold=threshold,bias_ms=float(np.mean(errors)) if errors else None,mae_ms=float(np.mean(abs(np.array(errors)))) if errors else None,max_abs_error_ms=float(np.max(abs(np.array(errors)))) if errors else None,shift_error_ms=float(np.mean(shift)) if shift else None,max_abs_shift_error_ms=float(np.max(abs(np.array(shift)))) if shift else None,removed_false_detection_fraction=float(np.mean(detected)),detected_channels=len(errors),repeat_sd_mean_ms=float(np.mean(sds)) if sds else None))
 result=dict(previous_input_count=len(inputs),previous_peak_max_replay_error_ms=max_error,inputs=inputs,previous_estimators=records,backend_estimators=backend_rows,geometry_checks=geometry,analytic_reference=refs,estimator_summary=summary,seconds=time.perf_counter()-start)
 (root/'analysis.json').write_text(json.dumps(result,indent=2));print('analysis',len(records),'prior estimator records;',len(backend_rows),'new mono records; replay error',max_error)
 for x in summary:
  if x['threshold']==1e-6 and x['distance'] in [2.,6.]:print(json.dumps(x))
if __name__=='__main__':main()

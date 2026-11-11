"""Same-position repeat and sequential/same-context channel phase diagnostics."""
import argparse,json,itertools
from pathlib import Path
import numpy as np

def similarity(x,y):
 n=min(len(x),len(y),3840);x=x[48:n].astype(float);y=y[48:n].astype(float);cos=float(x@y/max(np.linalg.norm(x)*np.linalg.norm(y),1e-30));a=np.fft.rfft(x,n=4096);b=np.fft.rfft(y,n=4096);f=np.fft.rfftfreq(4096,1/48000);mask=(f>=500)&(f<=8000);cross=a[mask]*b[mask].conj();valid=abs(cross)>max(abs(cross).max()*1e-8,1e-30);phase=float(abs(np.mean(cross[valid]/abs(cross[valid])))) if valid.any() else 0
 return dict(waveform_cosine=cos,phase_resultant=phase)
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root;manifest=json.loads((root/'backend_v2/manifest.json').read_text());analysis=json.loads((root/'analysis.json').read_text());result=[]
 def wave(r,rep,ch):return np.fromfile(root/'backend_v2'/f"{r['file_prefix']}_r{rep}_ch{ch}.f32",dtype=np.float32)
 for d in [2.,6.]:
  for mode in ['stationary_off','stationary_on']:
   ms=[r for r in manifest if r['mode']==mode and r['distance']==d];within=[];timing=[]
   for r in ms:
    pair=[similarity(wave(r,i,0),wave(r,j,0)) for i,j in itertools.combinations(range(5),2)];within.append({k:float(np.mean([x[k] for x in pair])) for k in pair[0]});rows=[x for x in analysis['backend_estimators'] if x['mode']==mode and x['distance']==d and x['context']==r['context']];values=[x['methods']['energy_onset'] for x in rows];timing.append(dict(context=r['context'],onsets_ms=values,sd_ms=float(np.std(values,ddof=1))))
   across=[similarity(wave(ms[i],0,0),wave(ms[j],0,0)) for i,j in itertools.combinations(range(3),2)]
   result.append(dict(distance=d,mode=mode,within_context=within,across_fresh_context=across,timing=timing))
  for mode in ['baseline','temporal_on','simultaneous']:
   values=[]
   for r in [r for r in manifest if r['mode']==mode and r['distance']==d]:
    pairs=[similarity(wave(r,0,i),wave(r,0,j)) for i,j in itertools.combinations(range(6),2)];values.append(dict(context=r['context'],**{k:float(np.mean([x[k] for x in pairs])) for k in pairs[0]}))
   result.append(dict(distance=d,mode=mode,interchannel_unaligned=values,interpretation='Same-context multiple listeners is not proof of synchronous physical phase; no per-channel peak realignment.'))
 (root/'channel_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()

"""Physical timing audits and matched-window subgroup contrasts; no model selection."""
import argparse,json
from pathlib import Path
import numpy as np
from geometry import paths,RING,C
from metrics import measure,bootstrap

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root;physical=[]
 for f in sorted((root/'diagnostic').glob('*.json')):
  r=json.loads(f.read_text());raw=np.load(f.with_suffix('.npy'));fs=r['settings']['sample_rate'];d=r['distance'];start=int(.003*fs);stop=int(.08*fs);peak=(np.argmax(raw[:,start:stop]**2,axis=-1)+start)/fs*1000
  if r['label']=='free_direct':length=np.linalg.norm(np.stack([.05*np.cos(RING),2*d-.05*np.sin(RING)],-1),axis=-1)
  else:length=np.array(paths([.17,d,0],np.array([[[-12,0],[12,0]]]))[0]['times_ms'])*C/1000
  r.update(expected_ms=(length/C*1000).tolist(),observed_peak_ms=peak.tolist(),mean_error_ms=float(np.mean(peak-length/C*1000)),interchannel_residual_std_ms=float(np.std(peak-length/C*1000)))
  physical.append(r)
 single=[]
 for d in [.3,2.,6.]:
  pred=paths([.17,d,0],np.array([[[-12,0],[12,0]]]))[0];variants={v:np.load(root/f'controlled/single_d{d}_{v}/rir.npy').astype(float) for v in ['base','repeat','removed','moved']};n=min(x.shape[1] for x in variants.values());sq={v:x[:,:n]**2 for v,x in variants.items()};delta=np.maximum(sq['base']-sq['removed'],0);indices=np.rint(np.array(pred['times_ms'])*48).astype(int)[:,None]+np.arange(-24,25);fraction=float(delta[np.arange(6)[:,None],indices].sum()/max(delta[:,48:3841].sum(),1e-30));full=delta[:,48:3841].sum();repeat=np.abs(sq['base'][:,48:3841]-sq['repeat'][:,48:3841]).sum();peaks={v:((np.argmax(sq[v][:,48:3841],axis=-1)+48)/48).tolist() for v in ['base','repeat','moved']}
  single.append(dict(distance=d,expected_ms=pred['times_ms'],observed_peaks_ms=peaks,expected_window_attributable_energy_fraction=fraction,repeat_abs_energy_change_over_removed_effect=float(repeat/max(full,1e-30))))
 (root/'timing_diagnosis.json').write_text(json.dumps(dict(diagnostic=physical,single_wall_attribution=single),indent=2))
 rows=[]
 for f in sorted((root/'hssd').glob('*.json')):
  data=json.loads(f.read_text());qs=data['queries'];raw=[np.roll(np.load(q['raw_path']),q['roll'],axis=0) for q in qs];local={}
  for qi,q in enumerate(qs):
   for h in q['hypotheses']:
    pp=[p for p in h['paths'] if min(p['times_ms'])>=2 and max(p['times_ms'])<=79];tt=np.array([np.mean(p['times_ms']) for p in pp])
    for p in pp:
     if not p['supported']:continue
     aligned=measure(raw[qi],p['times_ms'])['power_ratio'];donor=np.mean([measure(raw[(qi-int(np.random.default_rng(seed).integers(1,len(qs))))%len(qs)],p['times_ms'])['power_ratio'] for seed in [42,43,44]])
     distance='<1m' if p['distance_m']<1 else '1-3m' if p['distance_m']<3 else '>=3m';overlap=str(np.sum(abs(tt-np.mean(p['times_ms']))<1)>1);bearing=str((int((((p['bearing_deg']-h['pose'][2])%360)+1e-8)//90)%4)*90)
     for field,key in [('distance',distance),('overlap',overlap),('body_bearing_quadrant',bearing)]:local.setdefault((field,key,qi),[]).append(aligned-donor)
  for field,key in sorted({(f,k) for f,k,q in local}):
   values=[np.mean(v) for (f,k,q),v in local.items() if (f,k)==(field,key)];rows.append(dict(scene=data['scene'],field=field,group=key,queries=len(values),difference=float(np.mean(values))))
 groups=[]
 for field,key in sorted({(x['field'],x['group']) for x in rows}):
  subset=[x for x in rows if (x['field'],x['group'])==(field,key)];groups.append(dict(field=field,group=key,matched_power_difference=bootstrap([x['difference'] for x in subset]),per_scene=subset))
 (root/'matched_subgroups.json').write_text(json.dumps(groups,indent=2));print('diagnosed',len(physical),'renders and',len(groups),'strata')
if __name__=='__main__':main()

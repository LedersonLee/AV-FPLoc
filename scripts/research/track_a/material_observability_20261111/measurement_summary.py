"""Compact per-condition measurements with context variability, not scene confidence."""
import argparse,itertools,json
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);r=p.parse_args().root;rows=json.loads((r/'features.json').read_text())['rows'];rows=[v for v in rows if v['mode']=='both'];groups={}
 for v in rows:groups.setdefault((v['pose']['id'],v['state'],v['absorption'],v['scattering'],v['profile']),[]).append(v)
 output=[]
 for key,vs in sorted(groups.items()):
  rec=dict(zip(['pose','state','absorption','scattering','profile'],key));rec['contexts']=len(vs);metrics={}
  for field in ['peak_ms','onset_ms','first_strongest_gap_ms','sharpness','spread_ms','direct_energy','early_energy','late_energy_common400','early_late_ratio','peak_energy','direct_normalized_early','channel_timing_residual_sd_ms','channel_band_cosine','phase_resultant_mean']:
   values=[]
   for v in vs:
    a=v['features'][field];a=a if isinstance(a,list) else [a];a=[x for x in a if x is not None]
    if a:values.append(float(np.mean(a)))
   metrics[field]=dict(mean=float(np.mean(values)) if values else None,min=min(values) if values else None,max=max(values) if values else None,context_sd=float(np.std(values,ddof=1)) if len(values)>1 else None)
  rec['measurements']=metrics
  for field in ['band_early','band_direct_ratio','band_late','channel_relative_energy']:
   rec[field]=np.mean([v['features'][field] for v in vs],axis=0).tolist()
  output.append(rec)
 (r/'measurement_summary.json').write_text(json.dumps(dict(groups=output,scope='648conditions;each3fresh contexts;scalar means average channels first;context SD is not independent-scene uncertainty;raw units are squared float-sample sums, not calibrated Joules/SPL'),separators=(',',':')));print(len(output))
if __name__=='__main__':main()

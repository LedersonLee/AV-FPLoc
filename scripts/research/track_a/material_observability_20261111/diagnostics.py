import argparse,itertools,json
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);r=p.parse_args().root;rows=json.loads((r/'features.json').read_text())['rows'];result=json.loads((r/'results.json').read_text());poses=sorted(set(v['pose']['id'] for v in rows));out={}
 def select(**kw):return [v for v in rows if all((v['pose']['id'] if k=='pose' else v[k])==value for k,value in kw.items())]
 def avg(v,k):return float(np.mean(v['features'][k]))
 distance=[]
 for a,s in itertools.product([.05,.3,.7,.99],[0.,.5,1.]):
  energy=[np.mean([avg(v,'early_energy') for v in select(pose=pose,mode='both',state='present',profile='uniform',absorption=a,scattering=s)]) for pose in ['single2','single6']];distance.append(dict(absorption=a,scattering=s,energy6_over2=float(energy[1]/energy[0]),apparent_distance_power=float(-np.log(energy[1]/energy[0])/np.log(3))))
 out['distance']=distance
 controls=[]
 for pose in poses:
  c={mode:np.mean([avg(v,'early_energy') for v in select(pose=pose,state='present',profile='uniform',absorption=.3,scattering=.5,mode=mode)]) for mode in ['both','direct_only','indirect_only']};controls.append(dict(pose=pose,**{k:float(v) for k,v in c.items()}))
 out['direct_indirect']=controls
 profile=[v for v in rows if v['mode']=='both' and v['state']=='present' and v['absorption']==.3 and v['scattering']==.5];prototypes={}
 for pose,pr in itertools.product(poses,['uniform','low','high']):
  vals=[np.mean(v['features']['band_early'],axis=0) for v in profile if v['pose']['id']==pose and v['profile']==pr];shape=np.mean([v/max(v.sum(),1e-30) for v in vals],axis=0);prototypes[pose,pr]=shape
 samegeom=[float(np.linalg.norm(prototypes[pose,a]-prototypes[pose,b])) for pose in poses for a,b in itertools.combinations(['uniform','low','high'],2)];sameprofile=[float(np.linalg.norm(prototypes[a,pr]-prototypes[b,pr])) for pr in ['uniform','low','high'] for a,b in itertools.combinations(poses,2)];out['shape_distance']=dict(same_geometry_different_profile=dict(median=float(np.median(samegeom)),min=min(samegeom),max=max(samegeom)),different_geometry_same_profile=dict(median=float(np.median(sameprofile)),min=min(sameprofile),max=max(sameprofile)))
 banddecay=[]
 for pose,pr in itertools.product(poses,['uniform','low','high']):
  vs=[v for v in profile if v['pose']['id']==pose and v['profile']==pr];early=np.mean([v['features']['band_early'] for v in vs],axis=(0,1));late=np.mean([v['features']['band_late'] for v in vs],axis=(0,1));banddecay.append(dict(pose=pose,profile=pr,late_over_early_db=(10*np.log10(np.maximum(late,1e-30)/np.maximum(early,1e-30))).tolist(),common400_censored=sum(v['features']['late_horizon_censored'] for v in vs)))
 out['band_decay_window_ratio']=banddecay
 energy_partition=[]
 for pose in ['single2','single6']:
  e={a:np.mean([avg(v,'early_energy') for v in select(pose=pose,mode='both',state='present',profile='uniform',absorption=a,scattering=0.)]) for a in [.05,.3,.7]};ratio=e[.7]/e[.05];energy_partition.append(dict(pose=pose,observed_ratio70_over05=float(ratio),energy_fraction_prediction=(1-.7)/(1-.05),pressure_fraction_squared_prediction=((1-.7)/(1-.05))**2))
 out['coefficient_response']=energy_partition
 noise=json.loads((r/'noise_results.json').read_text())['records'];nd=[]
 for label in ['moderate','severe']:
  deltas=[]
  for v in noise:
   if v['noise']!=label:continue
   clean=next(x for x in noise if x['name']==v['name'] and x['noise']=='clean');deltas.append(v['spectral_tilt_db']-clean['spectral_tilt_db'])
  nd.append(dict(noise=label,tilt_delta_median_db=float(np.median(deltas)),min_db=min(deltas),max_db=max(deltas)))
 out['noise_shape_changes']=nd
 out['channel_energy_ranges']=[dict(pose=pose,min_relative=min(min(v['features']['channel_relative_energy']) for v in select(pose=pose,mode='both',state='present')),max_relative=max(max(v['features']['channel_relative_energy']) for v in select(pose=pose,mode='both',state='present'))) for pose in poses]
 out['late_censored_records']=sum(v['features']['late_horizon_censored'] for v in rows);out['rt60_gate_passed']=sum(v['features']['rt60_extrapolated_s'] is not None for v in rows);out['rt60_label']='Qualified T20 extrapolation only; not measured full60dB decay';out['raw_gate_weak_wall_cases']=[v['name'] for v in rows if v['mode']=='both' and v['state']!='removed' and all(x is None for x in v['features']['peak_ms'])]
 hssd=json.loads((r/'hssd_results.json').read_text());margin=[]
 for v in hssd['records']:
  e=sorted([x['supported_energy'] for x in v['candidate_evidence'] if x['supported_energy'] is not None],reverse=True);margin.append(dict(name=v['name'],scene=v['scene'],query_id=v['query_id'],relative_top_margin=(e[0]-e[1])/max(e[0],1e-30) if len(e)>1 else None))
 out['hssd_candidate_contrast']=margin
 (r/'diagnostics.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k in ['shape_distance','coefficient_response','late_censored_records','raw_gate_weak_wall_cases']},indent=2))
if __name__=='__main__':main()

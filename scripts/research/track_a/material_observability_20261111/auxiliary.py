"""Frozen-rule noise controls and bounded HSSD candidate-evidence diagnostics."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from analyze import features,analytic
from results import RULES,evaluate

def score(f,result):
 s=f['scores'].copy();s['combined']=float(np.mean([np.searchsorted(result['fit_score_references'][k],s[k],side='right')/len(result['fit_score_references'][k]) for k in RULES[:-1]]));return s

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);root=p.parse_args().root;result=json.loads((root/'results.json').read_text());data=json.loads((root/'features.json').read_text())['rows'];poses=sorted(set(v['pose']['id'] for v in data));noise=[]
 for pi,pose in enumerate(poses):
  base=next(v for v in data if v['pose']['id']==pose and v['state']=='present' and v['absorption']==.3 and v['scattering']==.5 and v['profile']=='uniform' and v['context']==0 and v['mode']=='both')
  with np.load(root/'controlled'/base['name']/'rir.npz') as z:rms=np.array([np.sqrt(np.mean(z[f'ch{c}'][48:3840].astype(float)**2)) for c in range(6)])
  selected=[v for v in data if v['pose']['id']==pose and v['state'] in ['present','removed'] and v['absorption'] in [.05,.7,.99] and v['profile']=='uniform' and v['context']==0 and v['mode']=='both']
  for v in selected:
   with np.load(root/'controlled'/v['name']/'rir.npz') as z:clean=[z[f'ch{c}'].astype(float) for c in range(6)]
   for label,db in [('clean',None),('moderate',20),('severe',0)]:
    waves=[x if db is None else x+np.random.default_rng(4211+pi*6+c).normal(size=len(x))*rms[c]*10**(-db/20) for c,x in enumerate(clean)];expected=np.array(v['features']['expected_ms']);f=v['features'] if db is None else features(waves,expected,expected);noise.append(dict(name=v['name'],pose=pose,state=v['state'],absorption=v['absorption'],scattering=v['scattering'],noise=label,scores=score(f,result),early_energy_mean=float(np.mean(f['early_energy'])),spectral_tilt_db=f['spectral_tilt_db'],peak_ms=f['peak_ms'],effective_snr_db=None if db is None else float(10*np.log10(max(np.mean([np.mean(x[48:3840]**2) for x in clean]),1e-30)/max(np.mean(rms*rms)*10**(-db/10),1e-30)))))
 noise_summary=[]
 for label in ['clean','moderate','severe']:
  for rule in RULES:noise_summary.append(dict(noise=label,rule=rule,**evaluate([v for v in noise if v['noise']==label],rule,result['models'][rule])))
 (root/'noise_results.json').write_text(json.dumps(dict(records=noise,summary=noise_summary,policy='Synthetic additive sensor noise, not new renderer outputs; fixed baseline RMS per pose, shared noise sequence across materials'),indent=2))
 inputs=json.loads((root/'hssd_inputs/manifest.json').read_text());manifest=json.loads((root/'hssd_render/manifest.json').read_text());assert len(manifest)==72;hssd=[]
 for rec in manifest:
  scene=next(v for v in inputs if v['scene']==rec['scene']);q=next(v for v in scene['queries'] if v['query_id']==rec['query_id']);path=root/'hssd_render'/rec['name']/'rir.npz';assert hashlib.sha256(path.read_bytes()).hexdigest()==rec['rir_sha256']
  with np.load(path) as z:channels=[z[f'ch{c}'] for c in range(6)]
  for c,x in enumerate(channels):assert hashlib.sha256(x.tobytes()).hexdigest()==rec['channel_bytes_sha256'][f'ch{c}']
  channels=[channels[c] for c in np.roll(np.arange(6),q['channel_roll'])];power=[x.astype(float)**2 for x in channels];candidates=[]
  for hyp in q['hypotheses']:
   allpaths=[p for p in hyp['paths'] if min(p['times_ms'])>=2 and max(p['times_ms'])<=79];supported=[p for p in allpaths if p['supported']]
   def energy(paths,shift=0):
    vals=[np.mean([power[c][int((t+shift-.5)*48):int((t+shift+.5)*48)+1].sum()/max(power[c][:48].sum(),1e-30) for c,t in enumerate(p['times_ms'])]) for p in paths if max(p['times_ms'])+shift+.5<80]
    return float(np.mean(vals)) if vals else None
   candidates.append(dict(rank=hyp['rank'],supported_paths=len(supported),unknown_paths=len(allpaths)-len(supported),supported_energy=energy(supported),unknown_energy=energy([p for p in allpaths if not p['supported']]),shifted_supported_energy=energy(supported,5)))
  be=[]
  for x in channels:
   z=np.fft.rfft(x[48:3840],n=4096);freq=np.fft.rfftfreq(4096,1/48000);be.append([float((abs(z[(freq>=lo)&(freq<hi)])**2).sum()) for lo,hi in [(20,176.020),(176.020,774.604),(774.604,3408.674),(3408.674,20000)]])
  hssd.append(dict(**{k:rec[k] for k in ['name','scene','query_id','absorption','scattering','profile','context']},early_energy=float(np.mean([x[48:3840].sum() for x in power])),direct_normalized_early=float(np.mean([x[48:3840].sum()/max(x[:48].sum(),1e-30) for x in power])),band_energy=np.mean(be,axis=0).tolist(),candidate_evidence=candidates))
 contrasts=[]
 for sid in sorted(set(v['scene'] for v in hssd)):
  for qid in sorted(set(v['query_id'] for v in hssd if v['scene']==sid)):
   rs=[v for v in hssd if v['scene']==sid and v['query_id']==qid]
   def group(alpha,scatter,profile):return [v for v in rs if v['absorption']==alpha and v['scattering']==scatter and v['profile']==profile]
   groups=[group(.05,0,'uniform'),group(.05,.5,'uniform'),group(.7,.5,'uniform'),group(.3,.5,'uniform'),group(.3,.5,'low'),group(.3,.5,'high')];gmean=[np.mean([v['early_energy'] for v in g]) for g in groups];rankings=[]
   for g in groups:
    vec=np.array([[np.nan if c['supported_energy'] is None else c['supported_energy'] for c in v['candidate_evidence']] for v in g]);avg=np.full(vec.shape[1],np.nan)
    for k in range(vec.shape[1]):
     if np.isfinite(vec[:,k]).any():avg[k]=np.nanmean(vec[:,k])
    rankings.append(np.argsort(np.nan_to_num(avg,nan=-1))[::-1].tolist())
   contrasts.append(dict(scene=sid,query_id=qid,mean_early_energy_by_condition=gmean,absorption70_over05=float(gmean[2]/gmean[1]),scattering05_over0=float(gmean[1]/gmean[0]),top_candidate_by_condition=[v[0] for v in rankings],candidate_order_by_condition=rankings,interpretation='Ranking of untrained energy diagnostic only; no GT pose filtering or localization accuracy'))
 (root/'hssd_results.json').write_text(json.dumps(dict(records=hssd,contrasts=contrasts,cost=dict(rings=len(manifest),bytes=sum(v['bytes'] for v in manifest),wall_seconds=sum(v['wall_seconds'] for v in manifest)),limitations='Native stage only; assigned vertical-face proxy, not validated semantic walls; same source mesh/FP/RGB remain unchanged; Train only; four poses in two reused scenes'),indent=2));print('noise',len(noise),'HSSD',len(hssd))
if __name__=='__main__':main()

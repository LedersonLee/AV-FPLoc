"""Independent geometric minimization, complete input checks, and explicit D gate."""
import argparse,json,hashlib,sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from analysis import analytic,METHODS,estimates

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root;old=root.parent/'reflection_observability_20261111';data=json.loads((root/'analysis.json').read_text());checks={}
 for x in data['inputs']:assert hashlib.sha256((old/x['path']).read_bytes()).hexdigest()==x['sha256']
 assert len(data['inputs'])==96 and data['previous_peak_max_replay_error_ms']==0;checks['old_rir_hashes']=96;checks['old_diagnostic_peak_replays']=60
 sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reflection_observability_20261111'));from geometry import paths
 comparisons=[]
 for d in [.3,2.,6.]:
  for moved in [False,True]:
   z=-1 if moved else 0;pred=paths([.17,d,0],np.array([[[-12,z],[12,z]]]))[0]['times_ms'];exact=analytic(d,moved);np.testing.assert_allclose(pred,exact,atol=1e-11)
   raw=np.zeros((6,3840))
   for ch in range(6):
    theta=np.pi+ch*np.pi/3;s=np.array([.17,d]);r=s+np.array([.05*np.cos(theta),-.05*np.sin(theta)])
    def length(qx):q=np.array([qx,z]);return np.linalg.norm(s-q)+np.linalg.norm(r-q)
    optimum=minimize_scalar(length,bounds=(-12,12),method='bounded',options={'xatol':1e-12});assert optimum.success;t=length(optimum.x)/343*1000;assert abs(t-exact[ch])<1e-9
    sample=t*48;n=int(sample);f=sample-n;raw[ch,n]=1-f;raw[ch,n+1]=f
   err=float(abs(raw.argmax(-1)/48-exact).max());assert err<1/48;file=root/f'independent_minimizer_reference_d{d}_m{int(moved)}.npy'
   if not file.exists():np.save(file,raw)
   comparisons.append(dict(distance=d,moved=moved,max_peak_error_ms=err))
 checks['independent_fermat_paths']=36;checks['analytic_reference']=comparisons
 # Check actual acoustic mesh export; not only visual bounding box.
 manifests=json.loads((root/'backend_v2/manifest.json').read_text());meshes=0;waves=0
 for r in manifests:
  for f in r['files']:assert hashlib.sha256((root/'backend_v2'/f['name']).read_bytes()).hexdigest()==f['sha256'];waves+=1
  vertices=[]
  for line in (root/'backend_v2'/(r['file_prefix']+'.obj')).read_text().splitlines():
   if line.startswith('# Listeners'):break
   if line.startswith('v '):vertices.append([float(v) for v in line.split()[1:4]])
  vertices=np.array(vertices);assert len(vertices)==8;assert np.sum(np.isclose(vertices[:,2],0))==4;np.testing.assert_allclose(vertices.min(0),[-12,-20,-12],atol=1e-6);np.testing.assert_allclose(vertices.max(0),[12,12,24],atol=1e-6);meshes+=1
  for rec in r['records']:assert rec['ray_hit']==1 and abs(rec['ray_distance']-r['distance'])<1e-5
 assert waves==276;checks['new_wave_hashes']=waves;checks['acoustic_mesh_exports']=meshes;checks['first_hit_queries']=len(data['geometry_checks'])
 # Gate requires both evaluation distances and moved response timing, not a low variance alone.
 gate=[]
 for method in METHODS:
  records=[x for x in data['estimator_summary'] if x['method']==method and x['threshold']==1e-6 and x['distance'] in [2.,6.]]
  ok=all(x['detected_channels']==6 and x['max_abs_error_ms']<=.1 and x['max_abs_shift_error_ms']<=.2 and x['removed_false_detection_fraction']<=.05 and x['repeat_sd_mean_ms'] is not None and x['repeat_sd_mean_ms']<=.1 for x in records)
  gate.append(dict(method=method,passes=bool(ok)))
 assert not any(x['passes'] for x in gate)
 # Recompute independent threshold crossing for evaluation6m, all channels.
 raw=np.load(old/'controlled/single_d6.0_base/rir.npy');removed=np.load(old/'controlled/single_d6.0_removed/rir.npy');manual=[]
 for ch in range(6):
  reference=(removed[ch,:48]**2).max();power=raw[ch].astype(float)**2/reference;env=np.convolve(power,np.ones(6)/6,mode='full')[:len(power)];ix=np.flatnonzero(env[48:3840]>1e-6)[0]+48;expected=ix/48;calc=estimates(raw[ch],48000,float(reference),removed[ch])['energy_onset'];assert abs(expected-calc)<1e-12;manual.append(calc)
 checks['independent_onset_replay']=6
 result=dict(passed=True,checks=checks,method_gate=gate,hssd_gate='FAIL — no physically validated arrival estimator; D skipped as preauthorized',outcome='B — Limited Observability',limits='Analytic reference validates geometry only; not SoundSpaces path GT. Fresh contexts are not independent scenes. Internal histogram/early-path timing unexported.')
 (root/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()

"""Fixed-estimator audit of the single-setting renderer mechanism intervention."""
import argparse,json,hashlib,copy
from pathlib import Path
import numpy as np
from analysis import estimates,analytic

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);root=p.parse_args().root
 manifest=json.loads((root/'specular_v3/manifest.json').read_text());rows=[];checked=[]
 for a in manifest:
  folder=root/'specular_v3';prefix=a['file_prefix'];d=a['distance'];mode=a['mode'];moved=mode=='specular_moved'
  assert hashlib.sha256((folder/(prefix+'.json')).read_bytes()).hexdigest()==a['receipt_sha256']
  for f in a['files']:assert hashlib.sha256((folder/f['name']).read_bytes()).hexdigest()==f['sha256']
  mat=json.loads((folder/(prefix+'_material.json')).read_text());reference=json.loads((folder/f'd{float(d)}_explicit_default_c0_material.json').read_text());target=copy.deepcopy(reference)
  if mode!='explicit_default':target['materials'][0]['scattering'][1::2]=[0.,0.]
  assert mat==target
  mesh=(folder/(prefix+'.obj')).read_text();value=.5 if mode=='explicit_default' else 0
  assert f'# Scattering - Index:0, Value: {value:.6f}' in mesh
  assert '# Absorption - Index:0, Value: 0.100000' in mesh and '# Transmission - Index:0, Value: 0.000000' in mesh
  vertices=np.array([[float(v) for v in l.split()[1:4]] for l in mesh.split('# Listeners')[0].splitlines() if l.startswith('v ')]);assert len(vertices)==8
  np.testing.assert_allclose(vertices[:4,2],-1 if moved else 0)
  rm=np.load(root.parent/f'reflection_observability_20261111/controlled/single_d{float(d)}_removed/rir.npy')
  for rec in a['records']:
   c=rec['channel'];assert abs(rec['ray_distance']-(d+int(moved)))<1e-5
   x=np.fromfile(folder/f'{prefix}_r0_ch{c}.f32',dtype=np.float32);assert len(x)==rec['samples'] and np.isfinite(x).all()
   for threshold in [1e-8,1e-6,1e-4]:rows.append(dict(mode=mode,distance=d,context=a['context'],channel=c,threshold=threshold,expected_ms=float(analytic(d,moved)[c]),methods=estimates(x,48000,float((rm[c,:48]**2).max()),rm[c],threshold)))
  checked.append(dict(prefix=prefix,material_sha256=hashlib.sha256((folder/(prefix+'_material.json')).read_bytes()).hexdigest(),mesh_sha256=hashlib.sha256((folder/(prefix+'.obj')).read_bytes()).hexdigest()))
 summary=[];effects=[]
 for d in [2,6]:
  for mode in ['explicit_default','specular','specular_radius0','specular_moved']:
   rs=[r for r in rows if r['distance']==d and r['mode']==mode and r['threshold']==1e-6]
   for method in ['wave_peak','energy_onset','energy_centroid','removed_difference_onset','band_high_difference_onset']:
    errors=np.array([r['methods'][method]-r['expected_ms'] for r in rs]);cm=[np.mean([r['methods'][method]-r['expected_ms'] for r in rs if r['context']==c]) for c in range(3)]
    sds=[np.std([r['methods'][method] for r in rs if r['channel']==c],ddof=1) for c in range(6)]
    shift=[]
    if mode=='specular_moved':
     for r in rs:
      b=next(b for b in rows if b['distance']==d and b['mode']=='specular' and b['channel']==r['channel'] and b['context']==r['context'] and b['threshold']==1e-6)
      shift.append((r['methods'][method]-b['methods'][method])-(r['expected_ms']-b['expected_ms']))
    summary.append(dict(distance=d,mode=mode,method=method,bias_ms=float(errors.mean()),max_abs_error_ms=float(abs(errors).max()),context_mean_range_ms=[min(cm),max(cm)],repeat_sd_mean_ms=float(np.mean(sds)),repeat_sd_max_ms=float(max(sds)),shift_error_mean_ms=float(np.mean(shift)) if shift else None,shift_error_max_abs_ms=float(max(abs(np.array(shift)))) if shift else None))
  a=next(x for x in summary if x['distance']==d and x['mode']=='explicit_default' and x['method']=='wave_peak');b=next(x for x in summary if x['distance']==d and x['mode']=='specular' and x['method']=='wave_peak');effects.append(dict(distance=d,specular_minus_default_peak_ms=b['bias_ms']-a['bias_ms']))
 cost={}
 for folder in ['backend_v2','specular_v3']:
  m=json.loads((root/folder/'manifest.json').read_text());cost[folder]=dict(contexts=len(m),mono_outputs=sum(len(r['files']) for r in m),bytes=sum(f['bytes'] for r in m for f in r['files']),render_call_seconds=sum(r['seconds'] for r in m),process_wall_seconds=sum(r['wall_seconds'] for r in m))
 result=dict(rows=rows,summary=summary,effects=effects,checks=dict(mono_hashes=144,material_only_scattering_pairs=24,acoustic_exports=24,receipts=24),inputs=checked,cost=cost,outcome='C — Renderer-dependent Artifact',hssd_gate='FAIL: no validated default-renderer path timing; default-vs-specular effect is not a validated HSSD arrival estimator')
 (root/'specular_analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(summary=summary,effects=effects,cost=cost),indent=2))
if __name__=='__main__':main()

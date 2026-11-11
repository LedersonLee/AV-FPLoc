"""Independent input, arithmetic, UI-source and scope checks for Experiment4A."""
import argparse,hashlib,json,re,itertools
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root;manifest=json.loads((root/'controlled/manifest.json').read_text());assert len(manifest)==1980
 keys=set();raw_count=0;export_count=0
 for v in manifest:
  d=root/'controlled'/v['name'];assert sha(d/'rir.npz')==v['rir_sha256'];assert sha(d/'mesh.txt')==v['mesh_sha256'];assert sha(d/'material.json')==v['material_sha256'];assert sha(d/'raw.obj')==v['export_sha256'];mat=json.loads((d/'material.json').read_text());coeff=mat['materials'][0]['absorption'][1::2];assert abs(np.mean(coeff)-v['absorption'])<1e-12;assert all(0<=x<=1 for x in coeff);assert mat['materials'][0]['transmission'][1::2]==[0,0]
  if v['mode']=='both':keys.add((v['pose']['id'],v['state'],v['absorption'],v['scattering'],v['profile'],v['context']))
  blocks=re.split(r'# Material Index : \d+\n',(d/'raw.obj').read_text().split('# Vertex Count')[0])[1:];assert blocks
  applied=v['pose']['kind']!='single' or v['state']!='removed';expected=coeff if applied else [.1,.1];absvals=[float(x) for x in re.findall(r'# Absorption - Index:\d+, Value: ([\d.]+)',blocks[0])];np.testing.assert_allclose(absvals,expected,atol=1e-6);scatter=[float(x) for x in re.findall(r'# Scattering - Index:\d+, Value: ([\d.]+)',blocks[0])];np.testing.assert_allclose(scatter,v['scattering'] if applied else .5,atol=1e-6);export_count+=1
  with np.load(d/'rir.npz') as z:
   for c in range(6):assert sha_bytes(z[f'ch{c}'])==v['channel_bytes_sha256'][f'ch{c}'];raw_count+=1
 expected=set(itertools.product(['single2','single6','room_near','room_far','corridor_near','corridor_far'],['present','removed','moved'],[.05,.3,.7,.99],[0.,.5,1.],['uniform','low','high'],range(3)));assert keys==expected
 features=json.loads((root/'features.json').read_text());byname={v['name']:v for v in features['rows']};checks=[]
 # Independent scalar geometry minimization and direct energy arithmetic, perpose/state all channels.
 for pose in sorted(set(v['pose']['id'] for v in manifest)):
  for state in ['present','removed','moved']:
   v=next(v for v in manifest if v['pose']['id']==pose and v['state']==state and v['absorption']==.3 and v['scattering']==.5 and v['profile']=='uniform' and v['context']==0 and v['mode']=='both');f=byname[v['name']]['features'];s=np.array(v['geometry']['source']);wall=-1 if state=='moved' else 0
   with np.load(root/'controlled'/v['name']/'rir.npz') as z:
    for c in range(6):
     theta=np.pi+c*np.pi/3;r=s+np.array([.05*np.cos(theta),0,-.05*np.sin(theta)])
     def length(qx):q=np.array([qx,1.5,wall]);return np.linalg.norm(s-q)+np.linalg.norm(r-q)
     opt=minimize_scalar(length,bounds=(-12,12),method='bounded');t=opt.fun/343*1000;assert abs(t-f['expected_ms'][c])<1e-7;x=z[f'ch{c}'].astype(float);np.testing.assert_allclose((x[48:3840]**2).sum(),f['early_energy'][c],rtol=1e-12);ref=(x[:48]**2).max();env=np.convolve(x*x,np.ones(6)/6)[:len(x)];ids=np.flatnonzero(env[48:3840]>ref*1e-6);onset=float((ids[0]+48)/48) if len(ids) else None;assert onset==f['onset_ms'][c];checks.append((pose,state,c))
 # Every snapshot uses the same mesh hash within pose/state across material conditions.
 for pose in sorted(set(v['pose']['id'] for v in manifest)):
  for state in ['present','removed','moved']:assert len(set(v['mesh_sha256'] for v in manifest if v['pose']['id']==pose and v['state']==state))==1
 result=json.loads((root/'results.json').read_text());ev=[r for r in result['predictions'] if not(r['pose']=='single2' and r['profile']=='uniform')]
 for rule in ['peak','sharp','time','band','channel','combined']:
  y=np.array([r['state']!='removed' for r in ev]);pred=np.array([r['predictions'][rule] for r in ev]);summary=next(x for x in result['reliability'] if x['group']=='evaluation' and x['rule']==rule);assert abs(pred[~y].mean()-summary['fpr'])<1e-12 and abs(pred[y].mean()-summary['tpr'])<1e-12
 selection=result['representative'];assert selection['default']==min(selection['distances'],key=selection['distances'].get);assert 'single6' in selection['exceptions'];failure=next(r for r in result['predictions'] if r['pose']=='single6' and r['state']=='moved' and r['context']==0 and not r['predictions']['combined']);assert failure['name'] in byname
 hssd=json.loads((root/'hssd_render/manifest.json').read_text());assert len(hssd)==72
 for v in hssd:
  d=root/'hssd_render'/v['name'];assert sha(d/'rir.npz')==v['rir_sha256'];assert sha(d/'material.json')==v['material_sha256'];assert sha(d/'raw.obj')==v['export_sha256']
  with np.load(d/'rir.npz') as z:
   for c in range(6):assert sha_bytes(z[f'ch{c}'])==v['channel_bytes_sha256'][f'ch{c}']
 for v in json.loads((root/'hssd_inputs/manifest.json').read_text()):assert sha(Path(v['source']))==v['source_sha256'];assert sha(root/'hssd_inputs'/v['mesh_file'])==v['mesh_sha256']
 out=dict(passed=True,controlled_rings=len(manifest),controlled_mono_hashes=raw_count,complete_factorial_cells=len(keys),acoustic_exports_checked=export_count,independent_geometry_energy_onset_checks=len(checks),reliability_rules_recounted=6,hssd_rings=len(hssd),hssd_raw_mono_hashes=len(hssd)*6,default_representative=selection['default'],exceptions=selection['exceptions'],saved_failure_case=failure['name'],limits='Per-path identity unverified; HSSD native stage vertical proxy only; browser checks recorded separately; no Test payload use')
 (root/'verification.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
def sha_bytes(x):return hashlib.sha256(x.tobytes()).hexdigest()
if __name__=='__main__':main()

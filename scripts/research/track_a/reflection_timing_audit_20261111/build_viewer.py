"""Standalone viewer of retained RIRs; only display aggregates, never synthetic evidence."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from scipy.signal import lfilter
from analysis import analytic

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();r=a.root;old=r.parent/'reflection_observability_20261111';data=json.loads((r/'analysis.json').read_text());sp=json.loads((r/'specular_analysis.json').read_text());cases=[]
 def add(name,d,mode,raw,records,source,hashes):
  channels=[]
  for c,x in enumerate(raw):
   y=np.zeros(3840);y[:min(len(x),3840)]=x[:3840];scale=max(float(abs(y[48:]).max()),1e-30);z=y/scale;energy=lfilter(np.ones(6)/6,[1],y*y);energy/=max(float(energy[48:].max()),1e-30)
   bins=z.reshape(480,8);env=energy.reshape(480,8).max(1);rec=next(k for k in records if k['channel']==c)
   channels.append(dict(lo=np.round(bins.min(1),5).tolist(),hi=np.round(bins.max(1),5).tolist(),energy=np.round(env,5).tolist(),methods=rec['methods'],expected=rec['expected_ms'],scale=scale))
  cases.append(dict(name=name,d=d,mode=mode,channels=channels,source=source,hashes=hashes))
 for d in [.3,2.,6.]:
  for mode in ['base','moved','removed','repeat']:
   name=f'single_d{d}_{mode}';f=old/f'controlled/{name}/rir.npy';records=[x for x in data['previous_estimators'] if x['name']==name and x['threshold']==1e-6];add('Previous · '+name,d,mode,np.load(f),records,'Experiment 1 retained recording',[hashlib.sha256(f.read_bytes()).hexdigest()])
 for folder,modes,records in [('backend_v2',['baseline','radius0','indirect_off','direct_off','temporal_on','simultaneous'],data['backend_estimators']),('specular_v3',['explicit_default','specular','specular_radius0','specular_moved'],sp['rows'])]:
  manifest=json.loads((r/folder/'manifest.json').read_text())
  for v in manifest:
   if v['context']!=0 or v['mode'] not in modes:continue
   raw=[np.fromfile(r/folder/f"{v['file_prefix']}_r0_ch{c}.f32",dtype=np.float32) for c in range(6)];rs=[x for x in records if x['mode']==v['mode'] and x['distance']==v['distance'] and x['context']==0 and x.get('threshold',1e-6)==1e-6];add('New · '+v['file_prefix'],v['distance'],v['mode'],raw,rs,'Documented RLR C API; fresh context 0',[f['sha256'] for f in v['files']])
 template=Path(__file__).with_name('viewer_template.html').read_text();a.out.write_text(template.replace('__DATA__',json.dumps(cases,separators=(',',':'))));print(len(cases),'cases',a.out.stat().st_size,'bytes')
if __name__=='__main__':main()

"""Small native-stage material intervention, not a furnished-scene material claim."""
import argparse,json,hashlib,subprocess,time
from pathlib import Path
import numpy as np
from design import material

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--exe',type=Path,required=True);a=p.parse_args();out=a.root/'hssd_render';out.mkdir(exist_ok=False);inputs=json.loads((a.root/'hssd_inputs/manifest.json').read_text());default=json.loads(Path('/opt/src/habitat-sim/src/deps/rlr-audio-propagation/RLRAudioPropagationPkg/data/mp3d_material_config.json').read_text())['materials'][0];rows=[]
 for scene in inputs:
  for q in scene['queries']:
   for alpha,scattering,profile in [(.05,0.,'uniform'),(.05,.5,'uniform'),(.7,.5,'uniform'),(.3,.5,'uniform'),(.3,.5,'low'),(.3,.5,'high')]:
    for context in range(3):
     name=f"{scene['scene']}_q{q['query_id']}_a{alpha:.2f}_s{scattering:.1f}_{profile}_r{context}";dest=out/name;dest.mkdir();mat=material(default,alpha,scattering,profile);(dest/'material.json').write_text(json.dumps(mat));start=time.perf_counter()
     with (dest/'render.log').open('w') as log:subprocess.run([str(a.exe),str(dest/'raw'),str(a.root/'hssd_inputs'/scene['mesh_file']),str(dest/'material.json'),*map(str,q['source']),'both','1'],stdout=log,stderr=subprocess.STDOUT,check=True)
     rec=json.loads((dest/'raw.json').read_text());waves={f'ch{c}':np.fromfile(dest/f'raw_ch{c}.f32',dtype=np.float32) for c in range(6)};assert all(len(waves[f'ch{c}'])==rec['lengths'][c] for c in range(6));np.savez_compressed(dest/'rir.npz',**waves)
     with np.load(dest/'rir.npz') as z:
      for c in range(6):np.testing.assert_array_equal(z[f'ch{c}'],waves[f'ch{c}'])
     hashes={k:hashlib.sha256(x.tobytes()).hexdigest() for k,x in waves.items()}
     for c in range(6):(dest/f'raw_ch{c}.f32').unlink()
     row=dict(name=name,scene=scene['scene'],query_id=q['query_id'],source=q['source'],absorption=alpha,scattering=scattering,profile=profile,context=context,renderer=rec,rir_sha256=sha(dest/'rir.npz'),channel_bytes_sha256=hashes,material_sha256=sha(dest/'material.json'),export_sha256=sha(dest/'raw.obj'),bytes=(dest/'rir.npz').stat().st_size,wall_seconds=time.perf_counter()-start,native_stage_only=True);(dest/'receipt.json').write_text(json.dumps(row,indent=2));rows.append(row);(out/'manifest.json').write_text(json.dumps(rows,indent=2));print(len(rows),name,round(row['wall_seconds'],2),flush=True)
if __name__=='__main__':main()

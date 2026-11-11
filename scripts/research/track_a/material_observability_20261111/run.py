import argparse,hashlib,itertools,json,random,subprocess,time,shutil
from pathlib import Path
import numpy as np
from design import POSES,mesh,material

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--exe',type=Path,required=True);p.add_argument('--pilot',action='store_true');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 default=json.loads(Path('/opt/src/habitat-sim/src/deps/rlr-audio-propagation/RLRAudioPropagationPkg/data/mp3d_material_config.json').read_text())['materials'][0]
 jobs=[]
 if a.pilot:
  for pose in [POSES[0],POSES[2],POSES[4]]:
   for alpha,scatter,profile in [(.05,0.,'uniform'),(.7,1.,'low')]:jobs.append((pose,'present',alpha,scatter,profile,0,'both'))
 else:
  for pose,state,alpha,scatter,profile,ctx in itertools.product(POSES,['present','removed','moved'],[.05,.30,.70,.99],[0.,.5,1.],['uniform','low','high'],range(3)):jobs.append((pose,state,alpha,scatter,profile,ctx,'both'))
  for pose,ctx,mode in itertools.product(POSES,range(3),['direct_only','indirect_only']):jobs.append((pose,'present',.3,.5,'uniform',ctx,mode))
  random.Random(4211).shuffle(jobs)
 (a.out/'planned_jobs.json').write_text(json.dumps([dict(pose=p['id'],state=s,absorption=x,scattering=y,profile=f,context=c,mode=m) for p,s,x,y,f,c,m in jobs],indent=2));manifest=[]
 for i,(pose,state,alpha,scatter,profile,ctx,mode) in enumerate(jobs):
  if shutil.disk_usage(a.out).free<3_000_000_000:raise RuntimeError('Storage guard: less than3GB free; preserve progress')
  name=f"{pose['id']}_{state}_a{alpha:.2f}_s{scatter:.1f}_{profile}_r{ctx}_{mode}";dest=a.out/name;dest.mkdir();prefix=dest/'raw';geo=mesh(pose,state,dest/'mesh.txt');mat=material(default,alpha,scatter,profile);(dest/'material.json').write_text(json.dumps(mat));start=time.perf_counter()
  with (dest/'render.log').open('w') as log:subprocess.run([str(a.exe),str(prefix),str(dest/'mesh.txt'),str(dest/'material.json'),*map(str,geo['source']),mode,'1'],stdout=log,stderr=subprocess.STDOUT,check=True)
  receipt=json.loads((dest/'raw.json').read_text());channels={f'ch{ch}':np.fromfile(dest/f'raw_ch{ch}.f32',dtype=np.float32) for ch in range(6)}
  assert all(len(channels[f'ch{ch}'])==receipt['lengths'][ch] and np.isfinite(channels[f'ch{ch}']).all() for ch in range(6))
  np.savez_compressed(dest/'rir.npz',**channels)
  with np.load(dest/'rir.npz') as packed:
   for ch in range(6):np.testing.assert_array_equal(channels[f'ch{ch}'],packed[f'ch{ch}'])
  source_hashes={f'ch{ch}':sha(dest/f'raw_ch{ch}.f32') for ch in range(6)}
  for ch in range(6):(dest/f'raw_ch{ch}.f32').unlink() # Own verified lossless intermediate only.
  row=dict(name=name,pose=pose,geometry=geo,state=state,absorption=alpha,scattering=scatter,profile=profile,context=ctx,mode=mode,renderer=receipt,settings=dict(sample_rate=48000,indirect_rays=20000,source_rays=512,depth=50,threads=2,diffraction=True,transmission=False,temporal_coherence=False),material=mat,rir_sha256=sha(dest/'rir.npz'),channel_bytes_sha256=source_hashes,material_sha256=sha(dest/'material.json'),mesh_sha256=sha(dest/'mesh.txt'),export_sha256=sha(dest/'raw.obj'),bytes=(dest/'rir.npz').stat().st_size,raw_bytes=sum(x.nbytes for x in channels.values()),wall_seconds=time.perf_counter()-start)
  (dest/'receipt.json').write_text(json.dumps(row,indent=2));manifest.append(row);(a.out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(i+1,len(jobs),name,round(row['wall_seconds'],3),row['bytes'],flush=True)
if __name__=='__main__':main()

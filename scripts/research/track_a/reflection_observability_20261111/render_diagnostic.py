"""Post-primary falsification of reflection delay; no HSSD-derived calibration."""
import sys,json,time,argparse,hashlib
from pathlib import Path
import numpy as np
sys.path[:0]=['/workspace/AV-FPLoc/src','/workspace/3Dto2D_FP/scripts']
from avfploc.preprocessing.hssd.hssd_positive_pairs import make_geometry_simulator,RING
from render_hssd_paired_rirs import _add_audio_sensor
from types import SimpleNamespace

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();dest=a.out/'diagnostic';dest.mkdir(exist_ok=True)
 configs=[('baseline',48000,20000,50,True),('fs24000',24000,20000,50,True),('fs44100',44100,20000,50,True),('fs96000',96000,20000,50,True),('no_diffraction',48000,20000,50,False),('depth1',48000,20000,1,False),('rays200k',48000,200000,50,False)]
 for distance in [2.,6.]:
  for label,fs,rays,depth,diff in configs:
   for repeat in range(3):
    path=dest/f'd{distance}_{label}_r{repeat}.npy'
    if path.with_suffix('.json').exists():continue
    stage=a.out/f'controlled/single_d{distance}_base/mesh_glb.stage_config.json';start=time.perf_counter();sim=make_geometry_simulator(stage,None);cfg=SimpleNamespace(sample_rate=fs,indirect_rays=rays,ray_depth=depth,source_rays=512,diffraction=diff,max_diffraction_order=10,acoustic_threads=2);sensor=_add_audio_sensor(sim,cfg);center=np.array([.17,1.5,distance],np.float32);channels=[]
    try:
     for ang in RING:
      agent=sim.get_agent(0);state=agent.get_state();state.position=center+np.array([.05*np.cos(ang),0,-.05*np.sin(ang)]);state.sensor_states={};agent.set_state(state,reset_sensors=True);sensor.setAudioSourceTransform(center);channels.append(np.asarray(sim.get_sensor_observations()['audio_sensor'])[0].copy())
     raw=np.stack([x[:min(map(len,channels))] for x in channels]);np.save(path,raw);r=dict(label=label,distance=distance,repeat=repeat,settings=vars(cfg),shape=list(raw.shape),seconds=time.perf_counter()-start,sha256=hashlib.sha256(path.read_bytes()).hexdigest());path.with_suffix('.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
    finally:sim.close()
if __name__=='__main__':main()

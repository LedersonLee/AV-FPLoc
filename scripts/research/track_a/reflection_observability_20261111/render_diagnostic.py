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
 configs=[('baseline',48000,20000,50,True),('fs24000',24000,20000,50,True),('fs44100',44100,20000,50,True),('fs96000',96000,20000,50,True),('no_diffraction',48000,20000,50,False),('depth1',48000,20000,1,False),('rays200k',48000,200000,50,False),('closed_box',48000,20000,50,True),('one_sided',48000,20000,50,True)]
 for distance in [2.,6.]:
  for label,fs,rays,depth,diff in configs:
   for repeat in range(3):
    path=dest/f'd{distance}_{label}_r{repeat}.npy'
    if path.with_suffix('.json').exists():continue
    stage=a.out/f'controlled/single_d{distance}_base/mesh_glb.stage_config.json'
    if label in ['closed_box','one_sided']:
     sys.path.insert(0,str(a.out/'deps'));import trimesh
     gd=dest/(label+'_geometry');gd.mkdir(exist_ok=True)
     floor=trimesh.Trimesh([[-12,-20,-12],[12,-20,-12],[12,-20,24],[-12,-20,24]],[[0,2,1],[0,3,2]],process=False)
     if label=='closed_box':wall=trimesh.creation.box([24,20,.1],transform=trimesh.transformations.translation_matrix([0,2,-.05]))
     else:wall=trimesh.Trimesh([[-12,-8,0],[12,-8,0],[12,12,0],[-12,12,0]],[[0,1,2],[0,2,3]],process=False)
     mesh=gd/'mesh.glb'
     if not mesh.exists():trimesh.util.concatenate([floor,wall]).export(mesh)
     stage=gd/'mesh.stage_config.json'
     if not stage.exists():stage.write_text(json.dumps(dict(render_asset='mesh.glb',up=[0,1,0],front=[0,0,-1],units_to_meters=1.)))
    start=time.perf_counter();sim=make_geometry_simulator(stage,None);cfg=SimpleNamespace(sample_rate=fs,indirect_rays=rays,ray_depth=depth,source_rays=512,diffraction=diff,max_diffraction_order=10,acoustic_threads=2);sensor=_add_audio_sensor(sim,cfg);center=np.array([.17,1.5,distance],np.float32);channels=[]
    try:
     for ang in RING:
      agent=sim.get_agent(0);state=agent.get_state();state.position=center+np.array([.05*np.cos(ang),0,-.05*np.sin(ang)]);state.sensor_states={};agent.set_state(state,reset_sensors=True);sensor.setAudioSourceTransform(center);channels.append(np.asarray(sim.get_sensor_observations()['audio_sensor'])[0].copy())
     raw=np.stack([x[:min(map(len,channels))] for x in channels]);np.save(path,raw);r=dict(label=label,distance=distance,repeat=repeat,settings=vars(cfg),shape=list(raw.shape),seconds=time.perf_counter()-start,sha256=hashlib.sha256(path.read_bytes()).hexdigest());path.with_suffix('.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
    finally:sim.close()
if __name__=='__main__':main()

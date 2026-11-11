"""Independent SoundSpaces mesh interventions; never synthesize image-source RIRs."""
import argparse, json, time, hashlib, sys
from pathlib import Path
import numpy as np
sys.path[:0]=['/workspace/AV-FPLoc/src','/workspace/3Dto2D_FP/scripts']
from avfploc.preprocessing.hssd.hssd_positive_pairs import make_geometry_simulator,RING
from render_hssd_paired_rirs import _add_audio_sensor
from types import SimpleNamespace
sys.path.insert(0,"/output/deps")
import trimesh

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--limit',type=int);a=p.parse_args();a.out.mkdir(exist_ok=True,parents=True)
 count=0
 for kind,verts in [('single',[[-12,0],[12,0]]),('room',[[-4,0],[4,0],[4,8],[-4,8],[-4,0]]),('corridor',[[-1.2,0],[1.2,0],[1.2,18],[-1.2,18],[-1.2,0]])]:
  walls=np.array(list(zip(verts[:-1],verts[1:])),float)
  for distance in [.3,2.,6.]:
   for variant in ['base','repeat','removed','moved']:
    name=f'{kind}_d{distance}_{variant}';dest=a.out/name
    if (dest/'receipt.json').exists():continue
    if a.limit is not None and count>=a.limit:return
    dest.mkdir(exist_ok=True);segments=walls.copy()
    if variant=='removed':segments=segments[1:]
    if variant=='moved':segments[0,:,1]-=1
    vertices=[];faces=[]
    for seg in segments:
     offset=len(vertices)
     vertices.extend([[seg[0,0],-8,seg[0,1]],[seg[1,0],-8,seg[1,1]],[seg[1,0],12,seg[1,1]],[seg[0,0],12,seg[0,1]]])
     faces.extend([[offset+i for i in f] for f in [[0,1,2],[0,2,3],[2,1,0],[3,2,0]]])
    # Remote floor guarantees nondegenerate navigation bounds, arrival >100ms.
    offset=len(vertices);vertices.extend([[-12,-20,-12],[12,-20,-12],[12,-20,24],[-12,-20,24]])
    faces.extend([[offset+i for i in f] for f in [[0,2,1],[0,3,2]]])
    vertices=np.array(vertices)
    trimesh.Trimesh(vertices,faces,process=False).export(dest/'mesh.glb')
    stage=dest/'mesh_glb.stage_config.json';stage.write_text(json.dumps(dict(render_asset='mesh.glb',up=[0,1,0],front=[0,0,-1],units_to_meters=1.)))
    start=time.perf_counter();sim=make_geometry_simulator(stage,None)
    cfg=SimpleNamespace(sample_rate=48000,indirect_rays=20000,ray_depth=50,source_rays=512,diffraction=True,max_diffraction_order=10,acoustic_threads=2)
    sensor=_add_audio_sensor(sim,cfg);center=np.array([.17,1.5,distance],np.float32);channels=[]
    try:
     box=sim.get_active_scene_graph().get_root_node().cumulative_bb;bounds=[np.asarray(box.min).tolist(),np.asarray(box.max).tolist()]
     np.testing.assert_allclose(bounds,[vertices.min(0),vertices.max(0)],atol=1e-4)
     for ang in RING:
      agent=sim.get_agent(0);state=agent.get_state();state.position=center+np.array([.05*np.cos(ang),0,-.05*np.sin(ang)]);state.sensor_states={};agent.set_state(state,reset_sensors=True);sensor.setAudioSourceTransform(center);channels.append(np.asarray(sim.get_sensor_observations()['audio_sensor'])[0].copy())
     length=min(map(len,channels));rir=np.stack([x[:length] for x in channels]);np.save(dest/'rir.npy',rir)
     rec=dict(name=name,kind=kind,distance=distance,variant=variant,segments=segments.tolist(),base_segments=walls.tolist(),center=center.tolist(),shape=list(rir.shape),seconds=time.perf_counter()-start,bytes=(dest/'rir.npy').stat().st_size,settings=vars(cfg),max_ir_length_default_s=4,temporal_coherence=False,materials=False,transmission=False,loaded_bounds=bounds,rir_sha256=hashlib.sha256((dest/'rir.npy').read_bytes()).hexdigest())
     (dest/'receipt.json').write_text(json.dumps(rec,indent=2));print(json.dumps(rec),flush=True);count+=1
    finally:sim.close()
if __name__=='__main__':main()

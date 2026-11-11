"""Native HSSD geometry validation and sequential physical mono receiver RIRs."""
import argparse,json,time,sys,subprocess,hashlib
from pathlib import Path
import numpy as np
import habitat_sim

def write(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def make(c,out,furnished=True,physics=False):
 r=Path(c['root'])/'source/hssd-hab';b=habitat_sim.SimulatorConfiguration();b.scene_id=str(r/('scenes' if furnished else 'stages')/(c['scene']+('.scene_instance.json' if furnished else '.stage_config.json')));b.scene_dataset_config_file=str(out/'furnished.scene_dataset_config.json');b.load_semantic_mesh=False;b.enable_physics=physics;b.gpu_device_id=-1;b.requires_textures=False
 a=habitat_sim.agent.AgentConfiguration();cam=habitat_sim.CameraSensorSpec();cam.uuid='depth';cam.resolution=[16,16];cam.sensor_type=habitat_sim.SensorType.DEPTH;a.sensor_specifications=[cam];return habitat_sim.Simulator(habitat_sim.Configuration(b,[a]))
def ray_distance(sim,p,d):
 ray=habitat_sim.geo.Ray();ray.origin=np.asarray(p,dtype=np.float32);ray.direction=np.asarray(d,dtype=np.float32);hits=sim.cast_ray(ray).hits;return min([float(x.ray_distance) for x in hits]+[1e9])
def clear(sim,p,margin):
 # Check inward-facing short rays from both sides, as well as outgoing rays.
 for d in np.vstack([np.eye(3),-np.eye(3)]):
  if ray_distance(sim,p,d)<margin:return False
  if ray_distance(sim,np.array(p)+margin*d,-d)<margin-.001:return False
 return True

def offsets(c,yaw):
 a=np.deg2rad(yaw+np.asarray(c['mic_clockwise_angles_deg']));return c['mic_radius_m']*np.stack([np.cos(a),np.zeros(6),np.sin(a)],1)
def audio(sim,c):
 sp=habitat_sim.AudioSensorSpec();sp.uuid='audio_sensor';sp.position=[0,0,0];sp.enableMaterials=False;sp.channelLayout.type=habitat_sim.sensor.RLRAudioPropagationChannelLayoutType.Mono;sp.channelLayout.channelCount=1;ac=sp.acousticsConfig
 for k,v in dict(sampleRate=48000,indirectRayCount=c['indirect_rays'],indirectRayDepth=c['ray_depth'],sourceRayCount=c['source_rays'],sourceRayDepth=c['ray_depth'],direct=True,indirect=True,diffraction=True,transmission=False,threadCount=c['acoustic_threads'],maxDiffractionOrder=10).items():setattr(ac,k,v)
 vals={}
 for k in dir(ac):
  if k.startswith('_'):continue
  v=getattr(ac,k)
  if isinstance(v,(int,float,bool,str)):vals[k]=v
 sim.add_sensor(sp);return sim.get_agent(0)._sensors['audio_sensor'],vals

def ring(sim,sensor,c,source,center,yaw):
 arr=[];positions=center+offsets(c,yaw)
 for pos in positions:
  agent=sim.get_agent(0);state=agent.get_state();state.position=pos.astype(np.float32);state.sensor_states={};agent.set_state(state,reset_sensors=True);sensor.setAudioSourceTransform(np.asarray(source,dtype=np.float32));v=np.asarray(sim.get_sensor_observations()['audio_sensor'])[0];arr.append(v.copy())
 n=max(map(len,arr));rir=np.zeros((6,n),np.float32)
 for k,v in enumerate(arr):rir[k,:len(v)]=v
 assert np.isfinite(rir).all() and np.all(np.max(abs(rir),axis=1)>0)
 return rir,positions

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--mode',choices=['geometry','render'],required=True);a=ap.parse_args();o=a.out;c=json.loads((o/'config.json').read_text());start=time.monotonic()
 if a.mode=='geometry':
  sim=make(c,o);nobj=sim.get_rigid_object_manager().get_num_objects();expected=len(json.loads((Path(c['root'])/'source/hssd-hab/scenes'/(c['scene']+'.scene_instance.json')).read_text())['object_instances']);assert nobj==expected
  bounds=sim.get_active_scene_graph().get_root_node().cumulative_bb;nav=habitat_sim.NavMeshSettings();nav.set_defaults();nav.agent_radius=.1;nav.agent_height=1.5;nav.cell_size=.05;nav_ok=sim.recompute_navmesh(sim.pathfinder,nav,include_static_objects=True);pools=[];total=0
  for room in json.loads((o/'source_proposals.json').read_text()):
   good=[]
   for x,z in room['points_xz']:
    p=[x,c['source_height_m'],z];floor=[x,.05,z];snap=np.array(sim.pathfinder.snap_point(floor)) if nav_ok else np.array([np.nan]*3)
    if not np.isfinite(snap).all() or np.linalg.norm(snap-np.array(floor))>.15:continue
    # Native cast_ray has no functional collision backend here; depth validation is mandatory.
    good.append(p)
    if len(good)>=c['source_generation_count_per_room']+c['source_integration_count_per_room']:break
   # Disjoint interleaved supports; small rooms retain their actual finite quadrature size.
   assert len(good)>=4,(room['label'],len(good));gen=good[::3];integ=[p for i,p in enumerate(good) if i%3];pools.append(dict(room=room['room'],label=room['label'],generation=gen,integration=integ));total+=len(good);print('room',room['room'],room['label'],len(good),flush=True)
  reps=[np.array(p['integration'][0]) for p in pools];conn=[]
  for i in range(len(reps)):
   for j in range(i+1,len(reps)):
    path=habitat_sim.ShortestPath();path.requested_start=sim.pathfinder.snap_point([reps[i][0],.05,reps[i][2]]);path.requested_end=sim.pathfinder.snap_point([reps[j][0],.05,reps[j][2]]);ok=sim.pathfinder.find_path(path);conn.append(dict(rooms=[i,j],connected=bool(ok),distance=float(path.geodesic_distance) if ok else None))
  # Three same-room pairs; acoustic visibility is NOT established by cast_ray.
  pairs=[]
  for label in ['bathroom','kitchen','living room']:
   room=next(r for r in pools if r['label']==label);found=None
   for src in room['generation']:
    for p in room['integration']:
     center=np.array(p);center[1]=c['listener_height_m'];delta=np.array(src)-center;d=np.linalg.norm(delta)
     if not 1<=d<=4:continue
     found=dict(room=room['room'],label=label,source=src,center=center.tolist(),distance=d);break
    if found:break
   assert found,label;pairs.append(found)
  env=dict(habitat_version=habitat_sim.__version__,habitat_path=habitat_sim.__file__,habitat_commit=Path('/opt/src/habitat-sim/.git/HEAD').read_text().strip(),soundspaces_commit=Path('/opt/src/soundspaces2/.git/HEAD').read_text().strip(),python=sys.version,numpy=np.__version__)
  write(o/'source_support.json',pools);write(o/'acoustic_pairs.json',pairs);write(o/'geometry_validation.json',dict(objects_loaded=nobj,objects_expected=expected,bounds=[list(bounds.min),list(bounds.max)],source_points=total,navmesh_built=bool(nav_ok),room_connectivity=conn,environment=env,seconds=time.monotonic()-start,placement='provisional polygon/free-map/navmesh points; actual depth clearance must follow; no object attachment',collision_limit='cast_ray backend nonfunctional; never treat it as free-space evidence'))
  sim.close()
 else:
  pairs=json.loads((o/'acoustic_pairs.json').read_text());folder=o/'rirs';folder.mkdir(exist_ok=True);records=[]
  for furnished in [True,False]:
   sim=make(c,o,furnished);sensor,settings=audio(sim,c)
   # stage-only matched comparison is only pair0/head0, three repeats; full furniture six configurations x3.
   for seed in c['audio_seeds']:
    sim.seed(seed)
    for i,p in enumerate(pairs if furnished else pairs[:1]):
     for yaw in c['rir_headings_deg'] if furnished else [0]:
      name=f'{"furnished" if furnished else "stage"}_p{i}_h{yaw}_s{seed}';dest=folder/(name+'.npy');assert not dest.exists(),'Use fresh output for new rendering'
      rir,pos=ring(sim,sensor,c,np.array(p['source']),np.array(p['center']),yaw);np.save(dest,rir);records.append(dict(name=name,path=str(dest),furnished=furnished,seed=seed,pair=i,heading=yaw,source=p['source'],center=p['center'],mic_positions=pos.tolist(),shape=list(rir.shape),sha256=hashlib.sha256(dest.read_bytes()).hexdigest()));print('render',name,rir.shape,flush=True)
   sim.close()
  write(o/'acoustic_render.json',dict(records=records,settings=settings,seconds=time.monotonic()-start,materials='disabled: default acoustic material, HSSD object material mapping not validated',seed_scope='sim.seed supplied; RLR random generator has no verified public seed control',channel_scope='6 sequential mono listener positions with unchanged source and emission origin; synchronization assessed by direct arrival timing; no Ambisonics',furniture_scope='loaded rigid object count and stage/furnished matched waveform intervention; investigate binding mesh upload separately'))
if __name__=='__main__':main()

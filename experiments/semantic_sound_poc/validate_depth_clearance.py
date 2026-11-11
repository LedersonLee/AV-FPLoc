"""Independent rendered-depth replacement for nonfunctional cast_ray checks."""
import argparse,json,time
from pathlib import Path
import numpy as np
import habitat_sim
from run_acoustic import make,write
ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--mode');a=ap.parse_args();o=a.out;c=json.loads((o/'config.json').read_text());support=json.loads((o/'source_support.json').read_text());sim=make(c,o);start=time.monotonic()
orientations=[[0,0,0],[0,np.pi,0],[0,-np.pi/2,0],[0,np.pi/2,0],[np.pi/2,0,0],[-np.pi/2,0,0]]
for i,ori in enumerate(orientations):
 sp=habitat_sim.CameraSensorSpec();sp.uuid='axis'+str(i);sp.resolution=[1,1];sp.position=[0,0,0];sp.orientation=ori;sp.sensor_type=habitat_sim.SensorType.DEPTH;sim.add_sensor(sp)
records=[]
for room in support:
 for group in ['generation','integration']:
  for idx,p in enumerate(room[group]):
   agent=sim.get_agent(0);state=agent.get_state();state.position=np.array(p,np.float32);state.sensor_states={};agent.set_state(state,reset_sensors=True);obs=sim.get_sensor_observations();depth=[float(obs['axis'+str(i)][0,0]) for i in range(6)];records.append(dict(room=room['room'],group=group,index=idx,depth_m=depth,pass_clearance=bool(min(depth)>=.15 and depth[-1]>0 and depth[-1]<3)))
 print('depth room',room['room'],flush=True)
result=dict(method='six 1x1 cardinal depth cameras at exact source xyz, native furnished render geometry; depth down/up sanity',count=len(records),invalid=sum(not r['pass_clearance'] for r in records),records=records,seconds=time.monotonic()-start,caveat='six directional rays and navmesh, not full volumetric source certification');write(o/'source_depth_clearance.json',result);print('invalid',result['invalid'],'seconds',result['seconds']);sim.close()

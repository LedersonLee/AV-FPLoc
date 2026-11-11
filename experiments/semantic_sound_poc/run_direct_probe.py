"""Discriminate missing direct path from array timing error; preserve primary renders."""
import argparse,json,time
from pathlib import Path
import numpy as np
from run_acoustic import make,audio,offsets,write

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--mode');a=ap.parse_args();o=a.out;c=json.loads((o/'config.json').read_text());pairs=json.loads((o/'acoustic_pairs.json').read_text());folder=o/'direct_probe';folder.mkdir(exist_ok=True);records=[];start=time.monotonic()
 for furnished in [True,False]:
  sim=make(c,o,furnished)
  # Explicit direct-only sensor settings before initialization.
  import habitat_sim
  sp=habitat_sim.AudioSensorSpec();sp.uuid='audio_sensor';sp.position=[0,0,0];sp.channelLayout.type=habitat_sim.sensor.RLRAudioPropagationChannelLayoutType.Mono;sp.channelLayout.channelCount=1;sp.enableMaterials=False;sp.acousticsConfig.sampleRate=48000;sp.acousticsConfig.direct=True;sp.acousticsConfig.indirect=False;sp.acousticsConfig.diffraction=False;sp.acousticsConfig.transmission=False;sp.acousticsConfig.temporalCoherence=False;sp.acousticsConfig.threadCount=2;sim.add_sensor(sp);sensor=sim.get_agent(0)._sensors['audio_sensor']
  for i,p in enumerate(pairs):
   for yaw in [0,90]:
    arrays=[]
    for pos in np.array(p['center'])+offsets(c,yaw):
     agent=sim.get_agent(0);state=agent.get_state();state.position=pos.astype(np.float32);state.sensor_states={};agent.set_state(state,reset_sensors=True);sensor.setAudioSourceTransform(np.array(p['source'],np.float32));arrays.append(np.asarray(sim.get_sensor_observations()['audio_sensor'])[0].copy())
    arr=np.zeros((6,max(map(len,arrays))),np.float32)
    for k,v in enumerate(arrays):arr[k,:len(v)]=v
    name=f'{"furnished" if furnished else "stage"}_p{i}_h{yaw}';dest=folder/(name+'.npy');assert not dest.exists();np.save(dest,arr);records.append(dict(name=name,nonzero_channels=np.any(arr!=0,axis=1).tolist(),peaks=np.max(abs(arr),axis=1).tolist()));print(records[-1],flush=True)
  sim.close()
 write(o/'direct_probe.json',dict(records=records,seconds=time.monotonic()-start,settings='direct=True; indirect=False; diffraction=False; transmission=False; temporalCoherence=False; sampleRate=48000'))
if __name__=='__main__':main()

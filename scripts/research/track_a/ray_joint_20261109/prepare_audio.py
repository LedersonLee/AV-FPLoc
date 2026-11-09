"""Matched100ms RIR/causal chirp inputs; only allowlisted non-Test scenes."""
from pathlib import Path
import json,hashlib,time
import numpy as np,torch
from scipy import signal,ndimage
from avfploc.acoustic.features import extract,DEFAULT_FEATURES
from avfploc.acoustic.model import AcousticEncoder
from avfploc.acoustic.contracts import recipe
P=Path('/output');W=Path('/workspace');OLD=W/'experiments/directional_tribranch_20261109';R=Path('/file2/jeongeon/AV-FPLoc/datasets/HSSD');PR=json.loads((P/'PROTOCOL.json').read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def envelope(raw):
 x=raw-raw.mean(0);x[:,:103]=0;x=np.pad(x,((0,0),(1024,1024)));x=signal.sosfiltfilt(signal.butter(3,[500,1500],btype='bandpass',fs=48000,output='sos'),x,axis=-1);v=ndimage.gaussian_filter1d(abs(signal.hilbert(x,axis=-1))[:,1127:5824],12,axis=-1);v=np.pad(v,((0,0),(0,7))).reshape(6,196,24).mean(-1);return (v/np.maximum(v.max(-1,keepdims=True),1e-15)).astype('float32')
def compressed(e,c,n):return np.r_[np.stack([e.reshape(6,-1)[:,i].mean(-1) for i in np.array_split(np.arange(e.size//6),8)],-1).ravel(),np.stack([n[:,i].mean(-1) for i in np.array_split(np.arange(n.shape[-1]),8)],-1).ravel(),c.ravel()]
def tokens(raw):
 f,t,z=signal.stft(raw[:,103:4800],fs=48000,nperseg=64,noverlap=48,nfft=512,boundary='zeros',padded=True,axis=-1);mag=abs(z);energy=mag**2;log=np.log1p(100*energy/max(energy.sum(),1e-30));cross=z*np.roll(z,-1,axis=0).conj();phase=cross/np.maximum(abs(cross),1e-30);v=np.concatenate([log,phase.real,phase.imag],0).astype('float32');return torch.nn.functional.adaptive_avg_pool2d(torch.from_numpy(v),(32,32)).numpy().astype('float16')
def main():
 torch.set_num_threads(2);cfg=json.loads((R/'ReliabilityFusion/expansions/train36_val8_test3_v1/config.json').read_text());cp=R/cfg['acoustic_checkpoint'];assert sha(cp)==cfg['acoustic_sha256'];ck=torch.load(cp,map_location='cpu',weights_only=False);model=AcousticEncoder(**ck['config']['model']).cuda().eval();model.load_state_dict(ck['model']);entries={e['scene_id']:e for e in json.loads((R/'ReliabilityFusion/expansions/train36_val8_test3_v1/bank_index.json').read_text())['scenes']};chirp=signal.chirp(np.arange(2400)/48000,f0=20,f1=20000,t1=.05,method='logarithmic',phi=0);chirp/=np.sqrt((chirp**2).sum());np.save(P/'chirp.npy',chirp);config={**DEFAULT_FEATURES,'end_ms':(4800-7)/48};receipts=[]
 for sid in PR['training_scenes']+PR['transfer_scenes']:
  dest=P/'audio'/sid;dest.mkdir(parents=True,exist_ok=True)
  if (dest/'receipt.json').exists():receipts.append(json.loads((dest/'receipt.json').read_text()));continue
  start=time.time();o=np.load(OLD/'data'/sid/'observations.npz');entry=entries[sid];assert entry['split']=='train';schedule=R/entry['observation_schedule_path'];assert sha(schedule)==entry['observation_schedule_sha256'];q={v['sample_index']:v for v in json.loads(schedule.read_text())['queries']};rows={k:[] for k in ['ecn','embedding','raw','chirp','permuted','isotropic','sorted']};sources=[];es=[];cs=[];ns=[]
  for qid in o['query_ids']:
   item=q[int(qid)];path=R/item['observed_rir_path'];raw=np.load(path).astype(float);assert raw.shape[0]==6 and raw.shape[1]>=4800;raw=np.roll(raw[:,:4800],item['channel_roll'],axis=0);assert item['channel_roll']==recipe(item['world_yaw_deg'])[1];e,c=extract(raw,config);n=envelope(raw);es.append(e);cs.append(c);ns.append(n);rows['ecn'].append(compressed(e,c,n));echo=signal.fftconvolve(raw,chirp[None],mode='full',axes=-1)[:,:4800];order=np.argsort((raw[:,103:]**2).sum(-1),kind='stable')
   for name,arr in [('raw',raw),('chirp',echo),('permuted',raw[[0,2,4,1,5,3]]),('isotropic',np.repeat(raw.mean(0,keepdims=True),6,axis=0)),('sorted',raw[order])]:rows[name].append(tokens(arr))
   sources.append(dict(query_id=int(qid),sha256=sha(path),path=str(path),roll=item['channel_roll']))
  with torch.inference_mode():
   for first in range(0,128,32):
    vals=[torch.tensor(np.array(v[first:first+32]),dtype=torch.float32,device='cuda') for v in [es,cs,ns]];rows['embedding'].extend(model(*vals).cpu().numpy())
  arrays={k:np.array(v) for k,v in rows.items()};assert all(np.isfinite(v).all() for v in arrays.values());np.savez_compressed(dest/'features.npz',**arrays,query_ids=o['query_ids']);receipt=dict(scene=sid,queries=128,shapes={k:list(v.shape) for k,v in arrays.items()},seconds=time.time()-start,feature_sha256=sha(dest/'features.npz'),checkpoint_sha256=sha(cp),chirp_sha256=sha(P/'chirp.npy'),source=sources,window_samples=[0,4800],feature_window=[103,4800],ecn_config=config,stft=dict(nfft=512,window=64,hop=16,channels='6 normalizedlogenergy,6 adjacentcosIPD,6 adjacentsinIPD',pool=[32,32]),note='Frozen checkpoint originally trained128ms; new100ms extraction is a duration shift. No downstream transfer scoring.',test_accessed=False);(dest/'receipt.json').write_text(json.dumps(receipt,indent=2));receipts.append(receipt);print('AUDIO',sid,round(time.time()-start,1),flush=True)
 (P/'audio_preparation.json').write_text(json.dumps(dict(records=receipts),indent=2))
if __name__=='__main__':main()

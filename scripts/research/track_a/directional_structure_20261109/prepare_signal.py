"""Raw waveform controls and synchronized E/C/N inputs; old artifacts are read-only."""
from pathlib import Path
import json,time,hashlib
import numpy as np
from scipy import signal
P=Path('/output');W=Path('/workspace');OLD=W/'experiments/directional_tribranch_20261109';H=W/'experiments/hssd_scaling_20261108_1332';R=Path('/file2/jeongeon/AV-FPLoc/datasets/HSSD')
from acoustic_branch.contracts import recipe,RING
PR=json.loads((P/'PROTOCOL.json').read_text());PERM=np.array([0,2,4,1,5,3])
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ecn(e,c,n):return np.r_[np.stack([e.reshape(6,-1)[:,ix].mean(-1) for ix in np.array_split(np.arange(e.size//6),8)],-1).ravel(),np.stack([n[:,ix].mean(-1) for ix in np.array_split(np.arange(n.shape[-1]),8)],-1).ravel(),c.ravel()].astype('float32')
def descriptor(raw):
 # Fixed common time origin, no per-channel peak alignment; scores are descriptors, not DoA.
 x=raw[:,:6151].astype(float);x=np.pad(x,((0,0),(0,max(0,6151-x.shape[1]))));sq=x*x;edges=7+np.round(np.array([0,2,4,8,16,32,64,128])*48).astype(int);mass=max(sq[:,103:].sum(),1e-30);energy=np.stack([sq[:,a:b].sum(-1) for a,b in zip(edges[:-1],edges[1:])],-1)/mass;cum=np.cumsum(sq[:,103:],axis=-1);total=np.maximum(cum[:,-1:],1e-30);quant=np.stack([np.argmax(cum>=q*total,axis=-1)/48 for q in [.1,.25,.5,.75,.9]],-1)/128;xx=x[:,103:2407];lags=[];corr=[]
 for i in range(6):
  a=xx[i]-xx[i].mean();b=xx[(i+1)%6]-xx[(i+1)%6].mean();cc=signal.correlate(a,b,mode='full',method='fft');mid=len(a)-1;cc=cc[mid-16:mid+17]/max(np.linalg.norm(a)*np.linalg.norm(b),1e-30);best=abs(cc).argmax();lags.append((best-16)/16);corr.append(cc[best])
 return np.r_[np.log1p(energy*100).ravel(),quant.ravel(),lags,corr].astype('float32')
def main():
 records=[];start=time.time();cfg=json.loads((R/'ReliabilityFusion/expansions/train36_val8_test3_v1/bank_index.json').read_text());entries={x['scene_id']:x for x in cfg['scenes']}
 for sid in PR['training_scenes']:
  dst=P/'signals'/sid;dst.mkdir(parents=True,exist_ok=True)
  if (dst/'receipt.json').exists():records.append(json.loads((dst/'receipt.json').read_text()));continue
  old=np.load(OLD/'data'/sid/'observations.npz');receipt=json.loads((OLD/'data'/sid/'receipt.json').read_text());rp=Path(receipt['raw_receipts']['Observed']['receipt']);assert sha(rp)==receipt['raw_receipts']['Observed']['sha256'];rec=json.loads(rp.read_text());aa=[np.load(rp.parent/(name+'.npy'),mmap_mode='r') for name in ['energy','coherence','envelope']];schedule=R/entries[sid]['observation_schedule_path'];assert sha(schedule)==entries[sid]['observation_schedule_sha256'];qs={q['sample_index']:q for q in json.loads(schedule.read_text())['queries']};lookup={s:i for i,s in enumerate(rec['paths'])};out={k:[] for k in ['raw','raw_single','raw_isotropic','raw_permuted','raw_rolled','ecn','ecn_rolls']};audits=[]
  for qid in old['query_ids']:
   q=qs[int(qid)];i=lookup[q['observed_rir_path']];path=R/q['observed_rir_path'];assert sha(path)==rec['raw_sha256'][i];raw=np.load(path);meta=json.loads(path.with_suffix('.json').read_text());assert raw.shape[0]==6 and np.isfinite(raw).all();assert meta['acoustics']['sample_rate_hz']==48000 and meta['acoustics']['ring_radius_m']==.05;np.testing.assert_allclose(meta['source_xyz_m'],meta['centre_xyz_m'],atol=1e-8);np.testing.assert_allclose(meta['microphone_angles_rad'],RING+np.deg2rad(meta['base_acoustic_yaw_deg']),atol=1e-8);roll=recipe(q['world_yaw_deg'])[1];assert roll==q['channel_roll'];raw=np.roll(raw,roll,axis=0)
   variants={'raw':raw,'raw_single':np.repeat(raw[:1],6,axis=0),'raw_isotropic':np.repeat(raw.mean(0,keepdims=True),6,axis=0),'raw_permuted':raw[PERM],'raw_rolled':np.roll(raw,1,axis=0)}
   for key,value in variants.items():out[key].append(descriptor(value))
   E=np.roll(aa[0][i],roll,0);C=np.roll(aa[1][i],roll,1);N=np.roll(aa[2][i],roll,0).astype('float16').astype('float32');out['ecn'].append(ecn(E,C,N));out['ecn_rolls'].append(np.stack([ecn(np.roll(E,r,0),np.roll(C,r,1),np.roll(N,r,0)) for r in range(6)]))
   peak=abs(raw[:,:96]).argmax(-1);audits.append(dict(query_id=int(qid),path=str(path),sha256=rec['raw_sha256'][i],metadata_sha256=sha(path.with_suffix('.json')),raw_shape=list(raw.shape),early_window_peak_samples=peak.tolist(),geometry_kind=meta['geometry_kind'],settings=meta['acoustics']))
  out={k:np.stack(v) for k,v in out.items()};np.testing.assert_array_equal(out['ecn'],old['features']);np.savez_compressed(dst/'features.npz',**out,query_ids=old['query_ids']);r=dict(scene=sid,queries=128,source_receipt_sha256=sha(rp),features_sha256=sha(dst/'features.npz'),raw_records=audits,test_accessed=False);(dst/'receipt.json').write_text(json.dumps(r,indent=2));records.append(r);print('SIGNALS',sid,flush=True)
 (P/'signal_audit.json').write_text(json.dumps(dict(records=records,seconds=time.time()-start,descriptor='84 dimensions:42 log normalized window energies,30 cumulative energy quantiles,6 adjacent lag/correlation pairs. Shared fixed clock; not physical echo labels.',test_accessed=False),indent=2))
if __name__=='__main__':main()

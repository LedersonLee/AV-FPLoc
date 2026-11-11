"""Independent invariants, source integrity, and numerical replay for Experiment1."""
import argparse,json,hashlib,ast
from pathlib import Path
import numpy as np
from geometry import paths,merge_segments,RING
from metrics import measure

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root;checks={}
 primary=list((root/'controlled').glob('*/receipt.json'));diag=list((root/'diagnostic').glob('*.json'));assert len(primary)==36 and len(diag)==60
 for f in primary+diag:
  r=json.loads(f.read_text());path=f.parent/'rir.npy' if f.name=='receipt.json' else f.with_suffix('.npy');x=np.load(path);assert x.shape[0]==6 and np.isfinite(x).all();assert hashlib.sha256(path.read_bytes()).hexdigest()==r.get('rir_sha256',r.get('sha256'))
 checks['new_rir_integrity']=96
 # Independent image-source solution for arbitrary walls from vector projections.
 rng=np.random.default_rng(71);tested=0
 for angle in np.linspace(0,2*np.pi,17)[:-1]:
  u=np.array([np.cos(angle),np.sin(angle)]);n=np.array([-u[1],u[0]]);s=n*2;wall=np.array([[-10*u,10*u]]);pp=paths([*s,37],wall);assert len(pp)==1
  mic=s+np.stack([.05*np.cos(RING-np.deg2rad(37)),-.05*np.sin(RING-np.deg2rad(37))],-1);image=-s;expected=np.linalg.norm(mic-image,axis=-1)/343*1000;np.testing.assert_allclose(pp[0]['times_ms'],expected,atol=1e-10)
  for q,r in zip(pp[0]['points'],mic):
   q=np.array(q);np.testing.assert_allclose(np.linalg.norm(s-q)+np.linalg.norm(r-q),np.linalg.norm(image-r),atol=1e-10)
  tested+=1
 assert not paths([20,2,0],np.array([[[-1,0],[1,0]]]))
 assert len(merge_segments(np.array([[[0,0],[1,0]],[[1,0],[2,0]],[[2,0],[1,0]]])))==1
 checks['rotated_specular_geometry_cases']=tested
 # Cyclic channel recipe independently maps measured ring to candidate-world yaw.
 for yaw in range(0,360,10):
  acoustic=(-yaw)%360;base=acoustic%60;roll=-round((acoustic-base)/60);raw=np.stack([.05*np.cos(RING+np.deg2rad(base)),-.05*np.sin(RING+np.deg2rad(base))],-1);rolled=np.roll(raw,roll,axis=0);pred=np.stack([.05*np.cos(RING-np.deg2rad(yaw)),-.05*np.sin(RING-np.deg2rad(yaw))],-1);np.testing.assert_allclose(rolled,pred,atol=1e-15)
 checks['yaw_convention_cases']=36
 oldroot=Path('/home/jeongeonlee/track1/experiments/directional_structure_20261109')
 for f in sorted((root/'hssd').glob('*.json')):
  sid=f.stem;cp=oldroot/'candidates'/sid/'pure_visual.npz';bank=np.load(oldroot/'candidates'/sid/'bank_geometry.npz');candidates=np.load(cp);idx=candidates['indices'];expected=np.concatenate([bank['xy'][idx//36],(idx%36*10)[...,None]],axis=-1);np.testing.assert_array_equal(candidates['pred'],expected)
  receipt=json.loads((cp.parent/'receipt.json').read_text());assert hashlib.sha256(cp.read_bytes()).hexdigest()==receipt['files']['pure_visual.npz']
 checks['pure_visual_pose_reconstruction_scenes']=12
 candidate_count=0;replays=0;donor_checks=0
 for f in sorted((root/'hssd').glob('*.json')):
  data=json.loads(f.read_text());q=data['queries'][0];raw=np.roll(np.load(q['raw_path']),q['roll'],axis=0);saved=json.loads((root/'hssd_metrics'/f.name).read_text());assert len(data['queries'])==32
  for q0 in data['queries']:
   assert not any(k in q0 for k in ['truth','gt','centre_xyz_m','world_yaw_deg']);assert len(q0['hypotheses'])==20;candidate_count+=20
  for seed in [42,43,44]:
   indices=np.roll(np.arange(32),int(np.random.default_rng(seed).integers(1,32)));assert sorted(indices.tolist())==list(range(32)) and not np.any(indices==np.arange(32));donor_checks+=1
  for h in q['hypotheses']:
   pp=[p for p in h['paths'] if p['supported'] and min(p['times_ms'])>=2 and max(p['times_ms'])<=79]
   if not pp:continue
   vals=[]
   for path in pp:
    centers=np.rint(np.array(path['times_ms'])*48).astype(int);values=np.stack([raw[i,j-24:j+25] for i,j in enumerate(centers)]).astype(float);manual=np.mean(values*values)/np.mean(raw[:,48:3841].astype(float)**2);vals.append(manual)
    np.testing.assert_allclose(manual,measure(raw,path['times_ms'])['power_ratio'],rtol=1e-13)
    delta=np.array(path['times_ms'])-np.mean(path['times_ms']);m=max(abs(delta))
    for center in [2+m,79-m]:assert min(center+delta)-1>=1-1e-12 and max(center+delta)+1<=80+1e-12
   record=next(x for x in saved['records'] if x['query']==0 and x['rank']==h['rank'] and x['supported'] and x['half_ms']==.5);np.testing.assert_allclose(record['arms']['aligned']['power_ratio'],np.mean(vals),rtol=1e-13);replays+=1
 checks.update(candidates=candidate_count,independent_power_replays=replays,exact_marginal_permutations=donor_checks)
 summary=json.loads((root/'hssd_summary.json').read_text());assert summary['scenes']==12 and summary['queries']==384 and not summary['test_accessed'];assert summary['counts']==dict(total=28902,supported=22295,unknown=4749,outside_analysis=1858)
 old=json.loads((root/'hssd_summary_v1_random_bounds.json').read_text())
 for x,y in zip(old['contrasts'],summary['contrasts']):
  if x['control']!='random':assert x==y
 checks['nonrandom_results_unchanged_after_support_fix']=True
 checks['viewer']=json.loads((root/'viewer_verification.json').read_text());assert checks['viewer']['candidate_states_rendered']==69
 (root/'verification.json').write_text(json.dumps(dict(passed=True,checks=checks,limitations='Independent scalar power replay uses first query of each scene; input SHA checks cover all384 in analyzer. No fresh independent scene/test inference claimed.'),indent=2));print(json.dumps(checks,indent=2))
if __name__=='__main__':main()

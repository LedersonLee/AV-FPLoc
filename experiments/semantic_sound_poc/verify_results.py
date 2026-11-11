"""Independent scalar likelihood, coordinate, metrics, and provenance sanity audit."""
import argparse,json,math,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from matplotlib.path import Path as Poly
from model import posterior
ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);o=ap.parse_args().out;c=json.loads((o/'config.json').read_text());s=json.loads((o/'source_support.json').read_text());v=np.load(o/'inference_candidates.npz');gt=np.load(o/'evaluation_truth.npz');res=np.load(o/'oracle_predictions.npz');obs=json.loads((o/'oracle_observations.json').read_text());n=len(v['poses']);r=Path(c['root']);w=Path(c['workspace']);sid=c['scene'];tests={}
# Verify old GT labels never changed, candidates independently recovered from bank indices.
old=np.load(w/'experiments/directional_structure_20261109/candidates'/sid/'pure_visual.npz');bank=np.load(w/'experiments/directional_structure_20261109/candidates'/sid/'bank_geometry.npz');idx=old['indices'][v['original_rows']];rebuilt=np.concatenate([bank['xy'][idx//36],(idx%36*10)[...,None]],axis=-1);np.testing.assert_allclose(rebuilt,v['poses']);tests['candidate_bank_reconstruction']=True
# Re-evaluate geometric success without importing evaluator code.
delta=v['poses'][:,:,:2]-gt['truth'][:,None,:2];pos=np.sqrt((delta**2).sum(-1));rot=abs((v['poses'][:,:,2]-gt['truth'][:,None,2]+180)%360-180);valid=(pos<=1)&(rot<=30+1e-8);np.testing.assert_array_equal(valid,res['valid']);tests['independent_success_labels']=True
for seedidx,seed in enumerate(c['seeds']):
 rng=np.random.default_rng(seed);events=list(c['event_probabilities']);events=rng.choice(events,size=n,p=list(c['event_probabilities'].values()))
 for q,event in enumerate(events):
  row=obs[seedidx*n+q];assert row['event']==event
  if event=='silence':continue
  rule=c['room_event_weights'][str(event)];weights=np.array([rule.get(x['label'],rule['default']) for x in s]);weights/=sum(weights);room=int(rng.choice(len(s),p=weights));source=s[room]['generation'][int(rng.integers(len(s[room]['generation'])))];assert row['source_xyz']==source and row['source_room']==room
 tests['source_schedule_independent_of_robot']=True
# Independent scalar likelihood on sampled observations, explicit coordinate loop.
maxerr=0
for ix in [0,10,19,50,100,150,250,350]:
 row=obs[ix];q=row['q'];si=c['seeds'].index(row['seed']);event=row['event'];score=v['scores'][q];prior=np.exp(score-score.max());prior/=sum(prior)
 if event=='silence':calc=prior
 else:
  rule=c['room_event_weights'][event];weights=np.array([rule.get(x['label'],rule['default']) for x in s]);weights/=sum(weights);like=[]
  for pose in v['poses'][q]:
   value=0
   for ww,room in zip(weights,s):
    vals=[]
    for x,y,z in room['integration']:
     theta=math.atan2(z-pose[1],x-pose[0])-math.radians(pose[2]);vals.append(math.exp((math.cos(theta-row['oracle_bearing_rad'])-1)/math.radians(15)**2))
    value+=ww*sum(vals)/len(vals)
   like.append(value)
  calc=prior*np.array(like);calc/=sum(calc)
 maxerr=max(maxerr,float(max(abs(calc-res['posterior'][si,3,q]))));np.testing.assert_allclose(calc,res['posterior'][si,3,q],atol=1e-12)
tests['scalar_likelihood_max_error']=maxerr
# Source pool validity and disjoint generation/inference supports.
regions=json.loads((r/'source/hssd-hab/semantics/scenes'/(sid+'.semantic_config.json')).read_text())['region_annotations'];count=0
for room,reg in zip(s,regions):
 assert not set(map(tuple,room['generation']))&set(map(tuple,room['integration']));poly=Poly(np.array(reg['poly_loop'])[:,[0,2]])
 for group in ['generation','integration']:
  points=np.array(room[group]);assert poly.contains_points(points[:,[0,2]]).all();assert np.all(points[:,1]==1.2);count+=len(points)
tests['disjoint_valid_source_support_points']=count
# Paired contrasts relative to B3 and position-cluster sensitivity.
correct=res['correct'].astype(float);rng=np.random.default_rng(20261111);boot=rng.integers(n,size=(10000,n));contrasts={}
for j in [2,4,5]:
 per=(correct[:,3]-correct[:,j]).mean(0);contrasts['B3-'+str(res['names'][j])]=dict(delta_pp=float(per.mean()*100),ci95_pp=np.percentile(per[boot].mean(1)*100,[2.5,97.5]).tolist())
xy,inv=np.unique(gt['truth'][:,:2],axis=0,return_inverse=True);difference=(correct[:,3]-correct[:,0]).mean(0);sums=np.bincount(inv,weights=difference);counts=np.bincount(inv);ids=rng.integers(len(xy),size=(10000,len(xy)));estimate=sums[ids].sum(1)/counts[ids].sum(1)*100
write=dict(tests=tests,contrasts=contrasts,position_cluster_sensitivity=dict(unique_positions=len(xy),delta_pp=float(difference.mean()*100),ci95_pp=np.percentile(estimate,[2.5,97.5]).tolist()),checks_passed=True)
(o/'verification.json').write_text(json.dumps(write,indent=2)+'\n');print(json.dumps(write,indent=2))

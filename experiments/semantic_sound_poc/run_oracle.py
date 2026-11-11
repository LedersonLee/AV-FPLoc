"""Paired oracle experiment, GT used only for observation generation/evaluation."""
import argparse,json,csv,time,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from model import posterior,softmax,bearing,wrap,room_weights

def write(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();o=a.out;c=json.loads((o/'config.json').read_text());support=json.loads((o/'source_support.json').read_text());d=np.load(o/'inference_candidates.npz');gt=np.load(o/'evaluation_truth.npz');np.testing.assert_array_equal(d['query_ids'],gt['query_ids']);poses=d['poses'];scores=d['scores'];truth=gt['truth'];n=len(truth);start=time.monotonic();labels=[r['label'] for r in support];inference_support=[dict(label=r['label'],integration=r['integration']) for r in support];names=['B0','B1','B2','B3','C1','C2','B3_noise15','B3_noise30','B3_noise60'];pout=np.empty((len(c['seeds']),len(names),n,20));records=[]
 meta=json.loads((Path(c['root'])/'2DFP'/c['scene']/'floorplan_metadata.json').read_text());free=np.asarray(Image.open(Path(c['root'])/'2DFP'/c['scene']/'map.png'))==255;origin=np.array([meta['world_bounds_m']['x_min'],meta['world_bounds_m']['z_min']]);spacing=meta['map_spacing_m']
 for si,seed in enumerate(c['seeds']):
  rng=np.random.default_rng(seed);events=list(c['event_probabilities']);events=rng.choice(events,size=n,p=[c['event_probabilities'][e] for e in events]);sources=[];roomids=[]
  # Entire activation/source schedule generated before accessing robot poses.
  for event in events:
   if event=='silence':sources.append([np.nan]*3);roomids.append(-1);continue
   room=int(rng.choice(len(support),p=room_weights(event,labels,c)));pool=support[room]['generation'];sources.append(pool[int(rng.integers(len(pool)))]);roomids.append(room)
  unit_noise=rng.normal(size=n);mismatch=rng.uniform(-np.pi,np.pi,size=n);shuffled=list(rng.permutation(labels))
  for q,event in enumerate(events):
   source=np.array(sources[q]);angle=0 if event=='silence' else float(bearing(truth[q:q+1],source[None])[0,0]);los=None
   if event!='silence':
    segment=truth[q,:2]+np.linspace(0,1,max(2,int(np.linalg.norm(source[[0,2]]-truth[q,:2])/.01)))[:,None]*(source[[0,2]]-truth[q,:2]);pix=np.rint((segment-origin)/spacing).astype(int);inside=(pix[:,0]>=0)&(pix[:,0]<free.shape[1])&(pix[:,1]>=0)&(pix[:,1]<free.shape[0]);los=bool(inside.all() and free[pix[:,1],pix[:,0]].all())
   for j,name in enumerate(names):
    observed=mismatch[q] if name=='C2' else angle
    if name.startswith('B3_noise'):observed=wrap(angle+np.deg2rad(int(name[8:]))*unit_noise[q])
    pout[si,j,q]=posterior(poses[q],scores[q],str(event),observed,inference_support,c,method=name,labels=shuffled if name=='C1' else None)
   records.append(dict(seed=seed,query_id=int(d['query_ids'][q]),q=q,event=str(event),source_room=roomids[q],source_xyz=source.tolist() if event!='silence' else None,oracle_bearing_rad=angle if event!='silence' else None,fp_line_clear=los,shuffled_labels=shuffled))
 np.testing.assert_allclose(pout.sum(-1),1,atol=1e-12);np.testing.assert_array_equal(pout[:,0],pout[:,1]);dist=np.linalg.norm(poses[:,:,:2]-truth[:,None,:2],axis=-1);yaw=np.abs(wrap(np.deg2rad(poses[:,:,2]-truth[:,None,2])));valid=(dist<=1)&(yaw<=np.pi/6+1e-10);best=pout.argmax(-1);correct=np.take_along_axis(np.broadcast_to(valid,(3,len(names),n,20)),best[...,None],axis=-1)[...,0];mass=(pout*valid[None,None]).sum(-1);entropy=-(pout*np.log(np.maximum(pout,1e-300))).sum(-1);covered=valid.any(-1)
 rng=np.random.default_rng(c['bootstrap_seed']);boot=rng.integers(n,size=(c['bootstrap_repetitions'],n));metrics=[]
 for j,name in enumerate(names):
  diff=correct[:,j].astype(float)-correct[:,0];byquery=diff.mean(0);ci=np.percentile(byquery[boot].mean(1)*100,[2.5,97.5]);selected_dist=np.take_along_axis(np.broadcast_to(dist,(3,n,20)),best[:,j,:,None],axis=-1)[...,0];m=dict(method=name,unique_queries=n,seed_trials=3*n,joint_top1_percent=float(correct[:,j].mean()*100),r1m_percent=float((selected_dist<=1).mean()*100),r05m_percent=float((selected_dist<=.5).mean()*100),fine01m_percent=float((selected_dist<=.1).mean()*100),covered_queries=int(covered.sum()),covered_subset_joint_percent=float(correct[:,j,covered].mean()*100),correct_pose_mass=float(mass[:,j].mean()),entropy_nats=float(entropy[:,j].mean()),covered_success_set_logloss=float(-np.log(np.maximum(mass[:,j,covered],1e-300)).mean()),improved=int((diff>0).sum()),degraded=int((diff<0).sum()),unchanged=int((diff==0).sum()),delta_pp=float(byquery.mean()*100),ci95_low_pp=float(ci[0]),ci95_high_pp=float(ci[1]));metrics.append(m)
 def summarize_mask(mask):
  return {name:dict(n=int(mask.sum()),joint_percent=float(correct[:,j][mask].mean()*100) if mask.any() else None) for j,name in enumerate(names)}
 events=np.array([x['event'] for x in records]).reshape(3,n);rooms=np.array([x['source_room'] for x in records]).reshape(3,n);line=np.array([x['fp_line_clear'] for x in records],object).reshape(3,n);breakdown={'event':{e:summarize_mask(events==e) for e in c['event_probabilities']},'source_room':{str(r):summarize_mask(rooms==r) for r in range(len(support))},'fp_line_clear':{str(v):summarize_mask(line==v) for v in [True,False]},'seed':{str(s):{name:float(correct[i,j].mean()*100) for j,name in enumerate(names)} for i,s in enumerate(c['seeds'])}}
 silent=events=='silence';assert np.max(np.abs(pout.transpose(0,2,1,3)[silent]-pout[:,0][silent,None,:]))==0
 # Local raw predictions, never Git artifacts.
 np.savez_compressed(o/'oracle_predictions.npz',posterior=pout,correct=correct,valid=valid,mass=mass,entropy=entropy,names=np.array(names),events=events,source_rooms=rooms)
 write(o/'oracle_observations.json',records);write(o/'metrics.json',dict(metrics=metrics,breakdown=breakdown,ci_unit='query cluster, mean over three seeds; within-scene only, not scene-generalization',activation_fraction=float((~silent).mean()),silence_trials=int(silent.sum()),silence_exact_prior=True,oracle_coverage_percent=float(covered.mean()*100),seconds=time.monotonic()-start,source_inputs_to_inference=False,semantic_only_constant_by_design=True,noise_kernel='fixed 15deg von Mises-equivalent concentration; additive wrapped Gaussian oracle noise 0/15/30/60, no re-tuning',NLL_scope='covered_success_set_logloss is diagnostic set-mass log loss on covered queries, not calibrated categorical NLL'))
 with (o/'metrics.csv').open('w') as f:wr=csv.DictWriter(f,fieldnames=list(metrics[0]),lineterminator="\n");wr.writeheader();wr.writerows(metrics)
 print(json.dumps(metrics,indent=2));print('time',time.monotonic()-start)
if __name__=='__main__':main()

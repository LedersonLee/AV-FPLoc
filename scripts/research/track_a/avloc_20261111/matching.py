"""Geometry-only global matching, Eq2, with separately called post-hoc diagnostics.
Pose IDs are XY-major, yaw-minor. No truth is accepted by matching or proposal NMS.
Normalized score mass assumes a uniform discrete pose prior; not proven calibrated.
"""
import numpy as np
from scipy.special import logsumexp

FOV=np.array([k%36 for k in range(-5,6)])

def laplace_field(world_rays,mu,scale,directions=None,xy_chunk=256):
 world=np.asarray(world_rays);mu=np.asarray(mu,dtype=np.float64);b=np.asarray(scale,dtype=np.float64)
 directions=np.arange(36) if directions is None else np.asarray(directions,dtype=int)
 assert world.ndim==2 and world.shape[1]==36 and mu.shape==b.shape==(len(directions),)
 assert np.isfinite(world).all() and np.isfinite(mu).all() and np.isfinite(b).all() and (b>0).all()
 assert len(np.unique(directions))==len(directions) and ((directions>=0)&(directions<36)).all()
 indices=(np.arange(36)[:,None]+directions[None])%36
 result=np.empty((len(world),36),dtype=np.float64)
 for first in range(0,len(world),xy_chunk):
  rays=world[first:first+xy_chunk,indices].astype(np.float64)
  result[first:first+len(rays)]=-(abs(rays-mu)/b+np.log(2*b)).sum(-1)
 return result.reshape(-1)

def pose_values(xy,ids):
 ids=np.asarray(ids,dtype=int);return np.column_stack([np.asarray(xy)[ids//36],(ids%36)*10.])

def circular_error(a,b):return abs((np.asarray(a)-b+180)%360-180)

def proposals(scores,xy,k=20,distance=1.,yaw=30.):
 scores=np.asarray(scores);assert scores.shape==(len(xy)*36,) and np.isfinite(scores).all()
 # Stable ties resolve by pose ID, independent of GT. Exclusion is the joint pose neighborhood.
 order=np.argsort(-scores,kind='stable');selected=[];selected_poses=[]
 for idx in order:
  pose=pose_values(xy,[idx])[0]
  if any(np.linalg.norm(pose[:2]-p[:2])<=distance and circular_error(pose[2],p[2])<=yaw for p in selected_poses):continue
  selected.append(int(idx));selected_poses.append(pose)
  if len(selected)==k:break
 return np.asarray(selected,dtype=int)

def diagnostic(scores,xy,truth,temperature=1.,candidate_ids=None):
 """GT enters here only after inference; never used to choose a candidate."""
 assert temperature>0 and np.isfinite(temperature)
 scores=np.asarray(scores,dtype=np.float64);xy=np.asarray(xy);truth=np.asarray(truth)
 assert scores.shape==(len(xy)*36,) and np.isfinite(scores).all()
 ids=proposals(scores,xy) if candidate_ids is None else np.asarray(candidate_ids,dtype=int)
 best=int(np.argmax(scores));pose=pose_values(xy,[best])[0];error=float(np.linalg.norm(pose[:2]-truth[:2]));yaw=float(circular_error(pose[2],truth[2]))
 poserr=np.linalg.norm(xy-truth[:2],axis=1);yawerr=circular_error(np.arange(36)*10,truth[2]);joint=(poserr[:,None]<=1)&(yawerr[None]<=30)
 # Nearest discrete pose is a grid approximation to GT, distinct from a tolerance region.
 nearest=int(np.argmin(poserr))*36+int(np.argmin(yawerr));scaled=scores/temperature;logp=scaled-logsumexp(scaled);p=np.exp(logp)
 cp=pose_values(xy,ids);ce=np.linalg.norm(cp[:,:2]-truth[:2],axis=1);cy=circular_error(cp[:,2],truth[2]);correct=(ce<=1)&(cy<=30)
 top_region=(np.linalg.norm(xy-pose[:2],axis=1)[:,None]<=1)&(circular_error(np.arange(36)*10,pose[2])[None]<=30)
 logmass=float(logsumexp(logp[joint.reshape(-1)])) if joint.any() else -np.inf
 raw_order=np.argsort(-scores,kind='stable')[:20];raw=pose_values(xy,raw_order);rawcorrect=(np.linalg.norm(raw[:,:2]-truth[:2],axis=1)<=1)&(circular_error(raw[:,2],truth[2])<=30)
 return dict(top1_id=best,position_error_m=error,yaw_error_deg=yaw,r1=error<=1,joint=error<=1 and yaw<=30,fine=error<=.1,
  top5_nms=bool(correct[:5].any()),top20_nms=bool(correct[:20].any()),top5_raw=bool(rawcorrect[:5].any()),top20_raw=bool(rawcorrect.any()),first_correct_nms_rank=int(np.flatnonzero(correct)[0]+1) if correct.any() else None,
  entropy_normalized=float(-(p*logp).sum()/np.log(len(scores))),gt_region_mass=float(np.exp(logmass)),gt_region_nll=float(-logmass),gt_region_exists=bool(joint.any()),nearest_grid_nll=float(-logp[nearest]),
  peak_confidence=float(p[best]),joint_confidence=float(p[top_region.reshape(-1)].sum()),raw_peak_margin=float(scores[best]-scores[raw_order[1]]),distinct_peak_margin=float(scores[ids[0]]-scores[ids[1]]) if len(ids)>1 else None)

def calibration(confidence,correct,bins=15):
 c=np.asarray(confidence,float);y=np.asarray(correct,float);assert c.shape==y.shape and len(c)>0 and ((c>=0)&(c<=1+1e-12)).all()
 assignments=np.minimum((c*bins).astype(int),bins-1);rows=[];ece=0.
 for i in range(bins):
  mask=assignments==i
  if not mask.any():continue
  mean=float(c[mask].mean());acc=float(y[mask].mean());count=int(mask.sum());ece+=count/len(c)*abs(mean-acc);rows.append(dict(bin=i,count=count,confidence=mean,accuracy=acc))
 return dict(ece=ece,brier=float(np.mean((c-y)**2)),bins=rows,queries=len(c))

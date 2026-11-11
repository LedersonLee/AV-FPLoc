"""Finite-segment image-source predictions; no waveform or true query pose input."""
import numpy as np
C=343.;FS=48000
RING=np.array([np.pi,4*np.pi/3,5*np.pi/3,2*np.pi,np.pi/3,2*np.pi/3])
def cross(a,b):return a[...,0]*b[...,1]-a[...,1]*b[...,0]
def merge_segments(segments):
 groups={}
 for a,b in np.asarray(segments):
  e=b-a;length=np.linalg.norm(e)
  if length<1e-5:continue
  u=e/length
  if u[0]<-1e-8 or (abs(u[0])<1e-8 and u[1]<0):u=-u
  n=np.array([-u[1],u[0]]);d=float(a@n);key=tuple(np.round(np.r_[u,d],4))
  groups.setdefault(key,[]).append(sorted([float(a@u),float(b@u)]))
 out=[]
 for key,intervals in groups.items():
  u=np.array(key[:2]);u/=np.linalg.norm(u);n=np.array([-u[1],u[0]]);d=key[2];intervals.sort();lo,hi=intervals[0]
  for l,h in intervals[1:]:
   if l<=hi+.001:hi=max(hi,h)
   else:out.append([u*lo+n*d,u*hi+n*d]);lo,hi=l,h
  out.append([u*lo+n*d,u*hi+n*d])
 return np.asarray(out).reshape(-1,2,2)
def clear_leg(a,b,segments):
 edge=segments[:,1]-segments[:,0];v=b-a;off=segments[:,0]-a;den=cross(v,edge);safe=np.where(abs(den)>1e-10,den,1)
 t=cross(off,edge)/safe;u=cross(off,v)/safe
 return not np.any((abs(den)>1e-10)&(t>1e-4)&(t<1-1e-4)&(u>=0)&(u<=1))
def paths(pose,segments,architecture=None):
 s=np.asarray(pose[:2]);theta=RING-np.deg2rad(pose[2]);r=s+np.stack([.05*np.cos(theta),-.05*np.sin(theta)],-1)
 a=segments[:,0];edge=segments[:,1]-a;length=np.linalg.norm(edge,axis=-1);u=edge/length[:,None];n=np.stack([-u[:,1],u[:,0]],-1);sd=np.sum((s-a)*n,-1);image=s-2*sd[:,None]*n
 v=r[None]-image[:,None];den=np.sum(v*n[:,None],-1);t=np.divide(np.sum((a-image)*n,-1)[:,None],den,out=np.zeros_like(den),where=abs(den)>1e-10)
 q=image[:,None]+t[...,None]*v;along=np.sum((q-a[:,None])*u[:,None],-1)
 valid=((t>0)&(t<1)&(along>=-1e-6)&(along<=length[:,None]+1e-6)&(sd[:,None]*np.sum((r[None]-a[:,None])*n[:,None],-1)>0)).all(-1)
 out=[]
 for j in np.flatnonzero(valid):
  if not all(clear_leg(s,q[j,k],segments) and clear_leg(r[k],q[j,k],segments) for k in range(6)):continue
  times=np.linalg.norm(v[j],axis=-1)/C*1000
  supported=True;gap=0.
  if architecture is not None:
   ae=architecture[:,1]-architecture[:,0];al=np.linalg.norm(ae,axis=-1);au=ae/np.maximum(al[:,None],1e-12);foot=q[j].mean(0);frac=np.clip(np.sum((foot-architecture[:,0])*ae,-1)/np.maximum(al*al,1e-12),0,1)
   dist=np.linalg.norm(foot-(architecture[:,0]+frac[:,None]*ae),axis=-1);parallel=abs(au@u[j])>=np.cos(np.deg2rad(10));gap=float(np.min(np.where(parallel,dist,np.inf)));supported=gap<=.20
  out.append(dict(wall=int(j),times_ms=times.tolist(),points=q[j].tolist(),distance_m=float(abs(sd[j])),bearing_deg=float(np.degrees(np.arctan2(*(q[j].mean(0)-s)[::-1]))),supported=bool(supported),architecture_gap_m=gap if np.isfinite(gap) else None))
 return out

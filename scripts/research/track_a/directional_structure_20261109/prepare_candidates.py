"""Pure visual proposals computed before GT labels and independently of audio."""
import json,time,hashlib
from pathlib import Path
import numpy as np,torch
from scipy.spatial import cKDTree
from avfploc.geometry.rays import angular_rays
from avfploc.evaluation.baselines import standardized
P=Path('/output');W=Path('/workspace');H=W/'experiments/hssd_scaling_20261108_1332';OLD=W/'experiments/directional_tribranch_20261109';PR=json.loads((P/'PROTOCOL.json').read_text());torch.set_num_threads(2)
def nms(field,neighbors):
 work=field.reshape(-1,36).copy();out=[]
 for _ in range(20):
  k=int(work.argmax());assert np.isfinite(work.ravel()[k]);out.append(k);c,y=divmod(k,36);yy=np.flatnonzero(abs((np.arange(36)*10-y*10+180)%360-180)<=30);work[np.ix_(neighbors[c],yy)]=-np.inf
 return np.array(out)
def labels(indices,xy,truth):
 pose=np.concatenate([xy[indices//36],(indices%36*10)[...,None]],-1);d=np.linalg.norm(pose[...,:2]-truth[:,None,:2],axis=-1);yaw=abs((pose[...,2]-truth[:,None,2]+180)%360-180);return pose,np.stack([d<=1,(d<=1)&(yaw<=30),d<=.1],-1)
def main(split="training_scenes"):
 recs=[]
 for sid in PR[split]:
  dest=P/'candidates'/sid;dest.mkdir(parents=True,exist_ok=True)
  if (dest/'receipt.json').exists():recs.append(json.loads((dest/'receipt.json').read_text()));continue
  t=time.time();o=np.load(OLD/'data'/sid/'observations.npz');folder=H/'inputs'/sid;win=np.load(folder/'windows.npy').reshape(-1,11);xy=np.load(folder/'xy.npy');fp=np.load(folder/'fp.npy');wg=torch.tensor(win,dtype=torch.float64,device='cuda');fg=torch.tensor(fp,dtype=torch.float64,device='cuda');neighbors=cKDTree(xy).query_ball_tree(cKDTree(xy),1+1e-7);rr,ss=map(np.stack,zip(*[angular_rays(m.astype(float),s.astype(float)) for m,s in zip(o['mu'],o['scale'])]));indices=[];vs=[];acs=[];vm=[];vstd=[];am=[];astd=[];baseline=[];fields=[]
  for first in range(0,128,16):
   r=torch.tensor(rr[first:first+16],device='cuda');s=torch.tensor(ss[first:first+16],device='cuda');v=(-abs(wg[None]-r[:,None])/s[:,None]).sum(-1)/40;a=torch.tensor(o['embedding'][first:first+16],dtype=torch.float64,device='cuda')@fg.T
   for i,(vv,aa) in enumerate(zip(v.cpu().numpy(),a.cpu().numpy())):
    idx=nms(vv,neighbors);indices.append(idx);vs.append(vv[idx]);acs.append(aa[idx]);vm.append(vv.mean());vstd.append(vv.std());am.append(aa.mean());astd.append(aa.std());baseline.append([vv.argmax(),aa.argmax(),(.85*standardized(vv)+.15*standardized(aa)).argmax()])
    if first+i in [0,32,64,96]:fields.append(dict(query=int(first+i),visual=vv.astype('float32'),cosine=aa.astype('float32')))
  idx=np.array(indices);base=np.array(baseline);z=np.load(OLD/'data'/sid/'candidates_seed42.npz');np.testing.assert_array_equal(base[:,0],z['baseline_indices'][:,0]);np.testing.assert_array_equal(base[:,1],z['baseline_indices'][:,1]);np.testing.assert_array_equal(base[:,2],z['baseline_indices'][:,2]);geo=np.load(Path('/ssd1/AV-FPLoc/datasets/HSSD/UnLoc/structural_targets')/sid/'desdf.npy',allow_pickle=True).item();ij=np.rint((xy-np.array([geo['origin_x_m'],geo['origin_z_min_m']]))/geo['desdf_spacing_m']).astype(int);depth=geo['desdf'][ij[:,1],ij[:,0]];g=np.take_along_axis(depth[idx//36],((idx%36)[...,None]+np.arange(36))%36,axis=-1)
  # GT accessed only after every proposal and scalar score was finalized.
  pred,y=labels(idx,xy,o['truth']);bp,by=labels(base,xy,o['truth']);np.savez_compressed(dest/'pure_visual.npz',indices=idx,geometry=g,pred=pred,labels=y,vs=np.array(vs),acs=np.array(acs),v_mean=vm,v_std=vstd,a_mean=am,a_std=astd,baseline_indices=base,baseline_labels=by,query_ids=o['query_ids']);np.savez_compressed(dest/'bank_geometry.npz',xy=xy,depth=depth);np.savez_compressed(dest/'example_fields.npz',query_indices=[x['query'] for x in fields],visual=np.stack([x['visual'] for x in fields]),cosine=np.stack([x['cosine'] for x in fields]));r=dict(scene=sid,bank_poses=len(fp),xy_cells=len(xy),queries=128,selection='Pure visual raw likelihood NMS20 at1m/30deg; no acoustic or GT feature; no local refinement',seconds=time.time()-t,reference_embedding_bytes=fp.nbytes,structural_geometry_bytes=depth.nbytes,files={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in dest.glob('*.npz')},test_accessed=False);(dest/'receipt.json').write_text(json.dumps(r,indent=2));recs.append(r);del wg,fg;torch.cuda.empty_cache();print('PUREV',sid,flush=True)
 (P/('candidate_preparation.json' if split=='training_scenes' else 'transfer_candidate_preparation.json')).write_text(json.dumps(dict(records=recs,test_accessed=False),indent=2))
if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--transfer',action='store_true');args=parser.parse_args();main('transfer_scenes' if args.transfer else 'training_scenes')

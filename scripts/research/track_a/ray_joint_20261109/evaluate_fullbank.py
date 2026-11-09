"""Fullbank localization separately from fixedcandidate reranking; no model fitting."""
from pathlib import Path
import json,time,sys
import numpy as np
from scipy.spatial import cKDTree
from oracle_capacity import represent
from evaluate_candidates import subset
sys.path.insert(0,'/workspace/experiments/directional_structure_20261109')
from prepare_candidates import nms
from avfploc.geometry.rays import angular_rays
from avfploc.evaluation.baselines import standardized
P=Path('/output');PREV=Path('/workspace/experiments/directional_structure_20261109');OLD=Path('/workspace/experiments/directional_tribranch_20261109');H=Path('/workspace/experiments/hssd_scaling_20261108_1332');PR=json.loads((P/'PROTOCOL.json').read_text())
def main():
 receipts=[(g,p,json.loads(p.read_text())) for g in ['acoustic','joint'] for p in sorted((P/g).glob('*/receipt.json'))];out=[]
 for fold,held in enumerate(PR['outer_folds']):
  for sid in held:
   o=np.load(OLD/'data'/sid/'observations.npz');bg=np.load(PREV/'candidates'/sid/'bank_geometry.npz');xy=bg['xy'];depth=bg['depth'];dense=depth[:,(np.arange(36)[:,None]+np.arange(36))%36].reshape(-1,36);banks={k:np.log1p(represent(dense,f'raw{k}')) for k in [4,8,12,36]};nb=cKDTree(xy).query_ball_tree(cKDTree(xy),1+1e-7);win=np.load(H/'inputs'/sid/'windows.npy').reshape(-1,11);visual=[];baseline=np.load(PREV/'fullbank'/sid/'predictions.npz')['visual_labels'][:,0,1];high=np.load(PREV/'discrimination'/f'pure_visual_seed42_fold{fold}'/f'{sid}.npz')['high_ambiguity']
   for mu,sc in zip(o['mu'],o['scale']):r,s=angular_rays(mu.astype(float),sc.astype(float));visual.append(standardized((-abs(win-r)/s).sum(-1)/40))
   for group,path,rec in receipts:
    if rec['fold']!=fold:continue
    root=P/'fullbank'/group/path.parent.name;root.mkdir(parents=True,exist_ok=True);target=root/(sid+'.json')
    if target.exists():out+=json.loads(target.read_text())['records'];continue
    start=time.time();pred=np.load(path.parent/f'{sid}.npz');np.testing.assert_array_equal(pred['query_ids'],o['query_ids']);k=rec.get('k',36);pp=pred['prediction'];rows=[];arrays={}
    for ek in ([k,4,8,12] if k==36 else [k]):
     p=pp if ek==k else subset(pp,ek);bank=banks[ek]
     for method in ['geometry','visual_geometry']:
      indices=[];labels=[]
      for i in range(128):
       score=-abs(bank-p[i]).mean(-1)
       if method=='visual_geometry':score=.85*visual[i]+.15*standardized(score)
       idx=nms(score,nb);d=np.linalg.norm(xy[idx//36]-o['truth'][i,:2],axis=-1);yaw=abs((idx%36*10-o['truth'][i,2]+180)%360-180);indices.append(idx);labels.append(np.stack([d<=1,(d<=1)&(yaw<=30),d<=.1],-1))
      lab=np.array(labels);j=lab[:,0,1];coverage=lab[:,:,1].any(1);rows.append(dict(group=group,kind=rec['kind'],seed=rec['seed'],fold=fold,trained_k=k,eval_k=ek,scene=sid,method=method,joint=float(j.mean()),fine=float(lab[:,0,2].mean()),R1=float(lab[:,0,0].mean()),top5=float(lab[:,:5,1].any(1).mean()),top20=float(coverage.mean()),conditional_joint=float(j[coverage].mean()) if coverage.any() else 0.,high_ambiguity_joint=float(j[high].mean()),repair=float((j&~baseline).mean()),harm=float((~j&baseline).mean()),preservation=float(j[baseline].mean())));arrays[f'k{ek}_{method}_indices']=np.array(indices);arrays[f'k{ek}_{method}_labels']=lab
    np.savez_compressed(root/(sid+'.npz'),**arrays,query_ids=o['query_ids']);target.write_text(json.dumps(dict(records=rows,seconds=time.time()-start),indent=2));out+=rows
   print('FULLBANK',sid,'records',len(out),flush=True)
 (P/'fullbank_evaluation.json').write_text(json.dumps(dict(records=out,scope='Same20cmXY36headinggrid; allmatchedmodels/Ksubsets, pureVreference, noGT scoring inputs; rawoutputs local.',test_accessed=False),indent=2))
if __name__=='__main__':main()

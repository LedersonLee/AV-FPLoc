"""GT-free candidate-geometry direction selection, calibrated fixed ray scales and controls."""
from pathlib import Path
import json,sys,time
import numpy as np
from scipy.spatial import cKDTree
from oracle_capacity import represent
sys.path.insert(0,'/workspace/experiments/directional_structure_20261109')
from discrimination import measures,standard
from prepare_candidates import nms
from avfploc.geometry.rays import angular_rays
from avfploc.evaluation.baselines import standardized
P=Path('/output');PREV=Path('/workspace/experiments/directional_structure_20261109');OLD=Path('/workspace/experiments/directional_tribranch_20261109');H=Path('/workspace/experiments/hssd_scaling_20261108_1332');PR=json.loads((P/'PROTOCOL.json').read_text())
def selection_weights(diff):
 ix=np.argsort(diff,axis=-1,kind='stable')[...,-9:];w=np.zeros_like(diff);np.put_along_axis(w,ix,1,axis=-1);w*=diff>1e-6;return w/np.maximum(w.sum(-1,keepdims=True),1)
def pair_score(g,p):
 w=selection_weights(abs(g[:,:,None]-g[:,None,:]));e=abs(g-p[:,None]);adv=(w*(e[:,None,:]-e[:,:,None])).sum(-1);return adv.mean(-1)
def main():
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--fullbank',action='store_true');args=ap.parse_args();records=[]
 for seed in PR['seeds']:
  for fold,held in enumerate(PR['outer_folds']):
   for sid in held:
    z=dict(np.load(PREV/'candidates'/sid/'pure_visual.npz'));mask=np.load(PREV/'discrimination'/f'pure_visual_seed{seed}_fold{fold}'/f'{sid}.npz');z['v']=mask['visual'];g=np.log1p(represent(z['geometry'],'raw36'));o=np.load(OLD/'data'/sid/'observations.npz')
    if args.fullbank:
     bg=np.load(PREV/'candidates'/sid/'bank_geometry.npz');xy=bg['xy'];dep=bg['depth'];dense=dep[:,(np.arange(36)[:,None]+np.arange(36))%36].reshape(-1,36);bank=np.log1p(represent(dense,'raw36'));nb=cKDTree(xy).query_ball_tree(cKDTree(xy),1+1e-7);win=np.load(H/'inputs'/sid/'windows.npy').reshape(-1,11);visual=[]
     for mu,scale in zip(o['mu'],o['scale']):r,s=angular_rays(mu.astype(float),scale.astype(float));visual.append(standardized((-abs(win-r)/s).sum(-1)/40))
    for group,kind in [('acoustic','raw'),('acoustic','chirp'),('acoustic','embedding'),('joint','attention')]:
     dest=P/('conditioned_fullbank' if args.fullbank else 'conditioned_candidates')/f'{group}_{kind}_seed{seed}_fold{fold}';dest.mkdir(parents=True,exist_ok=True);receipt=dest/f'{sid}.json'
     if receipt.exists():records+=json.loads(receipt.read_text())['records'];continue
     root=P/group/(f'{kind}_k36_seed{seed}_fold{fold}' if group=='acoustic' else f'{kind}_seed{seed}_fold{fold}');p=np.load(root/f'{sid}.npz')['prediction'];scales=np.load(P/'calibration'/f'{group}_{kind}_seed{seed}_fold{fold}'/'scales.npz');sigma=scales['sigma'];weights=1/sigma;weights/=weights.sum();rows=[];arrays={};start=time.time()
     for control,pp in [('matched',p),('donor17',np.roll(p,17,axis=0)),('shuffled',np.roll(p,1,axis=1))]:
      if not args.fullbank:
       scores=dict(uniform=-abs(g-pp[:,None]).mean(-1),uncertainty=-(abs(g-pp[:,None])*weights).sum(-1),pair_selection=pair_score(g,pp))
       for method,score in scores.items():
        for fusion in ['geometry','visual_geometry']:
         s=score if fusion=='geometry' else .85*z['v']+.15*standard(score);rows.append(dict(scene=sid,seed=seed,fold=fold,group=group,kind=kind,control=control,method=method,fusion=fusion,**measures(z,s,mask['ambiguous_mask'],mask['high_ambiguity'],z['v'].argmax(1))));arrays[control+'_'+method+'_'+fusion]=s
      else:
       choices={};labs={}
       for i in range(128):
        e=abs(bank-pp[i]);anchor=int(z['indices'][i,z['v'][i].argmax()]);w=selection_weights(abs(bank-bank[anchor]));scores=dict(uniform=-e.mean(-1),uncertainty=-(e*weights).sum(-1),anchor_selection=(w*(e[anchor]-e)).sum(-1))
        for method,score in scores.items():
         for fusion in ['geometry','visual_geometry']:
          s=score if fusion=='geometry' else .85*visual[i]+.15*standardized(score);idx=nms(s,nb);d=np.linalg.norm(xy[idx//36]-o['truth'][i,:2],axis=-1);yaw=abs((idx%36*10-o['truth'][i,2]+180)%360-180);key=method+'_'+fusion;choices.setdefault(key,[]).append(idx);labs.setdefault(key,[]).append(np.stack([d<=1,(d<=1)&(yaw<=30),d<=.1],-1))
       for key,lab in labs.items():
        lab=np.array(lab);method,fusion=next((m,f) for m in ['uniform','uncertainty','anchor_selection'] for f in ['geometry','visual_geometry'] if m+'_'+f==key);rows.append(dict(scene=sid,seed=seed,fold=fold,group=group,kind=kind,control=control,method=method,fusion=fusion,joint=float(lab[:,0,1].mean()),fine=float(lab[:,0,2].mean()),R1=float(lab[:,0,0].mean()),top5=float(lab[:,:5,1].any(1).mean()),top20=float(lab[:,:,1].any(1).mean())));arrays[control+'_'+key+'_indices']=np.array(choices[key]);arrays[control+'_'+key+'_labels']=lab
     np.savez_compressed(dest/f'{sid}.npz',**arrays,query_ids=o['query_ids']);receipt.write_text(json.dumps(dict(records=rows,seconds=time.time()-start),indent=2));records+=rows
    print('CONDITIONED',sid,seed,len(records),flush=True)
 (P/('conditioned_fullbank_results.json' if args.fullbank else 'conditioned_candidate_results.json')).write_text(json.dumps(dict(records=records,scope='PureVpairBorda vs fullbankVTop1anchor extension explicitlydifferent; nine directions selectedfromFPdifferencesonly;calibrationfrominnerOOF;notqueryspecificconfidence',test_accessed=False),indent=2))
if __name__=='__main__':main()

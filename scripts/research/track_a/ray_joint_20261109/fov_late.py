from pathlib import Path
import json,sys
import numpy as np
from oracle_capacity import represent
sys.path.insert(0,'/workspace/experiments/directional_structure_20261109')
from discrimination import standard,pairmask,measures,coarse
P=Path('/output');PREV=Path('/workspace/experiments/directional_structure_20261109');OLD=Path('/workspace/experiments/directional_tribranch_20261109');PR=json.loads((P/'PROTOCOL.json').read_text());FOV=np.r_[np.arange(6),np.arange(31,36)]
def load(s):
 z=dict(np.load(PREV/'candidates'/s/'pure_visual.npz'));z['v']=standard(z['vs']);z['geom']=np.log1p(coarse(z['geometry']));return z
rows=[];rayrows=[];fit=[load(s) for s in PR['training_scenes']];off=~np.eye(20,dtype=bool);cut=float(np.quantile(np.concatenate([abs(z['v'][:,:,None]-z['v'][:,None,:])[:,off].ravel() for z in fit]),1/3));hc=float(np.quantile(np.concatenate([np.sort(z['v'],axis=1)[:,-1]-np.sort(z['v'],axis=1)[:,-2] for z in fit]),1/3))
for split in ['development','transfer']:
 for seed in PR['seeds']:
  for fold,held in enumerate(PR['outer_folds'] if split=='development' else [PR['transfer_scenes']]):
   vroot=P/'joint'/f'vision_seed{seed}_fold{fold}' if split=='development' else P/'transfer_models'/f'joint_vision_seed{seed}';aroot=P/'joint'/f'acoustic_seed{seed}_fold{fold}' if split=='development' else P/'transfer_models'/f'joint_acoustic_seed{seed}'
   for sid in held:
    z=load(sid);v=np.load(vroot/f'{sid}.npz')['prediction'];a=np.load(aroot/f'{sid}.npz')['prediction'];p=a.copy();p[:,FOV]=.5*(a[:,FOV]+v[:,FOV]);g=np.log1p(represent(z['geometry'],'raw36'))
    if split=='development':m=np.load(PREV/'discrimination'/f'pure_visual_seed{seed}_fold{fold}'/f'{sid}.npz');mask=m['ambiguous_mask'];high=m['high_ambiguity']
    else:mask=pairmask(z,cut);sv=np.sort(z['v'],axis=1);high=sv[:,-1]-sv[:,-2]<=hc
    for kind,pred in [('fov_late',p),('late',.5*(a+v)),('acoustic',a),('vision',v)]:
     score=-abs(g-pred[:,None]).mean(-1);sc=.85*z['v']+.15*standard(score);rows.append(dict(scene=sid,seed=seed,split=split,kind=kind,**measures(z,sc,mask,high,z['v'].argmax(1))));y=represent(np.load(OLD/'data'/sid/'observations.npz')['target'],'raw36');err=abs(np.clip(np.expm1(pred),0,10)-y);outside=np.setdiff1d(np.arange(36),FOV);rayrows.append(dict(scene=sid,seed=seed,split=split,kind=kind,in_fov_mae=float(err[:,FOV].mean()),out_fov_mae=float(err[:,outside].mean())))
(P/'fov_late_results.json').write_text(json.dumps(dict(records=rows,ray_records=rayrows,FOV_indices=FOV.tolist(),scope='No newfit; diagnostic latefusionrestriction, not originaltransferfreeze;reuseddevelopment',test_accessed=False),indent=2));print('FOV DIAGNOSTIC',len(rows))

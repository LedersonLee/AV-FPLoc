"""Independent structural matching field; existing FP-RIR used only by cosine baselines."""
from pathlib import Path
import json,time
import numpy as np,torch
from scipy.spatial import cKDTree
from prepare_candidates import nms
from avfploc.geometry.rays import angular_rays
from avfploc.evaluation.baselines import standardized
P=Path('/output');W=Path('/workspace');OLD=W/'experiments/directional_tribranch_20261109';H=W/'experiments/hssd_scaling_20261108_1332';PR=json.loads((P/'PROTOCOL.json').read_text());torch.set_num_threads(2);CARD=[0,27,18,9]
def field_geometry(depth):
 sm=np.stack([np.median(depth[:,[(k+d)%36 for d in [-2,-1,0,1,2]]],axis=-1) for k in range(36)],axis=-1);return np.log1p(sm[:,(np.arange(36)[:,None]+np.array(CARD))%36]).reshape(-1,4)
def main():
 records=[];costs=[]
 for fold,held in enumerate(PR['outer_folds']):
  fit=[s for s in PR['training_scenes'] if s not in held];prior=np.concatenate([np.log1p(np.stack([np.median(np.load(OLD/'data'/s/'observations.npz')['target'][:,[(k+d)%36 for d in [-2,-1,0,1,2]]],axis=-1) for k in CARD],axis=-1)) for s in fit]).mean(0)
  for sid in held:
   dest=P/'fullbank'/sid;dest.mkdir(parents=True,exist_ok=True)
   if (dest/'complete.json').exists():j=json.loads((dest/'complete.json').read_text());records+=j['records'];costs.append(j['cost']);continue
   start=time.time();o=np.load(OLD/'data'/sid/'observations.npz');bg=np.load(P/'candidates'/sid/'bank_geometry.npz');xy=bg['xy'];g=field_geometry(bg['depth']);folder=H/'inputs'/sid;fp=np.load(folder/'fp.npy');win=np.load(folder/'windows.npy').reshape(-1,11);neighbors=cKDTree(xy).query_ball_tree(cKDTree(xy),1+1e-7);pred=np.load(P/'prediction'/f'seed42_fold{fold}'/f'legacy_ridge_{sid}.npz')['matched'];pred43=np.load(P/'prediction'/f'seed43_fold{fold}'/f'legacy_ridge_{sid}.npz')['matched'];np.testing.assert_array_equal(pred,pred43);u=np.load(P/'uncertainty'/f'seed42_fold{fold}'/f'{sid}.npz');sigma=u['uncertainty'];allidx={};alllab={};summary=[];examples={};structsec=0;cossec=0;vsec=0;gg=g-g.mean(-1,keepdims=True);gn=np.linalg.norm(gg,axis=-1).clip(1e-8);fg=torch.tensor(fp,dtype=torch.float64,device='cuda');ob=torch.tensor(o['embedding'],dtype=torch.float64,device='cuda');tt=time.perf_counter();cos=(ob@fg.T).cpu().numpy();cossec+=time.perf_counter()-tt
   for i in range(128):
    tt=time.perf_counter();r,s=angular_rays(o['mu'][i].astype(float),o['scale'][i].astype(float));v=-(abs(win-r)/s).sum(-1)/40;vsec+=time.perf_counter()-tt;tt=time.perf_counter();p=pred[i];res=abs(g-p);geom=-res.mean(-1);weighted=-(res/sigma[i]).mean(-1);ordinal=-((g<np.log(4))!=(p<np.log(4))).mean(-1);pc=p-p.mean();direction=(gg@pc)/(gn*max(np.linalg.norm(pc),1e-8));structsec+=time.perf_counter()-tt;zv=standardized(v);za=standardized(cos[i]);zg=standardized(geom)
    scores=dict(visual=v,cosine=cos[i],visual_cosine=.85*zv+.15*za,structure=geom,visual_structure=.85*zv+.15*zg,cosine_structure=.85*za+.15*zg,uncertainty_structure=weighted,ordinal_structure=ordinal,direction_structure=direction,geometry_prior=-abs(g-prior).mean(-1),donor_structure=-abs(g-pred[(i-17)%128]).mean(-1),shuffled_structure=-abs(g-p[[1,2,3,0]]).mean(-1))
    # Retrospective labels only after scores exist; no GT argument in matching/NMS.
    d=np.linalg.norm(xy-o['truth'][i,:2],axis=-1);dy=abs((np.arange(36)*10-o['truth'][i,2]+180)%360-180);success=((d[:,None]<=1)&(dy[None]<=30)).ravel()
    for arm,score in scores.items():
     ids=nms(score,neighbors);position=d[ids//36];yaw=dy[ids%36];labs=np.stack([position<=1,(position<=1)&(yaw<=30),position<=.1],-1);allidx.setdefault(arm,[]).append(ids);alllab.setdefault(arm,[]).append(labs);centered=score-score.max();mass=np.exp(np.clip(centered,-700,0));summary.append(dict(scene=sid,query=int(o['query_ids'][i]),arm=arm,joint=int(labs[0,1]),fine=int(labs[0,2]),R1=int(labs[0,0]),top5=bool(labs[:5,1].any()),top20=bool(labs[:,1].any()),gt_neighborhood_score_mass=float(mass[success].sum()/mass.sum()),mass_scope='fixed temperature1 descriptive normalized score concentration, not calibrated probability'))
    if i in [0,32,64,96]:
     for arm,score in scores.items():examples[f'{arm}_q{i}']=score.astype('float32')
   np.savez_compressed(dest/'predictions.npz',**{a+'_indices':np.array(v) for a,v in allidx.items()},**{a+'_labels':np.array(v) for a,v in alllab.items()},query_ids=o['query_ids']);np.savez_compressed(dest/'example_fields.npz',**examples,xy=xy,truth=o['truth']);cost=dict(scene=sid,queries=128,bank_poses=len(g),structural_score_seconds=structsec,cosine_matmul_seconds=cossec,visual_score_seconds=vsec,total_seconds=time.time()-start,fp_embedding_bytes=fp.nbytes,geometry4_log_bytes=g.nbytes,scope='CPU structural4variants timed together; GPU cosine matrix separately. Includes no rawrender/encoder extraction; full wall includes NMS/IO. FPbank needed by baseline only.');rec=dict(records=summary,cost=cost,test_accessed=False);(dest/'complete.json').write_text(json.dumps(rec,indent=2));records+=summary;costs.append(cost);del fg,ob;torch.cuda.empty_cache();print('FULLBANK',sid,round(cost['total_seconds'],2),flush=True)
 (P/'fullbank_results.json').write_text(json.dumps(dict(records=records,costs=costs,scope='One deterministic ridge predictor perouterfold; seed42/43 identical verified, not countedtwice. Structural inference hasnoFP-RIR input; frozen queryencoder historical training didusecontrastiveFP references. NoTrackBimplementation.',test_accessed=False),indent=2))
if __name__=='__main__':main()

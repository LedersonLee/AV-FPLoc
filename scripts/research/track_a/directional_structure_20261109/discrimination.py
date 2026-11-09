"""Same-candidate geometry matching on independent visual and originalB2 proposals."""
import json,itertools
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
P=Path('/output');W=Path('/workspace');OLD=W/'experiments/directional_tribranch_20261109';PR=json.loads((P/'PROTOCOL.json').read_text());CARD=[0,27,18,9]
def coarse(g):return np.stack([np.median(g[...,[(c+d)%36 for d in [-2,-1,0,1,2]]],-1) for c in CARD],-1)
def standard(g):return (g-g.mean(1,keepdims=True))/g.std(1,keepdims=True).clip(1e-6)
def load(s,seed,kind):
 path=P/'candidates'/s/'pure_visual.npz' if kind=='pure_visual' else OLD/'data'/s/f'candidates_seed{seed}.npz';z=dict(np.load(path));z['geom']=np.log1p(coarse(z['geometry']));z['v']=standard(z['vs']);z['a']=standard(z['acs']);return z
def pairmask(z,threshold):
 gap=abs(z['v'][:,:,None]-z['v'][:,None,:]);dist=np.linalg.norm(z['pred'][:,:,None,:2]-z['pred'][:,None,:,:2],axis=-1);gd=abs(z['geom'][:,:,None]-z['geom'][:,None,:]).mean(-1);return (gap<=threshold)&(dist>=1)&(gd>=.15)
def measures(z,g,mask,high,baseline):
 good=z['labels'][...,1];diff=g[:,:,None]-g[:,None,:];valid=good[:,:,None]&~good[:,None,:];pm=mask&valid;accuracy=(diff>0)+.5*(diff==0);choice=g.argmax(1);y=z['labels'][np.arange(128),choice];by=z['labels'][np.arange(128),baseline];coverage=good.any(1);ranks=np.argsort(np.argsort(-g,axis=1,kind='stable'),axis=1,kind='stable')+1;best=np.where(good,ranks,21).min(1);margins=[]
 for i in range(128):
  if valid[i].any():margins.append(float(diff[i][valid[i]].mean()))
 return dict(R1=float(y[:,0].mean()),joint=float(y[:,1].mean()),fine=float(y[:,2].mean()),pair_accuracy=float(accuracy[pm].mean()) if pm.any() else None,ambiguous_pairs=int(pm.sum()),ranking_auroc=float(np.mean([accuracy[i][valid[i]].mean() for i in range(128) if valid[i].any()])),GT_success_rank=float(best[coverage].mean()),success_minus_failure_margin=float(np.mean(margins)),high_ambiguity_joint=float(y[high,1].mean()) if high.any() else None,high_ambiguity_queries=int(high.sum()),coverage=float(coverage.mean()),conditional_joint=float(y[coverage,1].mean()),repair=float((y[:,1]&~by[:,1]).mean()),harm=float((~y[:,1]&by[:,1]).mean()),visual_success_preservation=float(y[by[:,1],1].mean()) if by[:,1].any() else None)
def main():
 records=[];settings=[]
 for seed in PR['seeds']:
  for fold,held in enumerate(PR['outer_folds']):
   ids=[s for s in PR['training_scenes'] if s not in held];predroot=P/'prediction'/f'seed{seed}_fold{fold}';trainy=np.concatenate([np.log1p(coarse(np.load(OLD/'data'/s/'observations.npz')['target'])) for s in ids]);prior=trainy.mean(0)
   for kind in ['pure_visual','B2']:
    ztrain=[load(s,seed,kind) for s in ids];off=~np.eye(20,dtype=bool);gaps=np.concatenate([abs(z['v'][:,:,None]-z['v'][:,None,:])[:,off].ravel() for z in ztrain]);threshold=float(np.quantile(gaps,1/3));topgaps=np.concatenate([np.sort(z['v'],axis=1)[:,-1]-np.sort(z['v'],axis=1)[:,-2] for z in ztrain]);highcut=float(np.quantile(topgaps,1/3));settings.append(dict(seed=seed,fold=fold,kind=kind,fit_scenes=ids,held_scenes=held,pair_gap=threshold,high_ambiguity_topgap=highcut))
    dst=P/'discrimination'/f'{kind}_seed{seed}_fold{fold}';dst.mkdir(parents=True,exist_ok=True)
    for sid in held:
     z=load(sid,seed,kind);mask=pairmask(z,threshold);sv=np.sort(z['v'],axis=1);high=sv[:,-1]-sv[:,-2]<=highcut;base=z['v'].argmax(1);scores=dict(visual=z['v'],cosine=z['a'],visual_cosine=.85*z['v']+.15*z['a'],geometry_prior=-abs(z['geom']-prior).mean(-1),query_only=np.zeros((128,20)))
     for model in ['legacy_ridge','legacy_embedding_ridge','raw_multi','cyclic_ecn']:
      pp=np.load(predroot/f'{model}_{sid}.npz')['matched'];gg=-abs(z['geom']-pp[:,None]).mean(-1);scores[model+'_geometry']=gg;scores[model+'_visual_geometry']=.85*z['v']+.15*standard(gg)
     pp=np.load(predroot/f'legacy_ridge_{sid}.npz')['matched']
     for lag in [1,17,31]:scores[f'donor{lag}_geometry']=-abs(z['geom']-np.roll(pp,lag,axis=0)[:,None]).mean(-1)
     scores['direction_shuffled_geometry']=-abs(z['geom']-pp[:,[1,2,3,0]][:,None]).mean(-1);scores['isotropic_geometry']=-abs(z['geom']-pp.mean(-1)[:,None,None]).mean(-1);scores['ordinal_geometry']=-( (z['geom']<np.log(4))!=(pp[:,None]<np.log(4)) ).mean(-1);scores['visual_ordinal']=.85*z['v']+.15*standard(scores['ordinal_geometry'])
     if kind=='B2':
      g=z['geoq_gain'].copy();g[:,1:]-=0 if seed==42 else .01;scores['GEOQ']=g
     for arm,g in scores.items():records.append(dict(seed=seed,fold=fold,scene=sid,kind=kind,arm=arm,**measures(z,g,mask,high,base)))
     np.savez_compressed(dst/f'{sid}.npz',**scores,ambiguous_mask=mask,high_ambiguity=high,query_ids=z['query_ids'])
   print('DISCRIMINATION',seed,fold,flush=True)
 (P/'discrimination_results.json').write_text(json.dumps(dict(records=records,settings=settings,scope='Pair selection uses V/FP only; GT determines retrospective success. Scene-held fixed settings. Scalar fusion here uses within-candidate zscores, distinct from fullbank baseline. Ties receive .5 in pairAUC, deterministic firstslot inTop1.',test_accessed=False),indent=2))
if __name__=='__main__':main()

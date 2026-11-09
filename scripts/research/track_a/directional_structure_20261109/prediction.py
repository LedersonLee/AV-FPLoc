"""Scene-held geometry probes: extend old work without retraining old models."""
from pathlib import Path
import json,time,hashlib,itertools,subprocess
import numpy as np,joblib
from sklearn.linear_model import Ridge
from sklearn.metrics import balanced_accuracy_score,roc_auc_score
from scipy.stats import spearmanr
P=Path('/output');W=Path('/workspace');OLD=W/'experiments/directional_tribranch_20261109';PR=json.loads((P/'PROTOCOL.json').read_text());CARD=[0,27,18,9]
def smooth(t):return np.stack([np.median(t[...,[(c+d)%36 for d in [-2,-1,0,1,2]]],axis=-1) for c in range(36)],axis=-1)
def target(o):return np.log1p(smooth(o['target'])[...,CARD])
def metrics(p,y):
 d=np.clip(np.expm1(p),0,10);gt=np.expm1(y);err=abs(d-gt);near=gt<3;score=np.log(4)-p;pn=score>0;acc=[];auc=[];rho=[]
 for j in range(4):
  acc.append(float(balanced_accuracy_score(near[:,j],pn[:,j])));auc.append(float(roc_auc_score(near[:,j],score[:,j])) if len(np.unique(near[:,j]))==2 else None)
 for a,b in zip(p,y):
  rho.append(float(spearmanr(a,b).statistic) if np.std(a)>1e-8 and np.std(b)>1e-8 else 0.)
 ordering=[]
 for a,b in itertools.combinations(range(4),2):
  keep=abs(y[:,a]-y[:,b])>.1;diff=(p[:,a]-p[:,b])[keep];truth=(y[:,a]-y[:,b])[keep];ordering.extend(((diff*truth>0)+.5*(diff==0)).tolist())
 return dict(mae_m=float(err.mean()),median_m=float(np.median(err)),p90_m=float(np.quantile(err,.9)),direction_mae_m=err.mean(0).tolist(),direction_median_m=np.median(err,axis=0).tolist(),direction_p90_m=np.quantile(err,.9,axis=0).tolist(),balanced_accuracy=float(np.mean(acc)),direction_balanced_accuracy=acc,auroc=float(np.mean([a for a in auc if a is not None])),direction_auroc=auc,rank_correlation=float(np.mean(rho)),pair_order_accuracy=float(np.mean(ordering)),openness_accuracy=float((near==pn).mean()),ordinal_accuracy=float((np.digitize(d,[2,5])==np.digitize(gt,[2,5])).mean()))
def fit(x,y):
 mean=x.mean(0);std=x.std(0).clip(.01);xx=np.clip((x-mean)/std,-10,10);model=Ridge(alpha=100).fit(xx,y);return dict(model=model,mean=mean,std=std)
def predict(m,x):return m['model'].predict(np.clip((x-m['mean'])/m['std'],-10,10))
def main():
 obs={s:dict(np.load(OLD/'data'/s/'observations.npz')) for s in PR['training_scenes']};sig={s:dict(np.load(P/'signals'/s/'features.npz')) for s in obs};records=[];fitrec=[]
 for seed in PR['seeds']:
  for fold,held in enumerate(PR['outer_folds']):
   ids=[s for s in obs if s not in held];dst=P/'prediction'/f'seed{seed}_fold{fold}';dst.mkdir(parents=True,exist_ok=True);oldroot=OLD/'stage_a'/f'seed{seed}_fold{fold}'
   for kind in ['raw_ridge','embedding_ridge','ridge','mlp']:
    for sid in held:
     old=np.load(oldroot/f'{kind}_{sid}.npz');y=target(obs[sid]);np.savez_compressed(dst/f'legacy_{kind}_{sid}.npz',**dict(old),query_ids=obs[sid]['query_ids'])
     for condition in old.files:
      yy=target(dict(target=np.roll(obs[sid]['target'],-6,axis=-1))) if condition=='rolled' else y;records.append(dict(seed=seed,fold=fold,scene=sid,model='legacy_'+kind,condition=condition,**metrics(old[condition],yy)))
   prior=np.concatenate([target(obs[s]) for s in ids]).mean(0)
   for arm in ['raw_multi','raw_single','raw_isotropic','cyclic_ecn']:
    key={'raw_multi':'raw','raw_single':'raw_single','raw_isotropic':'raw_isotropic'}.get(arm)
    if key:
     x=np.concatenate([sig[s][key] for s in ids]);y=np.concatenate([target(obs[s]) for s in ids])
    else:
     x=np.concatenate([sig[s]['ecn_rolls'].reshape(-1,552) for s in ids]);y=np.concatenate([np.stack([np.roll(np.log1p(smooth(obs[s]['target'])),-6*r,axis=-1) for r in range(6)],axis=1).reshape(-1,36) for s in ids])
    path=dst/f'{arm}.joblib';t=time.time()
    if path.exists():m=joblib.load(path)
    else:m=fit(x,y);joblib.dump(m,path)
    fitrec.append(dict(arm=arm,seed=seed,fold=fold,fit_scenes=ids,held_scenes=held,fit_rows=len(x),seconds=time.time()-t,checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),code_commit=subprocess.check_output(['git','-C',str(W/'AV-FPLoc'),'rev-parse','HEAD'],text=True).strip()))
    for sid in held:
     base=sig[sid];xx=base[key] if key else base['ecn'];pp=predict(m,xx);pp=pp if key else pp[:,CARD];preds=dict(matched=pp,prior=np.broadcast_to(prior,pp.shape).copy(),isotropic_prediction=np.repeat(pp.mean(1,keepdims=True),4,axis=1))
     for lag in [1,17,31]:preds[f'donor{lag}']=np.roll(pp,lag,axis=0)
     if arm=='raw_multi':preds['permuted']=predict(m,base['raw_permuted']);preds['rolled']=predict(m,base['raw_rolled'])
     if arm=='cyclic_ecn':preds['rolled']=predict(m,base['ecn_rolls'][:,1])[:,CARD]
     np.savez_compressed(dst/f'{arm}_{sid}.npz',**preds,query_ids=obs[sid]['query_ids'])
     for condition,pp in preds.items():
      yy=target(dict(target=np.roll(obs[sid]['target'],-6,axis=-1))) if condition=='rolled' else target(obs[sid]);records.append(dict(seed=seed,fold=fold,scene=sid,model=arm,condition=condition,**metrics(pp,yy)))
   print('PREDICTION',seed,fold,flush=True)
 (P/'prediction_results.json').write_text(json.dumps(dict(records=records,fits=fitrec,scope='Fixedsceneheld;legacy raw_ridge meansECN compressed features, newraw_multi meansactual waveformdescriptor;rolled condition scoredagainst rotatedtarget. NoTest.',test_accessed=False),indent=2))
if __name__=='__main__':main()

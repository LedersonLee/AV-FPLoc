import json,hashlib,importlib.util,joblib,time,subprocess
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from discrimination import coarse,measures,load,pairmask
P=Path('/output');W=Path('/workspace');OLD=W/'experiments/directional_tribranch_20261109';PR=json.loads((P/'PROTOCOL.json').read_text());spec=importlib.util.spec_from_file_location('geo',W/'experiments/structural_evidence_20261107_1730/inference.py');geo=importlib.util.module_from_spec(spec);spec.loader.exec_module(geo)
def features(z,p,prior):
 b=np.concatenate([z['S'],geo.geometry_summary(z['V'][...,:11]),geo.geometry_summary(z['V'][...,11:22])],-1);base=np.concatenate([b,b-b[:,:1]],-1);g=z['geom'];r=abs(g-p[:,None]);score=-r.mean(-1);prior_score=-abs(g-prior).mean(-1);gc=g-g.mean(-1,keepdims=True);pc=p-p.mean(-1,keepdims=True);direction=(gc*pc[:,None]).sum(-1)/(np.linalg.norm(gc,axis=-1)*np.linalg.norm(pc,axis=-1)[:,None]).clip(1e-8);ordinal=-((g<np.log(4))!=(p[:,None]<np.log(4))).mean(-1);extra=np.stack([score,ordinal,direction,prior_score,score-prior_score,r.max(-1),r.std(-1),np.broadcast_to(p.std(-1)[:,None],score.shape)],-1);return np.concatenate([base,extra,extra-extra[:,:1]],-1).astype('float32')
def main():
 settings=json.loads((P/'discrimination_results.json').read_text())['settings'];records=[];receipts=[]
 for seed in PR['seeds']:
  for fold,held in enumerate(PR['outer_folds']):
   ids=[s for s in PR['training_scenes'] if s not in held];zs={s:load(s,seed,'B2') for s in ids+held};pred={s:np.load(P/'uncertainty'/f'seed{seed}_fold{fold}'/f'{s}.npz')['prediction'] for s in ids+held};prior=np.concatenate([np.log1p(coarse(np.load(OLD/'data'/s/'observations.npz')['target'])) for s in ids]).mean(0);conf=next(c for c in settings if c['seed']==seed and c['fold']==fold and c['kind']=='B2');dst=P/'integration'/f'seed{seed}_fold{fold}';dst.mkdir(parents=True,exist_ok=True)
   for arm in ['base','matched','donor','nuisance']:
    def feat(s):
     x=features(zs[s],np.roll(pred[s],17,axis=0) if arm=='donor' else pred[s],prior)
     if arm=='base':x[...,156:]=0
     if arm=='nuisance':x[...,156:]=np.random.default_rng(int(hashlib.sha256(s.encode()).hexdigest()[:8],16)+seed).normal(size=x[...,156:].shape)
     return x
    X=np.concatenate([feat(s).reshape(-1,172) for s in ids]);Y=np.concatenate([zs[s]['labels'][...,1]+.5*zs[s]['labels'][...,2] for s in ids]);Y-=Y[:,:1].copy();path=dst/f'{arm}.joblib';t=time.time()
    if path.exists():m=joblib.load(path)
    else:m=HistGradientBoostingRegressor(max_iter=150,max_leaf_nodes=15,min_samples_leaf=100,l2_regularization=10,learning_rate=.07,early_stopping=False,random_state=seed).fit(X,Y.ravel());joblib.dump(m,path)
    receipts.append(dict(seed=seed,fold=fold,arm=arm,fit_scenes=ids,held_scenes=held,seconds=time.time()-t,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),code_commit=subprocess.check_output(['git','-C',str(W/'AV-FPLoc'),'rev-parse','HEAD'],text=True).strip()))
    for sid in held:
     z=zs[sid];g=m.predict(feat(sid).reshape(-1,172)).reshape(128,20);g[:,0]=0;mask=pairmask(z,conf['pair_gap']);v=np.sort(z['v'],axis=1);high=v[:,-1]-v[:,-2]<=conf['high_ambiguity_topgap'];records.append(dict(seed=seed,fold=fold,scene=sid,arm=arm,**measures(z,g,mask,high,z['v'].argmax(1))));np.savez_compressed(dst/f'{arm}_{sid}.npz',gain=g,choice=g.argmax(1),query_ids=z['query_ids'])
   for sid in held:
    z=zs[sid];g=z['geoq_gain'].copy();g[:,1:]-=0 if seed==42 else .01;score=-abs(z['geom']-pred[sid][:,None]).mean(-1);g+=.05*(score-score[:,:1]);mask=pairmask(z,conf['pair_gap']);v=np.sort(z['v'],axis=1);high=v[:,-1]-v[:,-2]<=conf['high_ambiguity_topgap'];records.append(dict(seed=seed,fold=fold,scene=sid,arm='fixed_addition',**measures(z,g,mask,high,z['v'].argmax(1))));np.savez_compressed(dst/f'fixed_addition_{sid}.npz',gain=g,choice=g.argmax(1),query_ids=z['query_ids'])
   print('INTEGRATION',seed,fold,flush=True)
 (P/'integration_results.json').write_text(json.dumps(dict(records=records,fits=receipts,scope='Crossfit structural predictions; same172dim treebudget;strongoriginalGEOQ reference separate. No held hyperparameterselection.',test_accessed=False),indent=2))
if __name__=='__main__':main()

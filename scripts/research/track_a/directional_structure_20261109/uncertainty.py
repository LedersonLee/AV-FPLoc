"""Fit-only residual uncertainty from existing inner scene-held acoustic probes."""
import json,importlib.util,joblib,hashlib,subprocess
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
P=Path('/output');W=Path('/workspace');OLD=W/'experiments/directional_tribranch_20261109';PR=json.loads((P/'PROTOCOL.json').read_text());spec=importlib.util.spec_from_file_location('prior',OLD/'stage_a.py');prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
def main():
 records=[]
 for seed in PR['seeds']:
  for fold,held in enumerate(PR['outer_folds']):
   ids=[s for s in PR['training_scenes'] if s not in held];obs={s:dict(np.load(OLD/'data'/s/'observations.npz')) for s in ids+held};pred={};errors=[];xs=[]
   for i in range(2):
    m=joblib.load(OLD/'stage_a'/f'seed{seed}_fold{fold}'/f'inner{i}_ridge.joblib')
    for sid in ids[i::2]:
     pp=prior.predict(m,obs[sid]);pred[sid]=pp;errors.append(abs(np.log1p(prior.coarse(obs[sid]['target']))-pp));xs.append(obs[sid]['embedding'])
   x=np.concatenate(xs);err=np.concatenate(errors);mean=x.mean(0);std=x.std(0).clip(.01);model=Ridge(alpha=100).fit((x-mean)/std,np.log(err+.05));fixed=np.quantile(err,.9,axis=0);dest=P/'uncertainty'/f'seed{seed}_fold{fold}';dest.mkdir(parents=True,exist_ok=True);joblib.dump(dict(model=model,mean=mean,std=std,fixed90=fixed,fit_scenes=ids),dest/'model.joblib')
   for sid in ids+held:
    pp=pred[sid] if sid in ids else np.load(P/'prediction'/f'seed{seed}_fold{fold}'/f'legacy_ridge_{sid}.npz')['matched'];sigma=np.exp(model.predict((obs[sid]['embedding']-mean)/std)).clip(.05,2);y=np.log1p(prior.coarse(obs[sid]['target']));error=abs(y-pp);np.savez_compressed(dest/f'{sid}.npz',prediction=pp,uncertainty=sigma,fixed90=fixed,query_ids=obs[sid]['query_ids']);records.append(dict(seed=seed,fold=fold,scene=sid,split='fit_innerOOF' if sid in ids else 'held',fixed90_coverage=(error<=fixed).mean(0).tolist(),fixed90_halfwidth_log=fixed.tolist(),predicted_absolute_error_mean=sigma.mean(0).tolist(),actual_absolute_error_mean=error.mean(0).tolist()))
   (dest/'receipt.json').write_text(json.dumps(dict(fit_scenes=ids,held_scenes=held,checkpoint_sha256=hashlib.sha256((dest/'model.joblib').read_bytes()).hexdigest(),code_commit=subprocess.check_output(['git','-C',str(W/'AV-FPLoc'),'rev-parse','HEAD'],text=True).strip()),indent=2))
 (P/'uncertainty_results.json').write_text(json.dumps(dict(records=records,scope='Ridge predicts log(abs inner-scene-OOF geometry residual+.05) from query embedding. No held target enters fit. Fixed90 uses inner-OOF residual quantiles, no scene-exchangeability guarantee; learned scale is not calibrated posterior. Fit uncertainty outputs are in-sample for this uncertainty regressor; do not claim crossfit uncertainty for secondstage heads.'),indent=2))
if __name__=='__main__':main()

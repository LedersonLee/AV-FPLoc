"""Replay saved held-scene predictions and metrics; never fit or open Test data."""
from pathlib import Path
import json, hashlib, joblib
import numpy as np
from prediction import predict,target,metrics,CARD
from discrimination import load,measures,pairmask,coarse
from integration import features
P=Path('/output'); OLD=Path('/workspace/experiments/directional_tribranch_20261109')
pr=json.loads((P/'PROTOCOL.json').read_text()); checks=[]; models=set()
def read(n):return json.loads((P/n).read_text())
def same(a,b,tag):
 np.testing.assert_allclose(a,b,rtol=1e-10,atol=1e-10);checks.append(dict(check=tag,max_abs=float(np.max(np.abs(np.asarray(a,dtype=float)-np.asarray(b,dtype=float))))))
def model(path,expected=None):
 h=hashlib.sha256(path.read_bytes()).hexdigest()
 if expected:assert h==expected,(path,h,expected)
 models.add(str(path.relative_to(P)));return joblib.load(path)
def record(rows,**keys):return next(r for r in rows if all(r[k]==v for k,v in keys.items()))
def metriccheck(pp,o,row,tag):
 for k,v in metrics(pp,target(o)).items():
  if v is not None and not (isinstance(v,list) and None in v):same(v,row[k],tag+'/'+k)
obs={s:dict(np.load(OLD/'data'/s/'observations.npz')) for s in pr['training_scenes']+pr['transfer_scenes']}
b=read('prediction_results.json');sy=read('symmetric_results.json');fa=read('feature_ablation_results.json');integ=read('integration_results.json');dis=read('discrimination_results.json')
for seed in pr['seeds']:
 for fold,held in enumerate(pr['outer_folds']):
  fit=[s for s in pr['training_scenes'] if s not in held];assert not set(fit)&set(held)
  root=P/'prediction'/f'seed{seed}_fold{fold}'
  for arm,key in [('raw_multi','raw'),('raw_single','raw_single'),('raw_isotropic','raw_isotropic'),('cyclic_ecn','ecn')]:
   receipt=record(b['fits'],seed=seed,fold=fold,arm=arm);assert receipt['fit_scenes']==fit and receipt['held_scenes']==held
   m=model(root/f'{arm}.joblib',receipt['checkpoint_sha256'])
   for sid in held:
    x=np.load(P/'signals'/sid/'features.npz');p=predict(m,x[key]);p=p if arm!='cyclic_ecn' else p[:,CARD];saved=np.load(root/f'{arm}_{sid}.npz')
    same(p,saved['matched'],f'{seed}/{sid}/{arm}');same(saved['query_ids'],obs[sid]['query_ids'],'query_ids')
    metriccheck(p,obs[sid],record(b['records'],seed=seed,scene=sid,model=arm,condition='matched'),arm)
  receipt=record(sy['fits'],seed=seed,fold=fold);m=model(P/'symmetric'/f'seed{seed}_fold{fold}.joblib',receipt['sha256'])
  for sid in held:
   pp=predict(m,np.load(P/'symmetric'/f'{sid}_features.npz')['features']);same(pp,np.load(P/'symmetric'/f'seed{seed}_{sid}.npz')['prediction'],'symmetric');metriccheck(pp,obs[sid],record(sy['records'],seed=seed,scene=sid),'symmetric')
  uroot=P/'uncertainty'/f'seed{seed}_fold{fold}';receipt=read(str((uroot/'receipt.json').relative_to(P)));m=model(uroot/'model.joblib',receipt['checkpoint_sha256']);assert m['fit_scenes']==fit
  for sid in held:
   sigma=np.exp(m['model'].predict((obs[sid]['embedding']-m['mean'])/m['std'])).clip(.05,2);same(sigma,np.load(uroot/f'{sid}.npz')['uncertainty'],'uncertainty')
  prior=np.concatenate([np.log1p(coarse(obs[s]['target'])) for s in fit]).mean(0);conf=record(dis['settings'],seed=seed,fold=fold,kind='B2')
  for arm in ['base','matched','donor','nuisance']:
   receipt=record(integ['fits'],seed=seed,fold=fold,arm=arm);assert receipt['fit_scenes']==fit and receipt['held_scenes']==held
   root2=P/'integration'/f'seed{seed}_fold{fold}';m=model(root2/f'{arm}.joblib',receipt['sha256'])
   for sid in held:
    z=load(sid,seed,'B2');pp=np.load(uroot/f'{sid}.npz')['prediction'];x=features(z,np.roll(pp,17,axis=0) if arm=='donor' else pp,prior)
    if arm=='base':x[...,156:]=0
    if arm=='nuisance':x[...,156:]=np.random.default_rng(int(hashlib.sha256(sid.encode()).hexdigest()[:8],16)+seed).normal(size=x[...,156:].shape)
    gain=m.predict(x.reshape(-1,172)).reshape(128,20);gain[:,0]=0;saved=np.load(root2/f'{arm}_{sid}.npz');same(gain,saved['gain'],'integration/'+arm);same(gain.argmax(1),saved['choice'],'choice')
    sv=np.sort(z['v'],axis=1);got=measures(z,gain,pairmask(z,conf['pair_gap']),sv[:,-1]-sv[:,-2]<=conf['high_ambiguity_topgap'],z['v'].argmax(1));row=record(integ['records'],seed=seed,scene=sid,arm=arm)
    for k,v in got.items():
     if v is not None:same(v,row[k],'integration_metric/'+k)
for fold,held in enumerate(pr['outer_folds']):
 for arm,sl in [('E_energy',slice(0,48)),('N_envelope',slice(48,96)),('C_early',slice(96,324)),('C_full_window',slice(324,552))]:
  receipt=record(fa['fits'],fold=fold,arm=arm);m=model(P/'feature_ablation'/f'{arm}_fold{fold}.joblib',receipt['sha256'])
  for sid in held:
   pp=predict(m,obs[sid]['features'][:,sl]);same(pp,np.load(P/'feature_ablation'/f'{arm}_{sid}.npz')['prediction'],'feature/'+arm);metriccheck(pp,obs[sid],record(fa['records'],scene=sid,arm=arm),'feature/'+arm)
tr=read('transfer/frozen_before_evaluation.json');assert not set(tr['fit_scenes'])&set(tr['held_scenes']);assert tr['fit_scenes']==pr['training_scenes'];m=model(P/'transfer/combined_ridge.joblib',tr['model_sha256'])
for sid in pr['transfer_scenes']:
 pp=predict(m,np.c_[obs[sid]['features'],obs[sid]['embedding']]);same(pp,np.load(P/'transfer'/f'{sid}.npz')['predicted_log_geometry'],'transfer')
assert len(models)==73,len(models)
# Independent body-frame check: left = up cross forward in world XZ; logical yaw decreases.
for yaw in [0,90,180,270]:
 a=np.deg2rad(yaw);forward=np.array([np.cos(a),0,np.sin(a)]);left=np.cross([0,1,0],forward);expected=np.array([np.cos(a-np.pi/2),0,np.sin(a-np.pi/2)]);same(left,expected,'body_frame_left')
(P/'model_replay_verification.json').write_text(json.dumps(dict(status='PASS',new_checkpoints=len(models),numerical_checks=len(checks),maximum_absolute_difference=max(c['max_abs'] for c in checks),models=sorted(models),checks=checks,scope='73 saved new checkpoints replayed on held scenes; metrics checked for raw/cyclic/symmetric/feature and integration. Existing probes separately audited. No retraining, Test reads, or Track B writes.'),indent=2));print('PASS',len(models),'models',len(checks),'numerical checks')

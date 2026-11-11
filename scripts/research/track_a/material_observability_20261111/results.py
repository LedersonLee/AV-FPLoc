"""Factorial effects, held-condition reliability and representative selection."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
RULES=['peak','sharp','time','band','channel','combined']
def auc(y,s):
 y=np.asarray(y,bool);s=np.asarray(s);n=y.sum();m=len(y)-n
 return float((rankdata(s)[y].sum()-n*(n+1)/2)/(n*m)) if n*m else None

def fit_rule(values,y):
 v=np.asarray(values);y=np.asarray(y,bool);thresholds=np.r_[v.min()-1,np.quantile(v,np.linspace(0,1,101)),v.max()+1];options=[]
 for t in thresholds:
  pred=v>=t;fpr=pred[~y].mean();fnr=(~pred[y]).mean()
  if fpr<=.05:options.append((fnr,-t,t))
 t=min(options)[2];edges=np.unique(np.quantile(v,np.linspace(0,1,6)));bins=np.searchsorted(edges[1:-1],v,side='right');probs=[float((y[bins==k].sum()+1)/((bins==k).sum()+2)) for k in range(len(edges)-1)]
 if not probs:edges=np.array([v.min()-1,v.max()+1]);probs=[float((y.sum()+1)/(len(y)+2))]
 return dict(threshold=float(t),edges=edges.tolist(),probs=probs)
def evaluate(rows,rule,model):
 y=np.array([v['state']!='removed' for v in rows]);s=np.array([v['scores'][rule] for v in rows]);pred=s>=model['threshold'];edges=model['edges'];probs=np.array(model['probs'])[np.searchsorted(edges[1:-1],s,side='right')];ece=0.
 for k in range(len(model['probs'])):
  mask=np.searchsorted(edges[1:-1],s,side='right')==k
  if mask.any():ece+=mask.mean()*abs(probs[mask].mean()-y[mask].mean())
 return dict(n=len(rows),tpr=float(pred[y].mean()) if y.any() else None,fpr=float(pred[~y].mean()) if (~y).any() else None,fnr=float((~pred[y]).mean()) if y.any() else None,auc=auc(y,s),brier=float(np.mean((probs-y)**2)),ece=float(ece),unknown_fraction=float((~pred).mean()))
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);root=p.parse_args().root;raw=json.loads((root/'features.json').read_text())['rows'];rows=[v for v in raw if v['mode']=='both'];poses=sorted(set(v['pose']['id'] for v in rows));fit=[v for v in rows if v['pose']['id']=='single2' and v['profile']=='uniform'];refs={k:np.sort([v['features']['scores'][k] for v in fit]) for k in RULES[:-1]}
 for v in rows:
  v['scores']=v['features']['scores'].copy();v['scores']['combined']=float(np.mean([np.searchsorted(refs[k],v['scores'][k],side='right')/len(refs[k]) for k in RULES[:-1]]))
 models={k:fit_rule([v['scores'][k] for v in fit],[v['state']!='removed' for v in fit]) for k in RULES};evaluation=[v for v in rows if v not in fit];summaries=[]
 for name,subset in [('fit',fit),('evaluation',evaluation)]+[(pose,[v for v in evaluation if v['pose']['id']==pose]) for pose in poses]+[(f'absorption{a}',[v for v in evaluation if v['absorption']==a]) for a in [.05,.3,.7,.99]]+[(profile,[v for v in evaluation if v['profile']==profile]) for profile in ['uniform','low','high']]:
  if subset:
   for k in RULES:summaries.append(dict(group=name,rule=k,**evaluate(subset,k,models[k])))
 oracle={};oracle_predictions=[]
 for alpha in [.05,.3,.7,.99]:
  for scatter in [0.,.5,1.]:
   f=[v for v in fit if v['absorption']==alpha and v['scattering']==scatter];model=fit_rule([v['scores']['peak'] for v in f],[v['state']!='removed' for v in f]);oracle[f'{alpha}:{scatter}']=model
   ev=[v for v in evaluation if v['absorption']==alpha and v['scattering']==scatter];oracle_predictions.extend([dict(y=v['state']!='removed',pred=v['scores']['peak']>=model['threshold']) for v in ev])
 oy=np.array([x['y'] for x in oracle_predictions]);op=np.array([x['pred'] for x in oracle_predictions]);oracle_summary=dict(tpr=float(op[oy].mean()),fpr=float(op[~oy].mean()),n=len(oy),inputs='known absorption and scattering used only in oracle threshold stratum')
 def choose(pose,**kw):return [v for v in rows if v['pose']['id']==pose and all(v[k]==value for k,value in kw.items())]
 def scalar(v,feature):
  f=v['features']
  if feature=='log_energy':return float(np.log10(max(np.mean(f['early_energy']),1e-30)))
  if feature=='linear_energy':return float(np.mean(f['early_energy']))
  if feature=='peak_error':return float(np.mean([x for x in f['peak_error_ms'] if x is not None])) if any(x is not None for x in f['peak_error_ms']) else np.nan
  if feature=='spread':return float(np.mean(f['spread_ms']))
  return f[feature]
 effects=[]
 for pose in poses:
  out=dict(pose=pose)
  for feature in ['log_energy','linear_energy','peak_error','spread','spectral_tilt_db']:
   def mean(**kw):return float(np.nanmean([scalar(v,feature) for v in choose(pose,state='present',profile='uniform',**kw)]))
   out[feature]=dict(absorption_70_minus_05=mean(absorption=.7)-mean(absorption=.05),scattering_1_minus_0=float(np.nanmean([scalar(v,feature) for v in choose(pose,state='present',profile='uniform',scattering=1.) if v['absorption']<.99]))-float(np.nanmean([scalar(v,feature) for v in choose(pose,state='present',profile='uniform',scattering=0.) if v['absorption']<.99])),interaction=(mean(absorption=.7,scattering=1.)-mean(absorption=.05,scattering=1.))-(mean(absorption=.7,scattering=0.)-mean(absorption=.05,scattering=0.)))
  prof={name:np.mean([v['features']['spectral_tilt_db'] for v in choose(pose,state='present',profile=name,absorption=.3,scattering=.5)]) for name in ['uniform','low','high']};out['profile_tilts_db']=prof;out['profile_high_minus_low_db']=float(prof['high']-prof['low']);effects.append(out)
 # Robust medoid of four prespecified effects, no extreme-case selection.
 vectors=np.array([[v['log_energy']['absorption_70_minus_05'],v['log_energy']['scattering_1_minus_0'],v['peak_error']['scattering_1_minus_0'],v['profile_high_minus_low_db']] for v in effects]);median=np.median(vectors,0);scale=np.maximum(np.median(abs(vectors-median),0),.01);dist=np.linalg.norm((vectors-median)/scale,axis=1);representative=poses[int(dist.argmin())];exceptions=[]
 extreme=poses[int(dist.argmax())]
 if extreme!=representative:exceptions.append(extreme)
 fn=[]
 for pose in poses:
  ps=choose(pose,state='present');fn.append(np.mean([v['scores']['combined']<models['combined']['threshold'] for v in ps]))
 worst=poses[int(np.argmax(fn))]
 if max(fn)>0 and worst!=representative and worst not in exceptions:exceptions.append(worst)
 represent=dict(default=representative,exceptions=exceptions,criterion='Robust distance to all-six-pose median absorption log-energy / scattering log-energy / scattering peak-error / high-low spectral tilt effects',vectors=vectors.tolist(),median=median.tolist(),mad_scale=scale.tolist(),distances=dict(zip(poses,dist.tolist())),combined_false_negative_by_pose=dict(zip(poses,list(map(float,fn)))),one_pose_sufficient=len(exceptions)==0)
 # Profile identification diagnoses known simulated profiles; not real material identity.
 profile_rows=[v for v in rows if v['state']=='present' and v['absorption']==.3 and v['scattering']==.5];classification=[]
 def vector(v,arm):
  b=np.mean(v['features']['band_early'],axis=0)
  if arm=='energy':return np.array([np.log10(max(b.sum(),1e-30))])
  if arm=='bands':return np.log10(np.maximum(b,1e-30))
  return b/max(b.sum(),1e-30)
 for split in ['context_holdout','hold_single','hold_room','hold_corridor']:
  if split=='context_holdout':tr=[v for v in profile_rows if v['context']<2];ev=[v for v in profile_rows if v['context']==2]
  else:family=split[5:];tr=[v for v in profile_rows if v['pose']['kind']!=family];ev=[v for v in profile_rows if v['pose']['kind']==family]
  for arm in ['energy','bands','shape','shuffle']:
   X=np.stack([vector(v,arm) for v in tr]);Y=np.array([v['profile'] for v in tr]);mu=X.mean(0);sd=np.maximum(X.std(0),1e-9);rng=np.random.default_rng(4211)
   if arm=='shuffle':Y=rng.permutation(Y)
   names=['uniform','low','high'];prototypes=np.stack([((X-mu)/sd)[Y==k].mean(0) for k in names]);pred=[names[int(np.linalg.norm(prototypes-(vector(v,arm)-mu)/sd,axis=1).argmin())] for v in ev];classification.append(dict(split=split,arm=arm,n=len(ev),accuracy=float(np.mean([p==v['profile'] for p,v in zip(pred,ev)])),labels=[v['profile'] for v in ev],predictions=pred))
 movement=[];windows=[];variability=[]
 for pose in poses:
  for alpha in [.05,.3,.7,.99]:
   for scatter in [0.,.5,1.]:
    errors=[];sds=[]
    for ctx in range(3):
     b=choose(pose,state='present',profile='uniform',absorption=alpha,scattering=scatter,context=ctx)[0];m=choose(pose,state='moved',profile='uniform',absorption=alpha,scattering=scatter,context=ctx)[0]
     for c in range(6):
      bp=b['features']['peak_ms'][c];mp=m['features']['peak_ms'][c]
      if bp is not None and mp is not None:errors.append((mp-bp)-(m['features']['expected_ms'][c]-b['features']['expected_ms'][c]))
    for c in range(6):
     v=[r['features']['peak_ms'][c] for r in choose(pose,state='present',profile='uniform',absorption=alpha,scattering=scatter)];v=[v for v in v if v is not None]
     if len(v)>1:sds.append(float(np.std(v,ddof=1)))
    movement.append(dict(pose=pose,absorption=alpha,scattering=scatter,n=len(errors),mean_shift_error_ms=float(np.mean(errors)) if errors else None,max_abs_shift_error_ms=float(max(abs(np.array(errors)))) if errors else None,mean_repeat_sd_ms=float(np.mean(sds)) if sds else None))
  ps=choose(pose,state='present',profile='uniform');windows.append(dict(pose=pose,**{k:float(np.mean([np.mean(v['features'][k])/max(np.mean(v['features']['direct_energy']),1e-30) for v in ps])) for k in ['aligned_energy','shifted_energy','matched_time_energy']}))
 record_predictions=[dict(name=v['name'],pose=v['pose']['id'],state=v['state'],absorption=v['absorption'],scattering=v['scattering'],profile=v['profile'],context=v['context'],scores=v['scores'],predictions={k:bool(v['scores'][k]>=models[k]['threshold']) for k in RULES}) for v in rows]
 result=dict(models=models,fit_score_references={k:v.tolist() for k,v in refs.items()},reliability=summaries,oracle=oracle_summary,oracle_models=oracle,effects=effects,representative=represent,profile_classification=classification,movement=movement,window_controls=windows,predictions=record_predictions,limitations='Wall-state intervention labels, no first-order path GT; six designed poses in three families; fit2m reused environment; contexts not guaranteed independent seeds; Exploratory')
 (root/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(dict(representative=represent,reliability=[v for v in summaries if v['group']=='evaluation'],effects=effects),indent=2))
if __name__=='__main__':main()

"""Fixed-grid temperature diagnostics and scene-family-held development selection."""
import hashlib
import numpy as np
from scipy.special import logsumexp
from matching import circular_error,calibration
TEMPERATURES=np.logspace(-3,3,25)

def curve(scores,xy,truth,temperatures=TEMPERATURES):
 scores=np.asarray(scores,dtype=np.float64);xy=np.asarray(xy);truth=np.asarray(truth);temps=np.asarray(temperatures,float)
 assert scores.shape==(len(xy)*36,) and np.isfinite(scores).all() and (temps>0).all()
 best=int(scores.argmax());cell,yaw=divmod(best,36);dist=np.linalg.norm(xy-truth[:2],axis=1);angle=circular_error(np.arange(36)*10,truth[2]);nearest=int(dist.argmin())*36+int(angle.argmin());gt=((dist[:,None]<=1)&(angle[None]<=30)).reshape(-1)
 pred=((np.linalg.norm(xy-xy[cell],axis=1)[:,None]<=1)&(circular_error(np.arange(36)*10,yaw*10)[None]<=30)).reshape(-1);rows=[]
 centered=scores-scores.max()
 for t in temps:
  z=centered/t;normal=logsumexp(z);lp=z-normal;prob=np.exp(lp)
  rows.append([float(normal-z[nearest]),float(normal-logsumexp(z[gt])) if gt.any() else np.inf,float(prob[pred].sum()),float(-(prob*lp).sum()/np.log(len(scores))),float(prob[gt].sum())])
 return np.array(rows),dict(joint=bool(dist[cell]<=1 and angle[yaw]<=30),top1_id=best,gt_region_exists=bool(gt.any()))

def crossfit(scenes,curves,correct,temperatures=TEMPERATURES,folds=5):
 """Inputs [queries,temperatures,5]. Select mean scene NLL, score held scenes only.
 Column0: nearest-grid NLL,1:region NLL,2:joint confidence,3:entropy,4:GT mass.
 """
 scenes=np.asarray(scenes);values=np.asarray(curves,float);correct=np.asarray(correct,bool);temps=np.asarray(temperatures,float)
 assert values.shape==(len(scenes),len(temps),5) and correct.shape==(len(scenes),)
 assert np.isfinite(values[:,:,[0,2,3,4]]).all()
 families={s:str(s).split('_')[0] for s in np.unique(scenes)};groups=sorted(set(families.values()),key=lambda x:hashlib.sha256(('avloc-temperature-v1:'+x).encode()).hexdigest());assert len(groups)>=folds>=2
 assignment={g:i%folds for i,g in enumerate(groups)};foldid=np.array([assignment[families[s]] for s in scenes]);indices=np.empty(len(scenes),int);records=[]
 for fold in range(folds):
  held=foldid==fold;fit=~held;fit_scenes=np.unique(scenes[fit]);loss=np.array([values[scenes==s,:,0].mean(0) for s in fit_scenes]).mean(0);choice=int(loss.argmin());indices[held]=choice
  records.append(dict(fold=fold,fit_scenes=fit_scenes.tolist(),held_scenes=np.unique(scenes[held]).tolist(),temperature=float(temps[choice]),index=choice,optimum_at_grid_boundary=choice in [0,len(temps)-1],fit_scene_macro_nearest_nll=float(loss[choice])))
 selected=values[np.arange(len(values)),indices];regions=np.isfinite(selected[:,1]);macro_nll=np.mean([selected[scenes==s,0].mean() for s in np.unique(scenes)])
 return dict(protocol='5 scene-family folds by stable hash; equal-scene nearest-grid NLL selection; fixed temperature grid',scope='held-development temperature fitting only; checkpoint selected using full Val19, not independent calibration',folds=records,temperatures=temps.tolist(),selected_indices=indices.tolist(),micro_nearest_grid_nll=float(selected[:,0].mean()),macro_nearest_grid_nll=float(macro_nll),region_nll_defined_queries=int(regions.sum()),micro_region_nll_defined=float(selected[regions,1].mean()) if regions.any() else None,joint_calibration=calibration(selected[:,2],correct),entropy_normalized=float(selected[:,3].mean()),gt_region_mass=float(selected[:,4].mean()))

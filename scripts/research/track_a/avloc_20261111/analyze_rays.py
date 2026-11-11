"""Post-hoc direction/uncertainty/geometry diagnostics of completed predictions.
Labels enter analysis only. No model fitting, checkpoint selection or proposals.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from matching import FOV
from predict_hash import sha


def correlation(x,y):
 x=np.asarray(x,dtype=float).ravel();y=np.asarray(y,dtype=float).ravel()
 valid=np.isfinite(x)&np.isfinite(y);x=x[valid];y=y[valid]
 if len(x)<3 or np.ptp(x)==0 or np.ptp(y)==0:return None
 return float(spearmanr(x,y)[0])


def ray_statistics(mu,scale,target,dirs):
 mu=np.asarray(mu,float);scale=np.asarray(scale,float);target=np.asarray(target,float);dirs=np.asarray(dirs)
 assert mu.shape==scale.shape==target.shape and mu.ndim==2 and mu.shape[1]==len(dirs)
 assert np.isfinite(mu).all() and np.isfinite(scale).all() and np.isfinite(target).all() and (scale>0).all()
 error=abs(mu-target);inside=np.isin(dirs,FOV);bins=[]
 for lo,hi in [(0,1),(1,2),(2,4),(4,8),(8,np.inf)]:
  mask=(target>=lo)&(target<hi);bins.append(dict(lower_m=lo,upper_m=hi if np.isfinite(hi) else None,rays=int(mask.sum()),mae=float(error[mask].mean()) if mask.any() else None))
 coverage=[dict(nominal=p,observed=float((error<=-np.log1p(-p)*scale).mean())) for p in [.5,.8,.9,.95]]
 return dict(queries=len(mu),ray_mae=float(error.mean()),in_fov_mae=float(error[:,inside].mean()),out_fov_mae=float(error[:,~inside].mean()) if (~inside).any() else None,per_direction=[dict(direction_bin=int(d),angle_deg=int(d*10),mae=float(error[:,i].mean()),mean_scale=float(scale[:,i].mean()),scale_abs_error_spearman=correlation(scale[:,i],error[:,i])) for i,d in enumerate(dirs)],distance_bins=bins,pooled_scale_abs_error_spearman=correlation(scale,error),laplace_central_interval_coverage=coverage)


def main():
 p=argparse.ArgumentParser()
 for k in ['predictions','data','evaluation','geometry','out']:p.add_argument('--'+k,type=Path,required=True)
 p.add_argument('--head',choices=['V','A','AV','F'],required=True);a=p.parse_args()
 pred=json.loads((a.predictions/'manifest.json').read_text());ev=json.loads((a.evaluation/'summary.json').read_text());geo=json.loads((a.geometry/'summary.json').read_text());data=json.loads((a.data/'manifest.json').read_text())
 assert pred['complete'] and pred['split']=='val' and pred['queries']==9965
 assert ev['complete'] and ev['queries']==9965 and ev['head']==a.head
 assert geo['complete'] and geo['queries']==9965 and geo['grid_cm']==ev['grid_cm']
 assert ev['prediction_manifest_sha256']==sha(a.predictions/'manifest.json')
 assert pred['input_manifest_sha256']==sha(a.data/'manifest.json')
 expected={r['scene'] for r in data['records'] if r['split']=='val'}
 assert len(expected)==19 and len(pred['records'])==19 and {r['scene'] for r in pred['records']}==expected
 gr={r['scene']:r for r in geo['records']};assert set(gr)==expected
 a.out.mkdir(parents=True,exist_ok=False);dirs=FOV if a.head=='V' else np.arange(36);name='visual_fov' if a.head=='V' else 'omnidirectional'
 mus=[];scales=[];targets=[];allrows=[];scene_stats=[];sources=[]
 for r in pred['records']:
  sid=r['scene'];pp=a.predictions/r['file'];assert sha(pp)==r['sha256'];z=np.load(pp);lab=np.load(a.data/sid/'labels.npz');np.testing.assert_array_equal(z['query_ids'],lab['query_ids'])
  ep=a.evaluation/f'{sid}.json';rows=json.loads(ep.read_text());gp=a.geometry/gr[sid]['file'];assert sha(gp)==gr[sid]['sha256'];gs=json.loads(gp.read_text())
  keys=[(sid,int(q)) for q in z['query_ids']]
  assert [(x['scene'],x['query_id']) for x in rows]==keys and [(x['scene'],x['query_id']) for x in gs]==keys
  mu=z[a.head+'_mu'];scale=z[a.head+'_scale'];target=lab['target'][:,dirs];error=np.abs(mu.astype(float)-target.astype(float)).mean(1)
  np.testing.assert_allclose(error,[x['ray_mae'] for x in rows],rtol=1e-5,atol=1e-6)
  stats=ray_statistics(mu,scale,target,dirs);stats['scene']=sid;stats['geometry_margin_joint_spearman']=correlation([g[name]['joint_geometry_margin_m'] if g[name]['joint_geometry_margin_m'] is not None else np.nan for g in gs],[x['joint'] for x in rows]);scene_stats.append(stats)
  for i,row in enumerate(rows):allrows.append(dict(scene=sid,query_id=int(z['query_ids'][i]),ray_mae=float(error[i]),joint=bool(row['joint']),fine=bool(row['fine']),position_error_m=row['position_error_m'],yaw_error_deg=row['yaw_error_deg'],geometry_margin_m=gs[i][name]['joint_geometry_margin_m']))
  mus.append(mu);scales.append(scale);targets.append(target);sources.append(dict(scene=sid,prediction_sha256=sha(pp),labels_sha256=sha(a.data/sid/'labels.npz'),evaluation_sha256=sha(ep),geometry_sha256=sha(gp)))
 assert len(allrows)==9965
 summary=ray_statistics(np.concatenate(mus),np.concatenate(scales),np.concatenate(targets),dirs)
 # Fixed diagnostic thresholds; no method or checkpoint selection. Keep full IDs
 # locally, plus uniform random and largest-discrepancy cases for balanced review.
 groups={'high_ray_error_pose_correct':[r for r in allrows if r['ray_mae']>=1 and r['joint']], 'low_ray_error_pose_wrong':[r for r in allrows if r['ray_mae']<=.2 and not r['joint']], 'fine_success':[r for r in allrows if r['fine']]}
 rng=np.random.default_rng(20261111);cases={}
 for k,rows in groups.items():
  random=[rows[i] for i in rng.choice(len(rows),min(5,len(rows)),replace=False)]
  ranked=sorted(rows,key=lambda r:r['ray_mae'],reverse=k=='high_ray_error_pose_correct')[:5]
  cases[k]=dict(count=len(rows),uniform_random=random,ray_error_extremes=ranked)
 summary.update(complete=True,head=a.head,seed=pred['seed'],grid_cm=ev['grid_cm'],scene_statistics=scene_stats,cases=cases,sources=sources,scope='Reused Val19 post-hoc descriptive diagnostics. Rays/queries within scenes are correlated; pooled correlations and interval coverage do not establish independent calibration or causality.',test_accessed=False,source_sha256={n:sha(Path(__file__).with_name(n)) for n in ['analyze_rays.py','matching.py','predict_hash.py']},prediction_manifest_sha256=sha(a.predictions/'manifest.json'),evaluation_summary_sha256=sha(a.evaluation/'summary.json'),geometry_summary_sha256=sha(a.geometry/'summary.json'))
 (a.out/'queries.json').write_text(json.dumps(allrows)+'\n');(a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 print('RAY ANALYSIS COMPLETE',len(allrows),flush=True)
if __name__=='__main__':main()

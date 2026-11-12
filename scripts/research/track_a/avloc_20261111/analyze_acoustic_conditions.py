"""Post-hoc acoustic success conditions; GT-derived strata never enter inference."""
import argparse,json
from pathlib import Path
import numpy as np
from paired_summary import align
from analyze_rays import correlation
from predict_hash import sha


def describe(rows):
 if not rows:return {'queries':0}
 return {'queries':len(rows),'scenes':len({r['scene'] for r in rows}),
  **{k:float(np.mean([r[k] for r in rows])) for k in ['r1','joint','fine']},
  **{k+'_median':float(np.median([r[k] for r in rows])) for k in ['ray_mae','yaw_error_deg','margin']},
  'fine_with_wrong_yaw':sum(r['fine'] and r['yaw_error_deg']>30 for r in rows),
  'per_scene_counts':{s:sum(r['scene']==s for r in rows) for s in sorted({r['scene'] for r in rows})}}


def main():
 p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 result={'complete':True,'seed':42,'test_accessed':False,'source_sha256':sha(__file__),'grids':{},'scope':'Exploratory reused Val19; post-hoc quartiles and GT geometry. No inference filtering, no causal acoustic-observability claim. Queries/rays are not independent scenes.'}
 for grid in [20,10]:
  root=a.study/f'evaluation/A1_seed42_{grid}cm';ev=json.loads((root/'summary.json').read_text());assert ev['complete'] and ev['queries']==9965 and ev['grid_cm']==grid
  geo=a.study/f'observability/geometry{grid}';gm=json.loads((geo/'summary.json').read_text());rows=[];sources=[]
  for rec in gm['records']:
   sid=rec['scene'];gp=geo/rec['file'];assert sha(gp)==rec['sha256'];geometry=json.loads(gp.read_text());ep=root/f'{sid}.json';rs=json.loads(ep.read_text());rs=align(rs,[(r['scene'],r['query_id']) for r in geometry])
   for r,g in zip(rs,geometry):rows.append({**r,'margin':g['omnidirectional']['joint_geometry_margin_m'],'perfect_joint':g['omnidirectional']['perfect_profile_joint']})
   sources.append({'scene':sid,'evaluation_sha256':sha(ep),'geometry_sha256':sha(gp)})
  assert len(rows)==9965 and len({r['scene'] for r in rows})==19
  for k in ['r1','joint','fine']:np.testing.assert_allclose(np.mean([r[k] for r in rows]),ev['metric_means'][k],rtol=0,atol=1e-12)
  strata={}
  for key in ['ray_mae','margin']:
   vals=np.array([r[key] for r in rows]);cuts=np.quantile(vals,[.25,.5,.75]);groups=np.searchsorted(cuts,vals,side='right')
   strata[key]={'quartile_cutpoints':cuts.tolist(),'quartiles':[describe([r for r,g in zip(rows,groups) if g==i]) for i in range(4)]}
  low,high=np.quantile([r['ray_mae'] for r in rows],[.25,.75])
  groups={'all':rows,'fine_success':[r for r in rows if r['fine']],'joint_success':[r for r in rows if r['joint']],
   'low_ray_error_joint_failure':[r for r in rows if r['ray_mae']<=low and not r['joint']],
   'high_ray_error_joint_success':[r for r in rows if r['ray_mae']>=high and r['joint']],
   'ideal_geometry_success_acoustic_failure':[r for r in rows if r['perfect_joint'] and not r['joint']]}
  scenes=[]
  for sid in sorted({r['scene'] for r in rows}):
   rs=[r for r in rows if r['scene']==sid];scenes.append({'scene':sid,'ray_mae_joint_spearman':correlation([r['ray_mae'] for r in rs],[r['joint'] for r in rs]),'margin_joint_spearman':correlation([r['margin'] for r in rs],[r['joint'] for r in rs])})
  result['grids'][str(grid)]={'groups':{k:describe(v) for k,v in groups.items()},'strata':strata,'scene_correlations':scenes,'sources':sources}
 a.out.mkdir(parents=True,exist_ok=False);(a.out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 print(json.dumps({g:r['groups'] for g,r in result['grids'].items()},indent=2))
if __name__=='__main__':main()

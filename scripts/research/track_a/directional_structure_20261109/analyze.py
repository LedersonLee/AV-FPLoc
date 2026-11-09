import json,collections,itertools
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;PR=json.loads((P/'PROTOCOL.json').read_text());SC=PR['training_scenes'];boot=np.random.default_rng(20261109).integers(0,len(SC),(20000,len(SC)))
def read(n):return json.loads((P/n).read_text())
def vec(rows,key):return np.array([np.mean([r[key] for r in rows if r['scene']==s and r[key] is not None]) for s in SC])
def ci(x):return dict(mean=float(np.mean(x)),ci95=np.quantile(np.array(x)[boot].mean(1),[.025,.975]).tolist(),scene_values=np.array(x).tolist())
def group(rows,keys,metrics):
 out={}
 for key in sorted(set(tuple(r[k] for k in keys) for r in rows)):
  rr=[r for r in rows if tuple(r[k] for k in keys)==key];out['/'.join(map(str,key))]={m:ci(vec(rr,m)) for m in metrics}
 return out
b=read('prediction_results.json')['records'];c=read('discrimination_results.json')['records'];d=read('fullbank_results.json')['records'];e=read('integration_results.json')['records'];out=dict(scope='12 scenes, seeds averaged within scene,20k paired scene bootstrap; reused teacher-exposed exploratorydata; no multiple-comparison adjustment',prediction=group(b,['model','condition'],['mae_m','median_m','p90_m','balanced_accuracy','auroc','rank_correlation','pair_order_accuracy']),discrimination=group(c,['kind','arm'],['joint','fine','pair_accuracy','ranking_auroc','GT_success_rank','high_ambiguity_joint','repair','harm','coverage']),fullbank=group(d,['arm'],['joint','fine','R1','top5','top20','gt_neighborhood_score_mass']),integration=group(e,['arm'],['joint','fine','pair_accuracy']))
contrasts={}
for kind in ['pure_visual','B2']:
 for a,bname in [('legacy_ridge_visual_geometry','visual'),('legacy_ridge_visual_geometry','visual_cosine'),('legacy_ridge_geometry','donor17_geometry'),('legacy_ridge_geometry','geometry_prior'),('legacy_ridge_geometry','cosine')]:
  aa=[r for r in c if r['kind']==kind and r['arm']==a];bb=[r for r in c if r['kind']==kind and r['arm']==bname];contrasts[f'{kind}/{a}-{bname}']={m:ci(vec(aa,m)-vec(bb,m)) for m in ['joint','fine','pair_accuracy']}
for a,bname in [('visual_structure','visual'),('structure','donor_structure'),('cosine_structure','cosine')]:contrasts[f'fullbank/{a}-{bname}']={m:ci(vec([r for r in d if r['arm']==a],m)-vec([r for r in d if r['arm']==bname],m)) for m in ['joint','fine','top20']}
for arm in ['base','matched','donor','nuisance','fixed_addition']:contrasts[f'integration/{arm}-GEOQ']={m:ci(vec([r for r in e if r['arm']==arm],m)-vec([r for r in c if r['kind']=='B2' and r['arm']=='GEOQ'],m)) for m in ['joint','fine']}
out['contrasts']=contrasts;out['costs']=read('fullbank_results.json')['costs'];out['repeat_by_role']={role:{m:float(np.mean([r[m] for r in read('repeat_results.json')['records'] if r['role']==role])) for m in ['mae_m','pair_distance_difference_m','near_label_agreement']} for role in ['Observed','FP']};(P/'analysis.json').write_text(json.dumps(out,indent=2));print(json.dumps(contrasts,indent=2))
# Additional preregistered followups and exploratory transfer, independently grouped by scene.
sym=read('symmetric_results.json')['records'];out['symmetric']={k:ci(vec(sym,k)) for k in ['mae_m','balanced_accuracy','auroc','rank_correlation','pair_order_accuracy']}
raw=[r for r in b if r['model']=='raw_multi' and r['condition']=='matched'];out['raw_vs_symmetric']={k:ci(vec(raw,k)-vec(sym,k)) for k in ['mae_m','balanced_accuracy','auroc','pair_order_accuracy']}
oracle=read('oracle_results.json')['records'];out['oracle']={k:ci(vec(oracle,k)) for k in ['joint','fine','top5','top20','nearest_grid_log_geometry_error']}
u=[r for r in read('uncertainty_results.json')['records'] if r['split']=='held'];out['uncertainty']=dict(fixed90_coverage=np.mean([r['fixed90_coverage'] for r in u],0).tolist(),fixed90_halfwidth_log=np.mean([r['fixed90_halfwidth_log'] for r in u],0).tolist(),predicted_absolute_error_mean=np.mean([r['predicted_absolute_error_mean'] for r in u],0).tolist(),actual_absolute_error_mean=np.mean([r['actual_absolute_error_mean'] for r in u],0).tolist())
tr=read('transfer_results.json');TS=PR['transfer_scenes'];tb=np.random.default_rng(20261110).integers(0,len(TS),(20000,len(TS)))
def tv(rows,k):return np.array([np.mean([r[k] for r in rows if r['scene']==s and r[k] is not None]) for s in TS])
def tc(x):return dict(mean=float(x.mean()),ci95=np.quantile(x[tb].mean(1),[.025,.975]).tolist(),scene_values=x.tolist())
out['transfer']={arm:{k:tc(tv([r for r in tr['records'] if r['arm']==arm],k)) for k in ['joint','fine','pair_accuracy','ranking_auroc','repair','harm']} for arm in sorted(set(r['arm'] for r in tr['records']))};out['transfer_baselines']={arm:{k:tc(tv([r for r in tr['baselines'] if r['arm']==arm],k)) for k in ['joint','fine']} for arm in sorted(set(r['arm'] for r in tr['baselines']))};out['transfer_contrasts']={}
for a,bname in [('visual_structure','visual'),('visual_structure','visual_cosine'),('visual_structure','visual_donor'),('structure','donor')]:out['transfer_contrasts'][a+'-'+bname]={k:tc(tv([r for r in tr['records'] if r['arm']==a],k)-tv([r for r in tr['records'] if r['arm']==bname],k)) for k in ['joint','fine','pair_accuracy']}
out['transfer_vs_GEOQ']={k:tc(tv([r for r in tr['records'] if r['arm']=='visual_structure'],k)-tv([r for r in tr['baselines'] if r['arm']=='GEOQ'],k)) for k in ['joint','fine']}
(P/'analysis.json').write_text(json.dumps(out,indent=2));print('TRANSFER CONTRAST',out['transfer_contrasts']['visual_structure-visual'])

# Complete direction-wise and stratified downstream reporting without refitting.
fa=read('feature_ablation_results.json')['records']
out['feature_ablation']=group(fa,['arm'],['mae_m','median_m','p90_m','balanced_accuracy','auroc','rank_correlation','pair_order_accuracy'])
out['direction_metrics']={}
for model in sorted(set(r['model'] for r in b)):
 rows=[r for r in b if r['model']==model and r['condition']=='matched']
 out['direction_metrics'][model]={m:[ci(np.array([np.mean([r[m][j] for r in rows if r['scene']==s]) for s in SC])) for j in range(4)] for m in ['direction_mae_m','direction_median_m','direction_p90_m','direction_balanced_accuracy','direction_auroc']}
out['discrimination_extended']=group(c,['kind','arm'],['R1','conditional_joint','visual_success_preservation','success_minus_failure_margin','high_ambiguity_joint'])
strata=[]; downstream=[]
for fold,held in enumerate(PR['outer_folds']):
 for sid in held:
  zz=np.load(P/'discrimination'/f'pure_visual_seed42_fold{fold}'/f'{sid}.npz')
  pred=np.load(P/'fullbank'/sid/'predictions.npz')
  np.testing.assert_array_equal(zz['query_ids'],pred['query_ids'])
  visual=pred['visual_labels'][:,0,1]
  for arm in sorted(set(r['arm'] for r in d)):
   lab=pred[arm+'_labels'];joint=lab[:,0,1];covered=lab[:,:,1].any(1)
   downstream.append(dict(scene=sid,arm=arm,conditional_joint=float(joint[covered].mean()),repair=float((joint&~visual).mean()),harm=float((~joint&visual).mean()),visual_success_preservation=float(joint[visual].mean())))
   for name,mask in [('high',zz['high_ambiguity']),('other',~zz['high_ambiguity'])]:
    strata.append(dict(scene=sid,arm=arm,stratum=name,queries=int(mask.sum()),joint=float(joint[mask].mean()),top5=float(lab[mask,:5,1].any(1).mean()),top20=float(covered[mask].mean())))
out['fullbank_ambiguity']=group(strata,['stratum','arm'],['joint','top5','top20'])
out['fullbank_downstream']=group(downstream,['arm'],['conditional_joint','repair','harm','visual_success_preservation'])
out['fullbank_stratum_counts']=strata
(P/'analysis.json').write_text(json.dumps(out,indent=2))
print('Extended direction metrics, feature ablations, GT-free ambiguity strata and fullbank repair/harm complete.')

out['ordinal_openness']=group(b,['model','condition'],['ordinal_accuracy','openness_accuracy'])
(P/'analysis.json').write_text(json.dumps(out,indent=2))

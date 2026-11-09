"""Independent numerical checks of metrics and candidate score semantics."""
from candidate_conditioned import *
from scipy.stats import spearmanr,rankdata

def main():
 rows=json.loads((P/'candidate_evaluation.json').read_text())['records'];conditioned=json.loads((P/'conditioned_candidate_results.json').read_text())['records'];errors=[]
 for r in conditioned:
  if r['method']!='uniform' or r['control']!='matched':continue
  rr=[x for x in rows if x['group']==r['group'] and x['kind']==r['kind'] and x['seed']==r['seed'] and x['scene']==r['scene'] and x['trained_k']==36 and x['eval_k']==36 and x['control']=='matched' and x['method']==r['fusion']];assert len(rr)==1
  for metric in ['joint','fine','R1','pair_accuracy','repair','harm']:errors.append(abs(rr[0][metric]-r[metric]))
 assert max(errors)==0
 sid=PR['training_scenes'][0];o=np.load(OLD/'data'/sid/'observations.npz');pp=np.load(P/'acoustic'/'raw_k36_seed42_fold0'/f'{sid}.npz')['prediction'];target=represent(o['target'],'raw36');ra=rankdata(pp,axis=1);rb=rankdata(target,axis=1);ra-=ra.mean(1,keepdims=True);rb-=rb.mean(1,keepdims=True);fast=(ra*rb).sum(1)/np.sqrt((ra*ra).sum(1)*(rb*rb).sum(1));slow=np.array([spearmanr(a,b).statistic for a,b in zip(pp,target)]);np.testing.assert_allclose(fast,slow,atol=1e-14)
 z=dict(np.load(PREV/'candidates'/sid/'pure_visual.npz'));g=np.log1p(represent(z['geometry'][:2],'raw36'));p=pp[:2];fast_pair=pair_score(g,p);slow_pair=np.zeros((2,20))
 for q in range(2):
  for i in range(20):
   for j in range(20):
    diff=abs(g[q,i]-g[q,j]);ix=np.argsort(diff,kind='stable')[-9:];ix=ix[diff[ix]>1e-6]
    if len(ix):slow_pair[q,i]+=np.mean(abs(g[q,j,ix]-p[q,ix])-abs(g[q,i,ix]-p[q,ix]))/20
 np.testing.assert_allclose(fast_pair,slow_pair,atol=1e-14)
 # Shared fullbank implementation uses vs for anchor; original uses standardized visual.
 anchors=0
 for fold,held in enumerate(PR['outer_folds']):
  for scene in held:
   zz=np.load(PREV/'candidates'/scene/'pure_visual.npz');mask=np.load(PREV/'discrimination'/f'pure_visual_seed42_fold{fold}'/f'{scene}.npz');assert np.array_equal(zz['vs'].argmax(1),mask['visual'].argmax(1));anchors+=len(zz['vs'])
 bg=np.load(PREV/'candidates'/sid/'bank_geometry.npz');dense=bg['depth'][:,(np.arange(36)[:,None]+np.arange(36))%36].reshape(-1,36);bank=np.log1p(represent(dense,'raw36'));anchor=int(z['indices'][0,z['vs'][0].argmax()]);e=abs(bank-pp[0]);w=selection_weights(abs(bank-bank[anchor]));score=(w*(e[anchor]-e)).sum(-1);rng=np.random.default_rng(7)
 for i in rng.choice(len(bank),size=min(64,len(bank)),replace=False):
  delta=abs(bank[i]-bank[anchor]);ix=np.argsort(delta,kind='stable')[-9:];ix=ix[delta[ix]>1e-6];expected=float(np.mean(e[anchor,ix]-e[i,ix])) if len(ix) else 0.;assert abs(score[i]-expected)<1e-12
 result=dict(uniform_metric_comparisons=len(errors),uniform_max_error=max(errors),spearman_max_error=float(abs(fast-slow).max()),pair_score_max_error=float(abs(fast_pair-slow_pair).max()),anchor_equivalence_queries=anchors,anchor_score_scalar_checks=64,scope='Numerical metric and score checks; selection function accepts FP differences only. No retraining or Test access.')
 (P/'matching_verification.json').write_text(json.dumps(result,indent=2));print(result)
if __name__=='__main__':main()

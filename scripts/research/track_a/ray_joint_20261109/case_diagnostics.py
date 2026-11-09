"""Deterministically choose one repair and harm; diagnostic GT never affects scoring."""
from candidate_conditioned import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 cases=[]
 for scene in PR['training_scenes']:
  fold=next(i for i,x in enumerate(PR['outer_folds']) if scene in x);seed=42;z=np.load(PREV/'candidates'/scene/'pure_visual.npz');o=np.load(OLD/'data'/scene/'observations.npz');scores=np.load(P/'conditioned_candidates'/f'acoustic_raw_seed{seed}_fold{fold}'/f'{scene}.npz');g=np.log1p(represent(z['geometry'],'raw36'));pred=np.load(P/'acoustic'/f'raw_k36_seed{seed}_fold{fold}'/f'{scene}.npz')['prediction'];uni=scores['matched_uniform_visual_geometry'].argmax(1);pair=scores['matched_pair_selection_visual_geometry'].argmax(1)
  # Read the exact existing candidate success matrix.
  from discrimination import measures
  for q in range(128):
   if uni[q]==pair[q]:continue
   inds=z['indices'][q];bg=np.load(PREV/'candidates'/scene/'bank_geometry.npz');xy=bg['xy'];distance=np.linalg.norm(xy[inds//36]-o['truth'][q,:2],axis=1);yaw=abs((inds%36*10-o['truth'][q,2]+180)%360-180);success=(distance<=1)&(yaw<=30);u=int(uni[q]);v=int(pair[q]);label='repair' if success[v] and not success[u] else 'harm' if success[u] and not success[v] else 'unchanged_failure'
   if label not in ['repair','harm'] or any(c['type']==label for c in cases):continue
   dirs=np.flatnonzero(selection_weights(abs(g[q,u]-g[q,v]))>0);case=dict(type=label,scene=scene,query_id=int(o['query_ids'][q]),query_index=q,seed=seed,uniform_candidate=u,pair_candidate=v,uniform_error_m=float(distance[u]),pair_error_m=float(distance[v]),uniform_yaw_error=float(yaw[u]),pair_yaw_error=float(yaw[v]),pair_directions=dirs.tolist(),selection='First qualifying query in fixed protocol scene order; retrospective illustrative case, not representative effect estimate.');cases.append(case)
   fig,ax=plt.subplots(figsize=(8,3.5));degrees=np.arange(36)*10;ax.plot(degrees,np.expm1(pred[q]).clip(0,10),label='Predicted acoustic');ax.plot(degrees,np.expm1(g[q,u]),label='Uniform choice');ax.plot(degrees,np.expm1(g[q,v]),label='Pair choice');ax.scatter(degrees[dirs],np.expm1(g[q,v,dirs]),marker='x',s=50,label='Directions for this pair');ax.set(xlabel='Body-frame ray angle (degrees)',ylabel='Distance (m)',title=f'{label}: {scene}, query {int(o["query_ids"][q])}');ax.legend(fontsize=8,ncol=2);fig.tight_layout();(P/'figures').mkdir(exist_ok=True);fig.savefig(P/'figures'/f'pair_{label}.png',dpi=160);plt.close(fig)
   if len(cases)==2:break
  if len(cases)==2:break
 (P/'case_examples.json').write_text(json.dumps(dict(cases=cases,scope='Pair-selection changes candidate ordering; good geometry separation need not be predictable from audio.'),indent=2));print(cases)
if __name__=='__main__':main()

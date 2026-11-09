from candidate_conditioned import *
def main():
 perm=np.random.default_rng(20261109).permutation(36);records=[]
 for seed in PR['seeds']:
  for fold,held in enumerate(PR['outer_folds']):
   for sid in held:
    z=dict(np.load(PREV/'candidates'/sid/'pure_visual.npz'));mask=np.load(PREV/'discrimination'/f'pure_visual_seed{seed}_fold{fold}'/f'{sid}.npz');z['v']=mask['visual'];g=np.log1p(represent(z['geometry'],'raw36'))
    for group,kind in [('acoustic','raw'),('acoustic','chirp'),('acoustic','embedding'),('joint','attention')]:
     root=P/group/(f'{kind}_k36_seed{seed}_fold{fold}' if group=='acoustic' else f'{kind}_seed{seed}_fold{fold}');pp=np.load(root/f'{sid}.npz')['prediction'][:,perm];sigma=np.load(P/'calibration'/f'{group}_{kind}_seed{seed}_fold{fold}'/'scales.npz')['sigma'];weights=1/sigma;weights/=weights.sum();scores=dict(uniform=-abs(g-pp[:,None]).mean(-1),uncertainty=-(abs(g-pp[:,None])*weights).sum(-1),pair_selection=pair_score(g,pp))
     for method,score in scores.items():
      for fusion in ['geometry','visual_geometry']:
       sc=score if fusion=='geometry' else .85*z['v']+.15*standard(score);records.append(dict(scene=sid,seed=seed,fold=fold,group=group,kind=kind,control='fixed_random_direction_permutation',method=method,fusion=fusion,**measures(z,sc,mask['ambiguous_mask'],mask['high_ambiguity'],z['v'].argmax(1))))
 (P/'direction_permutation_results.json').write_text(json.dumps(dict(permutation=perm.tolist(),records=records,scope='Fixed visual Top20 only; supplementary stronger control than registered10degree rotation; no fitting/selection',test_accessed=False),indent=2));print('DONE',len(records))
if __name__=='__main__':main()

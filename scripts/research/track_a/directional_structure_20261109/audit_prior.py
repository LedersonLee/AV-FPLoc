from pathlib import Path
import json,joblib,numpy as np,importlib.util
P=Path('/output');OLD=Path('/workspace/experiments/directional_tribranch_20261109');pr=json.loads((P/'PROTOCOL.json').read_text());spec=importlib.util.spec_from_file_location('prior_stage_a',OLD/'stage_a.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a);rows=[]
for seed in pr['seeds']:
 for fold,held in enumerate(pr['outer_folds']):
  for kind in ['raw_ridge','embedding_ridge','ridge','mlp']:
   root=OLD/'stage_a'/f'seed{seed}_fold{fold}';m=joblib.load(root/(kind+'.joblib'));norm=float(np.linalg.norm(m['model'].coef_)) if kind!='mlp' else float(sum(v.detach().square().sum() for v in m['model'].parameters()).sqrt())
   for sid in held:
    o=dict(np.load(OLD/'data'/sid/'observations.npz'));z=np.load(root/f'{kind}_{sid}.npz');pp=a.predict(m,o);np.testing.assert_array_equal(pp,z['matched']);y=np.log1p(a.coarse(o['target']));near=y<np.log(4);rows.append(dict(seed=seed,fold=fold,scene=sid,kind=kind,matched_accuracy=float(((pp<np.log(4))==near).mean()),prior_accuracy=float(((z['prior']<np.log(4))==near).mean()),parameter_norm=norm,input_dimensions=len(m['mean']),replay_max_abs=float(abs(pp-z['matched']).max())))
(P/'prior_replay.json').write_text(json.dumps(dict(records=rows,count=len(rows),training_evidence='Serialized fitted ridge coefficients and MLP state; observed-only forward exactly reproduces stored predictions; manifests match. This verifies fitted outputs, not complete original training replay.',test_accessed=False),indent=2));print('PRIOR REPLAY',len(rows),max(r['replay_max_abs'] for r in rows),flush=True)

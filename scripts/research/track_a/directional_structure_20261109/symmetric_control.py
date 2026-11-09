"""Same84-feature ridge budget, but remove absolute channel identity by canonical sorting."""
from pathlib import Path
import json,hashlib,joblib
import numpy as np
from prepare_signal import descriptor
from prediction import fit,predict,target,metrics
P=Path('/output');W=Path('/workspace');OLD=W/'experiments/directional_tribranch_20261109';PR=json.loads((P/'PROTOCOL.json').read_text());signals={};obs={};receipts=[]
for sid in PR['training_scenes']:
 r=json.loads((P/'signals'/sid/'receipt.json').read_text());x=[]
 for row in r['raw_records']:
  p=Path(row['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'];raw=np.load(p);order=np.argsort((raw[:,103:6151].astype(float)**2).sum(-1),kind='stable');v=descriptor(raw[order]);permuted=raw[[2,4,0,5,1,3]];order2=np.argsort((permuted[:,103:6151].astype(float)**2).sum(-1),kind='stable');np.testing.assert_array_equal(v,descriptor(permuted[order2]));x.append(v)
 signals[sid]=np.stack(x);obs[sid]=dict(np.load(OLD/'data'/sid/'observations.npz'));dest=P/'symmetric';dest.mkdir(exist_ok=True);np.savez_compressed(dest/f'{sid}_features.npz',features=signals[sid],query_ids=obs[sid]['query_ids'])
records=[]
for seed in PR['seeds']:
 for fold,held in enumerate(PR['outer_folds']):
  ids=[s for s in signals if s not in held];x=np.concatenate([signals[s] for s in ids]);y=np.concatenate([target(obs[s]) for s in ids]);m=fit(x,y);path=dest/f'seed{seed}_fold{fold}.joblib';joblib.dump(m,path);receipts.append(dict(seed=seed,fold=fold,fit_scenes=ids,held_scenes=held,input_dimensions=84,varying_features=int((x.std(0)>.00001).sum()),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
  for sid in held:
   pp=predict(m,signals[sid]);np.savez_compressed(dest/f'seed{seed}_{sid}.npz',prediction=pp);records.append(dict(scene=sid,seed=seed,fold=fold,**metrics(pp,target(obs[sid]))))
(P/'symmetric_results.json').write_text(json.dumps(dict(records=records,fits=receipts,scope='Canonical channel sort by reflectedenergy removes absolute microphone identity; exact permutation invariance verified. Same84input ridge and4outputs asrawmulti, with variable feature counts reported. Not a learned physical direction model.',test_accessed=False),indent=2));print('SYMMETRIC CONTROL COMPLETE',len(records))

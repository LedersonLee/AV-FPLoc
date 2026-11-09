"""Deterministic one-seed ridge feature isolation; no redundant seed replication."""
import json,joblib,hashlib
from pathlib import Path
import numpy as np
from prediction import fit,predict,target,metrics
P=Path('/output');OLD=Path('/workspace/experiments/directional_tribranch_20261109');PR=json.loads((P/'PROTOCOL.json').read_text());obs={s:dict(np.load(OLD/'data'/s/'observations.npz')) for s in PR['training_scenes']};rows=[];receipts=[];dest=P/'feature_ablation';dest.mkdir(exist_ok=True)
for fold,held in enumerate(PR['outer_folds']):
 ids=[s for s in obs if s not in held]
 for arm,sl in [('E_energy',slice(0,48)),('N_envelope',slice(48,96)),('C_early',slice(96,324)),('C_full_window',slice(324,552))]:
  x=np.concatenate([obs[s]['features'][:,sl] for s in ids]);y=np.concatenate([target(obs[s]) for s in ids]);m=fit(x,y);f=dest/f'{arm}_fold{fold}.joblib';joblib.dump(m,f);receipts.append(dict(arm=arm,fold=fold,fit_scenes=ids,held_scenes=held,dimensions=x.shape[1],sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
  for sid in held:
   p=predict(m,obs[sid]['features'][:,sl]);rows.append(dict(scene=sid,fold=fold,arm=arm,**metrics(p,target(obs[sid]))));np.savez_compressed(dest/f'{arm}_{sid}.npz',prediction=p)
(P/'feature_ablation_results.json').write_text(json.dumps(dict(records=rows,fits=receipts,scope='Same regularizer/data/target;different feature counts except equal228early/fullcoherence. Early<50ms vs full2..128ms, not late-only. E flattened-bin averages not physical temporal sectors. One deterministic ridge seed only.',test_accessed=False),indent=2));print('FEATURE ABLATIONS',len(receipts))

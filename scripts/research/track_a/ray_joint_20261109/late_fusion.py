from pathlib import Path
import json,hashlib,numpy as np
P=Path('/output');pr=json.loads((P/'PROTOCOL.json').read_text())
for seed in pr['seeds']:
 for fold,held in enumerate(pr['outer_folds']):
  v=P/'joint'/f'vision_seed{seed}_fold{fold}';a=P/'joint'/f'acoustic_seed{seed}_fold{fold}';dst=P/'joint'/f'late_seed{seed}_fold{fold}';dst.mkdir(exist_ok=True)
  for sid in held:
   vv=np.load(v/f'{sid}.npz');aa=np.load(a/f'{sid}.npz');np.testing.assert_array_equal(vv['query_ids'],aa['query_ids']);np.savez_compressed(dst/f'{sid}.npz',prediction=.5*(vv['prediction']+aa['prediction']),donor=.5*(vv['prediction']+aa['donor']),query_ids=vv['query_ids'])
  (dst/'receipt.json').write_text(json.dumps(dict(kind='late',seed=seed,fold=fold,k=36,held_scenes=held,fit_scenes=[s for s in pr['training_scenes'] if s not in held],rule='Fixed0.5/0.5 mean of independent log-ray predictions; no new fit',source_receipt_hashes={str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [v/'receipt.json',a/'receipt.json']}),indent=2))
print('LATE FUSION COMPLETE; no new fit')

"""Hash local input dependencies and actual source/checkpoint provenance; no raw data export."""
from pathlib import Path
import json,hashlib,subprocess
P=Path(__file__).resolve().parent;W=P.parent.parent;R=W/'AV-FPLoc';PR=json.loads((P/'PROTOCOL.json').read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
files=[]
for s in PR['training_scenes']+PR['transfer_scenes']:
 for p in [P/'audio'/s/'features.npz',P/'audio'/s/'receipt.json',W/'experiments/directional_tribranch_20261109/data'/s/'observations.npz',W/'experiments/directional_structure_20261109/candidates'/s/'pure_visual.npz']:
  files.append(dict(path=str(p.relative_to(W)),bytes=p.stat().st_size,sha256=sha(p)))
sources=[]
for p in sorted(P.glob('*.py'))+sorted(P.glob('*.sh')):
 copy=R/'scripts/research/track_a/ray_joint_20261109'/p.name
 sources.append(dict(path=str(p.relative_to(W)),sha256=sha(p),git_copy=str(copy.relative_to(R)) if copy.exists() else None,copy_equal=sha(p)==sha(copy) if copy.exists() else False))
receipts=[]
for group in ['acoustic','joint','joint_followup','transfer_models']:
 for p in sorted((P/group).glob('*/receipt.json')):
  r=json.loads(p.read_text());receipts.append(dict(path=str(p.relative_to(P)),kind=r['kind'],code_commit=r.get('code_commit','UNRECORDED'),checkpoint_sha256=r.get('checkpoint_sha256'),updates=r.get('updates'),parameters=r.get('parameters'),seconds=r.get('seconds'),fit_scenes=r.get('fit_scenes'),held_scenes=r.get('held_scenes')))
for p in sorted((P/'calibration').glob('*/complete.json')):
 r=json.loads(p.read_text());receipts.append(dict(path=str(p.relative_to(P)),code_commit=r['code_commit'],inner_fits=r['inner_fits']))
(P/'source_inventory.json').write_text(json.dumps(dict(files=sources,code_base=PR['base_commit'],note='Execution receipts record repository HEAD at execution; local script hashes identify exact external source. Concurrent Track B commits can appear in HEAD without changes to these scripts.'),indent=2));(P/'input_inventory.json').write_text(json.dumps(dict(files=files),indent=2));(P/'execution_receipts.json').write_text(json.dumps(dict(receipts=receipts),indent=2));print('HASHED',len(files),'inputs',len(sources),'sources',len(receipts),'receipts')

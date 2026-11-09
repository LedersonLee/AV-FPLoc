"""Read-only preservation and provenance audit; no evaluation or training."""
from pathlib import Path
import hashlib,json,subprocess
import numpy as np
from PIL import Image
P=Path(__file__).resolve().parent;W=P.parents[1];R=W/'AV-FPLoc';OLD=P.parent/'directional_tribranch_20261109'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
old=json.loads((OLD/'MANIFEST.json').read_text())['files']
for name,e in old.items():assert sha(OLD/name)==e['sha256'],name
pr=json.loads((P/'PROTOCOL.json').read_text());assert len(set(sum(pr['outer_folds'],[])))==12
assert set(sum(pr['outer_folds'],[]))==set(pr['training_scenes'])
assert not set(pr['training_scenes'])&set(pr['transfer_scenes'])
count=0
for receipt in (P/'candidates').glob('*/receipt.json'):
 for name,h in json.loads(receipt.read_text())['files'].items():assert sha(receipt.parent/name)==h;count+=1
raw=set()
for receipt in (P/'signals').glob('*/receipt.json'):
 r=json.loads(receipt.read_text());assert sha(receipt.parent/'features.npz')==r['features_sha256']
 for e in r['raw_records']:
  p=Path(e['path']);assert sha(p)==e['sha256'];assert sha(p.with_suffix('.json'))==e['metadata_sha256'];raw.add(str(p))
diff=subprocess.check_output(['git','-C',str(R),'diff','--binary','--','scripts/gibson_runtime/global_acoustic_likelihood_gpu.py','scripts/gibson_runtime/render_f3loc_echoscan_rirs.py'])
assert hashlib.sha256(diff).hexdigest()=='02ff3357f1cead0b70e2255c5f27eb15e325c161c6966ca1dbe06743fb1eabd3'
figs={}
for p in sorted((P/'figures').glob('*.png')):
 with Image.open(p) as im:im.verify()
 with Image.open(p) as im:figs[p.name]=list(im.size)
assert len(figs)==7
replay=json.loads((P/'model_replay_verification.json').read_text());assert replay['new_checkpoints']==73 and replay['numerical_checks']==4287 and replay['status']=='PASS'
models=[dict(path=str(p.relative_to(P)),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(P.rglob('*.joblib'))];assert len(models)==73
sources=[]
for p in sorted(P.glob('*.py')):
 q=R/'scripts/research/track_a/directional_structure_20261109'/p.name
 sources.append(dict(path=p.name,sha256=sha(p),repository_path=str(q.relative_to(R)),repository_matches=q.exists() and sha(q)==sha(p)))
(P/'artifact_verification.json').write_text(json.dumps(dict(status='PASS',prior_manifest_files=len(old),prior_manifest_sha256=sha(OLD/'MANIFEST.json'),candidate_cache_hashes=count,raw_unique_files=len(raw),original_runtime_diff_preserved=True,figure_dimensions=figs,new_model_count=len(models),model_replay_checks=replay['numerical_checks'],scope='Hashes/replay and explicit 23-scene allowlist; no claim of OS-wide access tracing. Test exclusion also inherited from preserved prior final_verification.json and scene protocol. TrackB is outside audited mutation scope.'),indent=2))
(P/'model_inventory.json').write_text(json.dumps(models,indent=2));(P/'source_inventory.json').write_text(json.dumps(sources,indent=2));print('PASS preserved',len(old),'priorfiles;',count,'candidate caches;',len(raw),'rawfiles;',len(models),'models;',len(figs),'figures')

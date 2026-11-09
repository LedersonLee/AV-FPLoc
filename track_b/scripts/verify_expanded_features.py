"""Check every feature hash/query alignment and exact pilot feature overlap."""
import argparse,json,hashlib
from pathlib import Path
import torch
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser()
 for k in ['features','cache','pilot','out']:p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();torch.set_num_threads(2);manifest=read(a.features/'manifest.json');protocol=read(a.features/'protocol.json');data=read(a.cache/'protocol.json');old=read(a.pilot/'protocol.json')
 assert len(manifest)==131 and len({r['scene'] for r in manifest})==131
 assert protocol['data']==data and protocol['selected_scenes']==data['scenes'] and not protocol['smoke']
 for k in ['checkpoint_sha256','source_hashes','preprocessing','features']:assert protocol[k]==old[k],k
 # Original normal-cache schema predates explicit augmentation fields.
 assert protocol.get('brightness',1)==old.get('brightness',1)==1
 assert protocol.get('center_occlusion',0)==old.get('center_occlusion',0)==0
 assert {r['scene'] for r in manifest}==set(data['scenes']['train']+data['scenes']['val'])
 a.out.mkdir(parents=True,exist_ok=False);rows=[]
 pilot_ids=set(old['data']['scenes']['train']+old['data']['scenes']['val'])
 for r in manifest:
  sid=r['scene'];path=a.features/f'{sid}.pt';assert sha(path)==r['sha256'];pack=torch.load(path,weights_only=True,map_location='cpu');inputs=read(a.cache/f'{sid}_inputs.json')
  assert r['input_manifest_sha256']==sha(a.cache/f'{sid}_inputs.json')
  assert pack['query_ids']==[x['query_id'] for x in inputs] and pack['scene']==sid and len(pack['query_ids'])==64
  assert pack['visual_features'].shape==(64,1024,4,16) and torch.isfinite(pack['visual_features']).all()
  if sid in pilot_ids:
   prior=torch.load(a.pilot/f'{sid}.pt',weights_only=True,map_location='cpu');assert prior['query_ids']==pack['query_ids'] and torch.equal(prior['visual_features'],pack['visual_features']),f'Pilot feature mismatch: {sid}'
  rows.append(dict(scene=sid,sha256=r['sha256'],queries=64,pilot_equal=sid in pilot_ids))
 assert sum(r['pilot_equal'] for r in rows)==22
 result=dict(scenes=131,queries=8384,pilot_exact_scenes=22,feature_bytes=sum(r['cache_bytes'] for r in manifest),rows=rows,feature_protocol_sha256=sha(a.features/'protocol.json'),test_accessed=False)
 (a.out/'validation.json').write_text(json.dumps(result,indent=2));print('COMPLETE')
if __name__=='__main__':main()

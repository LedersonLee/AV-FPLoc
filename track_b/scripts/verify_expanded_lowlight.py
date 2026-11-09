"""Verify expanded brightness controls and exact overlap with the pilot."""
import argparse,hashlib,json
from pathlib import Path
import torch

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser()
 for k in ['features','cache','pilot','out']:p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();new=read(a.features/'protocol.json');old=read(a.pilot/'protocol.json');data=read(a.cache/'protocol.json');ids=data['scenes']['val']
 assert len(ids)==19 and new['data']==data and new['selected_scenes']=={'val':ids} and not new['smoke']
 assert new['brightness']==old['brightness']==.2 and new.get('center_occlusion',0)==old.get('center_occlusion',0)==0
 for key in ['checkpoint_sha256','source_hashes','preprocessing','features']:assert new[key]==old[key],key
 manifest=read(a.features/'manifest.json');assert len(manifest)==19 and {r['scene'] for r in manifest}==set(ids)
 pilot={r['scene']:r for r in read(a.pilot/'manifest.json')};assert len(pilot)==6 and set(pilot)<=set(ids)
 rows=[]
 for row in manifest:
  sid=row['scene'];path=a.features/f'{sid}.pt';assert sha(path)==row['sha256'] and row['input_manifest_sha256']==sha(a.cache/f'{sid}_inputs.json')
  pack=torch.load(path,weights_only=True,map_location='cpu');assert pack['query_ids']==[q['query_id'] for q in read(a.cache/f'{sid}_inputs.json')]
  assert pack['visual_features'].shape==(64,1024,4,16) and torch.isfinite(pack['visual_features']).all()
  if sid in pilot:
   pp=a.pilot/f'{sid}.pt';assert sha(pp)==pilot[sid]['sha256'];prior=torch.load(pp,weights_only=True,map_location='cpu');assert pack['query_ids']==prior['query_ids'] and torch.equal(pack['visual_features'],prior['visual_features'])
  rows.append(dict(scene=sid,feature_sha256=row['sha256'],pilot_exact=sid in pilot))
 a.out.mkdir(parents=True,exist_ok=False);(a.out/'validation.json').write_text(json.dumps(dict(scenes=19,queries=1216,pilot_exact_scenes=6,feature_protocol_sha256=sha(a.features/'protocol.json'),rows=rows,test_accessed=False),indent=2));print('COMPLETE',flush=True)
if __name__=='__main__':main()

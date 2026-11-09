"""Require complete expanded cache, frozen selection, hashes and pilot equality."""
import argparse,json,hashlib,math
from pathlib import Path
import numpy as np
import torch

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser()
 for k in ['cache','pilot','preflight','root','out']:p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();torch.set_num_threads(2);manifest=read(a.cache/'manifest.json');preflight=read(a.preflight/'preflight.json');protocol=read(a.cache/'protocol.json');pilot=read(a.pilot/'protocol.json')
 expected={r['scene']:r for r in preflight['rows']}
 assert len(manifest)==len(expected)==131 and {r['scene'] for r in manifest}==set(expected),'Await complete unique131-scene cache'
 assert len({r['scene'] for r in manifest})==131
 assert {k:len(v) for k,v in protocol['scenes'].items()}=={'train':112,'val':19}
 assert protocol['split_source_sha256']==preflight['source_split_sha256']
 assert not {s.split('_')[0] for s in protocol['scenes']['train']}&{s.split('_')[0] for s in protocol['scenes']['val']}
 assert sum(r['queries'] for r in manifest)==8384
 a.out.mkdir(parents=True,exist_ok=False);rows=[];overlap=[]
 for row in manifest:
  sid=row['scene'];part=row['split'];e=expected[sid]
  assert e['split']==part and sid in protocol['scenes'][part] and row['queries']==64 and row['candidates']==e['candidates']
  path=a.cache/f'{sid}.pt';assert sha(path)==row['cache_sha256']
  schedule_path=a.root/f'RIR/manifests/positive_pairs_v1/schedules/{part}/{sid}.json'
  assert sha(schedule_path)==row['schedule_sha256']==e['schedule_sha256']
  assert sha(a.root/f'2DFP/{sid}/map.png')==row['map_sha256']==e['map_sha256']
  assert sha(a.root/f'2DFP/{sid}/floorplan_metadata.json')==e['metadata_sha256']
  schedule=read(schedule_path)['queries'];chosen=[schedule[i] for i in np.linspace(0,len(schedule)-1,64).round().astype(int)]
  s=torch.load(path,weights_only=True,map_location='cpu');inputs=read(a.cache/f'{sid}_inputs.json')
  assert s['query_ids']==[q['sample_index'] for q in chosen]==[q['query_id'] for q in inputs]
  assert len(set(s['query_ids']))==64 and len(inputs)==64 and s['scene']==sid and s['split']==part
  for q,source in zip(chosen,inputs):
   assert q['observed_rir_path']==source['rir_path'] and q['image_path']==source['rgb_path']
   assert source['rir_path'].startswith('RIR/Observed/')
   side=read((a.root/source['rir_path']).with_suffix('.json'))
   assert side['role']=='Observed' and side['rir_sha256']==source['rir_sha256']
  assert s['spectrum'].shape==(64,18,44,513) and s['rgb'].shape==(64,3,96,128)
  assert s['candidates'].shape==(row['candidates'],35)
  assert all(torch.isfinite(v).all() for v in s.values() if isinstance(v,torch.Tensor))
  expected_gt=torch.tensor([[q['centre_xyz_m'][0],q['centre_xyz_m'][2],math.radians(q['world_yaw_deg'])] for q in chosen],dtype=torch.float32)
  assert torch.equal(s['gt'],expected_gt)
  if sid in pilot['scenes'][part]:
   old=torch.load(a.pilot/f'{sid}.pt',weights_only=True,map_location='cpu')
   assert old.keys()==s.keys() and all(torch.equal(v,s[k]) if isinstance(v,torch.Tensor) else v==s[k] for k,v in old.items()),f'Pilot tensor mismatch {sid}'
   assert read(a.pilot/f'{sid}_inputs.json')==inputs
   overlap.append(sid)
  rows.append(dict(scene=sid,split=part,queries=64,candidates=len(s['candidates']),cache_sha256=row['cache_sha256'],input_manifest_sha256=sha(a.cache/f'{sid}_inputs.json'),pilot_equal=sid in overlap))
  print('VERIFIED',sid,flush=True)
 assert len(overlap)==22
 result=dict(scenes=131,queries=8384,pilot_overlap_scenes=len(overlap),pilot_overlap_queries=len(overlap)*64,pilot_tensor_and_input_manifests_exact=True,actual_cache_bytes=sum(r['cache_bytes'] for r in manifest),rows=rows,source_protocol_sha256=sha(a.cache/'protocol.json'),test_accessed=False,scope='all cached tensors/source metadata/selection checked; observed-RIR sidecar hashes matched cached input provenance; raw audio/RGB bytes were checked during construction, not reread here; no scientific accuracy result')
 (a.out/'validation.json').write_text(json.dumps(result,indent=2));print('COMPLETE',flush=True)
if __name__=='__main__':main()

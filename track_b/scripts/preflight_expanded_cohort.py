"""Freeze expanded development membership/resources without Test observations."""
import argparse,hashlib,json,shutil
from pathlib import Path
import numpy as np
from PIL import Image
from track_b.data.geometry import candidate_bank

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser()
 for name in ['root','audit','pilot','out']:p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 split=read(a.audit/'split_manifest.json')['development'];assert len(split['train'])==112 and len(split['val'])==19
 assert not {s.split('_')[0] for s in split['train']}&{s.split('_')[0] for s in split['val']}
 pilot=read(a.pilot/'protocol.json');rows=[]
 for part in ['train','val']:
  for sid in sorted(split[part],key=lambda s:hashlib.sha256(s.encode()).hexdigest()):
   sch_path=a.root/f'RIR/manifests/positive_pairs_v1/schedules/{part}/{sid}.json';sch=read(sch_path)
   meta_path=a.root/f'2DFP/{sid}/floorplan_metadata.json';meta=read(meta_path);map_path=a.root/f'2DFP/{sid}/map.png'
   free=np.asarray(Image.open(map_path))==255;origin=[meta['world_bounds_m'][k] for k in ['x_min','z_min']]
   candidates,valid=candidate_bank(free,meta['map_spacing_m'],origin,.25,36)
   selected=[sch['queries'][i] for i in np.linspace(0,len(sch['queries'])-1,min(64,len(sch['queries']))).round().astype(int)]
   assert len(selected)==64 and len({q['sample_index'] for q in selected})==64
   for q in selected:assert (a.root/q['observed_rir_path']).is_file() and (a.root/q['image_path']).is_file()
   estimate=64*(18*44*513*2+3*128*96+3*4+16*4+16)+len(candidates)*35*4+valid.nbytes+free.size*8
   rows.append(dict(scene=sid,split=part,total_schedule_queries=len(sch['queries']),selected_queries=64,in_pilot=sid in pilot['scenes'][part],candidates=len(candidates),estimated_cache_bytes_upperish=int(estimate),map_sha256=sha(map_path),metadata_sha256=sha(meta_path),schedule_sha256=sha(sch_path)))
 estimate=sum(x['estimated_cache_bytes_upperish'] for x in rows);freebytes=shutil.disk_usage(a.out).free
 assert estimate*2+10*1024**3<freebytes,'Insufficient headroom for separate caches and preservation'
 result=dict(rows=rows,scenes={k:len(v) for k,v in split.items()},selected_query_count=len(rows)*64,selected_train_queries=112*64,selected_val_queries=19*64,new_scenes={part:len(set(split[part])-set(pilot['scenes'][part])) for part in split},estimated_cache_bytes=estimate,available_bytes=freebytes,source_split_sha256=sha(a.audit/'split_manifest.json'),scope='all eligible audited development scenes; fixed64 queries/scene, not all available queries; no Test observations; expanded Val has historical upstream/research exposure and is not independent',test_accessed=False)
 (a.out/'preflight.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='rows'}));print('COMPLETE')
if __name__=='__main__':main()

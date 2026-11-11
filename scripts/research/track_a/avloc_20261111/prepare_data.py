"""Materialize audited full-resolution training inputs on SSD; never renders RIRs.
May prepare finished audit scenes incrementally, but emits READY only after full audit.
"""
import argparse,hashlib,json,shutil,time
from pathlib import Path
import numpy as np
from PIL import Image
from geometry import cast_rays

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--audit',type=Path,required=True);ap.add_argument('--root',type=Path,required=True);ap.add_argument('--ssd-root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True);split=read(a.audit/'split.json');records=[]
 for part,scenes in split['requested'].items():
  for sid in scenes:
   source=a.audit/f'{sid}.json'
   if not source.exists():continue
   audited=read(source);summary=audited['summary'];queries=[q for q in audited['queries'] if not q['issues']];dest=a.out/sid;dest.mkdir(exist_ok=True);receipt=dest/'receipt.json'
   if receipt.exists():
    rec=read(receipt);assert rec['audit_sha256']==sha(source);records.append(rec);continue
   assert queries,('No usable queries',sid);start=time.time();meta=read(a.root/f'2DFP/{sid}/floorplan_metadata.json');mapfile=a.root/f'2DFP/{sid}/map.png';assert sha(mapfile)==summary['map_sha256'];free=np.asarray(Image.open(mapfile).convert('L'))==255;origin=[meta['world_bounds_m'][k] for k in ['x_min','z_min']]
   poses=np.array([q['pose'] for q in queries]);targets,known=cast_rays(free,meta['map_spacing_m'],origin,poses);assert known.all(),('Unbounded raster exits need review',sid,int((~known).sum()))
   waves=np.lib.format.open_memmap(dest/'rir.npy',mode='w+',dtype='float32',shape=(len(queries),6,4800));rgb_dir=dest/'rgb';rgb_dir.mkdir(exist_ok=True);paths=[];loaded={};mirror_used=0
   for i,q in enumerate(queries):
    # Prefer existing SSD RGB only after matching the authoritative audit's hash.
    imsrc=a.ssd_root/q['rgb']
    if imsrc.exists() and sha(imsrc)==q['rgb_sha256']:mirror_used+=1
    else:imsrc=a.root/q['rgb'];assert sha(imsrc)==q['rgb_sha256']
    imdst=rgb_dir/f"{q['query_id']}.png";shutil.copyfile(imsrc,imdst);assert sha(imdst)==q['rgb_sha256'];paths.append(str(imdst.relative_to(a.out)))
    if q['rir'] not in loaded:
     rp=a.root/q['rir'];assert sha(rp)==q['rir_info']['sha256'];raw=np.load(rp,allow_pickle=False).astype('float32');loaded[q['rir']]=np.pad(raw[:,:4800],((0,0),(0,max(0,4800-raw.shape[1]))))
    waves[i]=np.roll(loaded[q['rir']],q['channel_roll'],axis=0)
   waves.flush();del waves
   np.savez(dest/'labels.npz',target=targets,known=known,truth=poses,query_ids=np.array([q['query_id'] for q in queries]))
   (dest/'images.json').write_text(json.dumps(paths)+'\n');rec=dict(scene=sid,split=part,queries=len(queries),invalid_queries=summary['queries']-len(queries),source_queries=summary['queries'],audit_sha256=sha(source),map_sha256=sha(mapfile),map_spacing=meta['map_spacing_m'],target_max_m=float(targets.max()),target_mean_m=float(targets.mean()),ssd_rgb_verified=mirror_used,ray_policy='exact binary-pixel boundary DDA; uncapped; raster exits rejected',input_policy='original640x480 RGB bytes; body-aligned float32 RIR first100ms, zero-tail padded, direct path retained; chirp on training/evaluation path',files={name:sha(dest/name) for name in ['rir.npy','labels.npz','images.json']},seconds=time.time()-start)
   receipt.write_text(json.dumps(rec,indent=2)+'\n');records.append(rec);print('PREPARED',sid,len(queries),round(time.time()-start,1),flush=True)
 ready=(a.audit/'summary.json').exists() and len(records)==sum(map(len,split['requested'].values()))
 manifest=dict(ready=ready,records=records,expected_scenes=sum(map(len,split['requested'].values())),split_sha256=sha(a.audit/'split.json'),test_accessed=False)
 (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print('READY',ready,'SCENES',len(records),flush=True)
if __name__=='__main__':main()

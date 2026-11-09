"""Exact geometry duplicates in selected development maps only."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(mask,spacing):
 y,x=np.where(mask);assert len(x),'Empty footprint'
 cropped=mask[y.min():y.max()+1,x.min():x.max()+1]
 hashes=[]
 for reflected in [False,True]:
  z=cropped[:,::-1] if reflected else cropped
  for k in range(4):
   a=np.ascontiguousarray(np.rot90(z,k));header=json.dumps(dict(shape=list(a.shape),spacing=float(spacing)),sort_keys=True).encode()
   hashes.append(hashlib.sha256(header+np.packbits(a).tobytes()).hexdigest())
 return min(hashes),list(cropped.shape)
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--code-sha',required=True);a=p.parse_args()
 # Known transformed masks verify translation, rotation, reflection invariance;
 # pixel and physical-scale changes must remain distinct.
 toy=np.array([[1,1,0],[0,1,0],[0,1,1]],dtype=bool);key,_=canonical(toy,.1)
 assert canonical(np.pad(toy,((2,4),(3,1))),.1)[0]==key
 assert all(canonical(np.rot90(toy,k),.1)[0]==key and canonical(np.rot90(toy[:,::-1],k),.1)[0]==key for k in range(4))
 changed=toy.copy();changed[0,2]=True;assert canonical(changed,.1)[0]!=key and canonical(toy,.2)[0]!=key
 protocol=read(a.cache/'protocol.json');manifest={r['scene']:r for r in read(a.cache/'manifest.json')};rows=[]
 for split,scenes in protocol['scenes'].items():
  assert split in ['train','val']
  for sid in scenes:
   path=a.root/'2DFP'/sid/'map.png';assert sha(path)==manifest[sid]['map_sha256'];meta=read(path.parent/'floorplan_metadata.json');mask=np.asarray(Image.open(path))==255;assert mask.ndim==2
   digest,shape=canonical(mask,meta['map_spacing_m']);rows.append(dict(scene=sid,split=split,map_sha256=sha(path),geometry_sha256=digest,spacing=meta['map_spacing_m'],cropped_shape=shape,free_pixels=int(mask.sum())))
 groups={}
 for row in rows:groups.setdefault(row['geometry_sha256'],[]).append(dict(scene=row['scene'],split=row['split']))
 duplicates=[g for g in groups.values() if len(g)>1];cross=[g for g in duplicates if len({x['split'] for x in g})>1]
 a.out.mkdir(parents=True,exist_ok=False)
 result=dict(code_sha=a.code_sha,cache_protocol_sha256=sha(a.cache/'protocol.json'),scenes=len(rows),duplicate_groups=duplicates,cross_split_groups=cross,transform_test_passed=True,scope='Exact white-pixel footprint, tight bounding box, same physical pixel spacing, rotations by90deg and reflection; ignores world translation. Does not detect approximate/resampled maps or prove source/upstream independence.',test_accessed=False,rows=rows)
 (a.out/'audit.json').write_text(json.dumps(result,indent=2));print('COMPLETE',len(rows),'duplicate groups',len(duplicates),'cross-split',len(cross),flush=True)
if __name__=='__main__':main()

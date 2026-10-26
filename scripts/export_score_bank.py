"""Export a bounded subset of a trusted historical scene bank into portable NPZ.

Reads scores.npy [Q,2,C,36], xy.npy, queries.json and optionally truth.npy.
No fitting or candidate filtering. Explicit paths replace machine-specific roots.
"""
import argparse
import json
from pathlib import Path
import hashlib
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('cache','depth','desdf','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--scene',required=True);p.add_argument('--count',type=int,default=3)
    p.add_argument('--include-truth',action='store_true');a=p.parse_args()
    if a.count<1 or a.output.exists():raise ValueError('Positive count and new output directory required')
    s=np.load(a.cache/'scores.npy',mmap_mode='r');xy=np.load(a.cache/'xy.npy')
    queries=json.loads((a.cache/'queries.json').read_text())[:a.count]
    if len(queries)!=a.count or s.shape[1:]!=(2,len(xy),36):raise ValueError('Invalid bank/count')
    with np.load(a.depth,allow_pickle=False) as f:
        lut={int(v):i for i,v in enumerate(f['sample_indices'])};ids=[lut[q['sample_index']] for q in queries]
        depth=f['location'][ids];scale=f['scale'][ids]
    # Historical DESDF format is a trusted local pickled dictionary.
    field=np.load(a.desdf,allow_pickle=True).item()
    if field['desdf_spacing_m']!=.1 or field['max_range_m']!=10:raise ValueError('Historical ray contract mismatch')
    origin=np.array([field['origin_x_m'],field['origin_z_min_m']]);ij=np.rint((xy-origin)/.1).astype(int)
    np.testing.assert_allclose(origin+ij*.1,xy,atol=1e-6,rtol=0)
    if (ij<0).any() or (ij[:,0]>=field['desdf'].shape[1]).any() or (ij[:,1]>=field['desdf'].shape[0]).any():raise ValueError('Grid outside DESDF')
    rays=field['desdf'][ij[:,1],ij[:,0]]
    qids=np.array([f"{a.scene}:{q['sample_index']}" for q in queries]);a.output.mkdir(parents=True)
    shared=dict(xy=xy,query_ids=qids,yaw_deg=np.arange(36)*10)
    np.savez_compressed(a.output/'vision.npz',scores=np.asarray(s[:a.count,0]),depth=depth,scale=scale,ray_bank=rays,**shared)
    np.savez_compressed(a.output/'acoustic.npz',scores=np.asarray(s[:a.count,1]),**shared)
    if a.include_truth:np.savez_compressed(a.output/'truth.npz',poses=np.load(a.cache/'truth.npy',mmap_mode='r')[:a.count],query_ids=qids)
    hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in a.output.glob('*.npz')}
    (a.output/'receipt.json').write_text(json.dumps(dict(scene=a.scene,query_ids=qids.tolist(),source_cache=str(a.cache),files=hashes,normalization='historical per-query z-scores; no new model fit'),indent=2)+'\n')


if __name__=='__main__':main()

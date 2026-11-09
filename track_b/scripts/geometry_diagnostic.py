"""Quantify additional geometric distinctions, not localization accuracy."""
import argparse,json,time
from pathlib import Path
import numpy as np
from PIL import Image
from track_b.data.geometry import candidate_bank
from track_b.data.rays import boundary_rays


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--pilot',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    protocol=json.loads((args.pilot/'protocol.json').read_text());results=[]
    for split in ('train','val'):
      for scene in protocol[split]:
        meta=json.loads((args.root/f'2DFP/{scene}/floorplan_metadata.json').read_text());free=np.array(Image.open(args.root/f'2DFP/{scene}/map.png'))==255
        origin=[meta['world_bounds_m'][k] for k in ('x_min','z_min')];p,_=candidate_bank(free,meta['map_spacing_m'],origin,.5,12)
        start=time.perf_counter();d,h=boundary_rays(free,meta['map_spacing_m'],origin,p[::12,:2],12)
        signatures=np.column_stack([np.round(d,3),h]);u,counts=np.unique(signatures,axis=0,return_counts=True)
        results.append(dict(scene=scene,split=split,candidates=len(p),unique_boundary_ray_signatures=len(u),largest_equal_group=int(counts.max()),seconds=time.perf_counter()-start,
                            cache_bytes=d.nbytes+h.nbytes,boundary_hit_fraction=float(h.mean()),range_step_m=.1,max_range_m=20))
    args.out.write_text(json.dumps(dict(results=results,interpretation='FP-only representation diagnostic. Does not establish usefulness of audio or localization generalization.'),indent=2));print(json.dumps(results))

if __name__=='__main__':main()

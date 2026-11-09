"""Measured full 0.25m/36 inference on the largest audited development map."""
import argparse,json,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from track_b.data.geometry import candidate_bank
from track_b.models.dense import DenseScorer,rasterize
from track_b.scripts.pilot import materialize


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--audit',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    torch.set_num_threads(2);torch.manual_seed(42)
    qc=[json.loads(l) for l in (args.audit/'dataset_qc.jsonl').read_text().splitlines()]
    allowed=set(sum(json.loads((args.audit/'split_manifest.json').read_text())['development'].values(),[]))
    row=max((r for r in qc if r.get('scene_id') in allowed),key=lambda r:r['bank_candidates'])
    s=materialize(args.root,row['scene_id'],row['split'],1)
    meta=json.loads((args.root/f"2DFP/{row['scene_id']}/floorplan_metadata.json").read_text());free=np.array(Image.open(args.root/f"2DFP/{row['scene_id']}/map.png"))==255
    origin=[meta['world_bounds_m'][k] for k in ('x_min','z_min')];p,valid=candidate_bank(free,meta['map_spacing_m'],origin,.25,36);p=torch.tensor(p,dtype=torch.float32).cuda()
    model=DenseScorer().cuda().eval();rgb,rir,_,_=s['data'][0]
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for _ in range(3):score=model(rgb,rir,s['fp'],s['cal'],p)
        torch.cuda.synchronize();start=time.perf_counter()
        for _ in range(5):score=model(rgb,rir,s['fp'],s['cal'],p)
        torch.cuda.synchronize();ms=(time.perf_counter()-start)*200
        field=rasterize(score,torch.tensor(valid,device='cuda'),36)
        assert score.shape==(1,len(p)) and torch.isfinite(score).all()
        assert torch.isneginf(field[:,~torch.tensor(valid,device='cuda')]).all()
    result=dict(scene=row['scene_id'],split=row['split'],initialization='random seed42; no accuracy claim',grid=.25,yaw_bins=36,candidates=len(p),field_shape=list(field.shape),
                valid_field_complete=True,invalid_field_negative_infinity=True,warm_ms=ms,peak_vram_bytes=torch.cuda.max_memory_allocated(),
                scope='FP/RGB/RIR encoding + full scorer; excludes I/O and preprocessing and topK; shared GPU timing, not final cost comparison',training_full_resolution_tested=False)
    args.out.write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()

"""Frozen-checkpoint fit and audio-window diagnostics, without model selection."""
import argparse,json,math
from pathlib import Path
import numpy as np
import torch
from track_b.models.dense import DenseScorer
from track_b.models.ray_dense import RayDenseScorer
from track_b.scripts.pilot import materialize as local_data
from track_b.scripts.pilot_geometry import materialize as ray_data


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--pilot',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    torch.set_num_threads(2);protocol=json.loads((args.pilot/'protocol.json').read_text());geometry=protocol.get('geometry');loader=ray_data if geometry else local_data;rows=[]
    models={}
    for seed in protocol['seeds']:
      for mode in ('acoustic','joint'):
        model=(RayDenseScorer(mode=mode,geometry=geometry) if geometry else DenseScorer(mode=mode)).cuda().eval()
        model.load_state_dict(torch.load(args.pilot/f'{mode}_{seed}.pt',weights_only=True));models[(seed,mode)]=model
    with torch.no_grad():
      for split in ('train','val'):
        for sid in protocol[split]:
          scene=loader(args.root,sid,split,16);p=scene['p'];score_p=scene.get('score_p',p)
          for seed in protocol['seeds']:
            for mode in ('acoustic','joint'):
              model=models[(seed,mode)]
              for rgb,rir,gt,qid in scene['data']:
                interventions=('normal',) if split=='train' else ('normal','direct_only','direct_masked','late_only')
                for arm in interventions:
                    audio=rir.clone()
                    if arm=='direct_only':audio[:,:,32:]=0
                    elif arm=='direct_masked':audio[:,:,:32]=0
                    elif arm=='late_only':audio[:,:,:7200]=0
                    s=model(rgb,audio,scene['fp'],scene['cal'],score_p)[0];d=p-gt;dist=d[:,:2].norm(dim=-1);angle=torch.atan2(d[:,2].sin(),d[:,2].cos()).abs();ok=(dist<=1)&(angle<=math.pi/6+1e-6);idx=s.argmax()
                    rows.append(dict(split=split,scene=sid,seed=seed,mode=mode,arm=arm,query=qid,joint=float(ok[idx]),r1=float(dist[idx]<=1),raw20=float(ok[s.topk(20).indices].any())))
    summary=[]
    for split in ('train','val'):
      for mode in ('acoustic','joint'):
        for arm in ('normal','direct_only','direct_masked','late_only'):
          rr=[r for r in rows if (r['split'],r['mode'],r['arm'])==(split,mode,arm)]
          if rr:summary.append(dict(split=split,mode=mode,arm=arm,**{k:float(np.mean([r[k] for r in rr])) for k in ('joint','r1','raw20')}))
    args.out.write_text(json.dumps(dict(summary=summary,rows=rows,limits='Frozen intervention, no re-normalization after masking. Direct-only samples0:32; late-only7200:8192 (150–170.7ms). Timing/gain shift may confound physical interpretation.'),indent=2));print(json.dumps(summary))

if __name__=='__main__':main()

"""Correct diagnostic ray errors under interventions; preserve original run files."""
import argparse,json,math
from pathlib import Path
import numpy as np
import torch
from track_b.models.structure import StructureMatcher,visible_rays


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--run',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();torch.set_num_threads(2)
    protocol=json.loads((a.run/'protocol.json').read_text());models={};old={}
    for seed in protocol['seeds']:
      for mode in protocol['modes']:
        m=StructureMatcher().cuda().eval();m.load_state_dict(torch.load(a.run/f'{mode}_{seed}.pt',weights_only=True));models[seed,mode]=m
        old[seed,mode]={(r['scene'],r['query'],r['arm']):r for r in json.loads((a.run/f'{mode}_{seed}.json').read_text())['val']['rows']}
    rows=[];normal_errors=[];changes=0
    with torch.no_grad():
      for sid in protocol['data']['scenes']['val']:
        s=torch.load(a.cache/f'{sid}.pt',weights_only=True);rgb=s['rgb'].cuda().float()/255;spec=s['spectrum'].cuda().float();target=s['target'].cuda();n=len(target)
        for (seed,mode),model in models.items():
            mask=visible_rays(device='cuda') if mode=='visual' else torch.ones(16,device='cuda')
            for ids in torch.arange(n,device='cuda').split(8):
              for arm in ('normal','audio_shuffle','audio_zero','rgb_zero'):
                if mode=='visual':pred=model.visual(torch.zeros_like(rgb[ids]) if arm=='rgb_zero' else rgb[ids])
                else:pred=model.acoustic(torch.zeros_like(spec[ids]) if arm=='audio_zero' else spec[(ids+1)%n] if arm=='audio_shuffle' else spec[ids])
                mse=((pred-target[ids]).square()*mask).sum(-1)/mask.sum();mae=((torch.expm1(pred*math.log(21))-torch.expm1(target[ids]*math.log(21))).abs()*mask).sum(-1)/mask.sum()
                for j,q in enumerate(ids.tolist()):
                    row=dict(scene=sid,query=s['query_ids'][q],seed=seed,mode=mode,arm=arm,ray_mse=float(mse[j]),ray_mae_m=float(mae[j]));before=old[seed,mode][sid,s['query_ids'][q],arm]
                    if arm=='normal' and not np.isclose(row['ray_mse'],before['ray_mse'],atol=1e-6):normal_errors.append(row)
                    if not np.isclose(row['ray_mse'],before['ray_mse'],atol=1e-6):changes+=1
                    rows.append(row)
    summary=[]
    for mode in protocol['modes']:
      for arm in ('normal','audio_shuffle','audio_zero','rgb_zero'):
        rr=[r for r in rows if (r['mode'],r['arm'])==(mode,arm)];summary.append(dict(mode=mode,arm=arm,**{k:float(np.mean([r[k] for r in rr])) for k in ('ray_mse','ray_mae_m')}))
    out=dict(summary=summary,rows=rows,changed_rows=changes,normal_mismatches=normal_errors,
             correction='Original supervised_v1 non-normal ray error columns referenced normal predictions. Pose metrics and normal ray errors unaffected; originals preserved.')
    a.out.write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k!='rows'}))

if __name__=='__main__':main()

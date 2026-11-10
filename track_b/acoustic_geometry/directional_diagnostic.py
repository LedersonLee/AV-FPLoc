"""Distinguish pose-dependent scalar range from directional shape evidence."""
import argparse
import json
from pathlib import Path
import numpy as np
from .prepare_targets import read
from .summarize import paired


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--code-sha',required=True);a=p.parse_args();a.out.mkdir(exist_ok=False,parents=True)
    config=read(a.run/'protocol.json');groups={};rows=[]
    for record in read(a.run/'registry.json'):
      for scene in config['scene_ids']['val']:
        z=np.load(a.run/record['run']/f'{scene}__normal.npz');m=z['mean'];y=z['target'];mask=z['mask']
        normal=abs(m-y)[mask].mean()
        scalar=np.broadcast_to(m.mean(-1,keepdims=True),m.shape)
        corr=[]
        for i in range(len(m)):
            good=mask[i]
            if good.sum()>2 and m[i,good].std()>1e-7 and y[i,good].std()>1e-7:
                corr.append(float(np.corrcoef(m[i,good],y[i,good])[0,1]))
        r=dict(scene=scene,run=record['run'],seed=record['seed'],architecture=record['architecture'],head=record['head'],
               scalarized_minus_normal_mae=float(abs(scalar-y)[mask].mean()-normal),
               rotated180_minus_normal_mae=float(abs(np.roll(m,18,-1)-y)[mask].mean()-normal),
               prediction_angular_std_m=float(m.std(-1).mean()),
               within_query_direction_correlation=float(np.mean(corr)) if corr else None)
        rows.append(r)
        for k in ['scalarized_minus_normal_mae','rotated180_minus_normal_mae','prediction_angular_std_m','within_query_direction_correlation']:
            if r[k] is not None:groups.setdefault((record['architecture'],record['head'],k),{}).setdefault(scene,[]).append(r[k])
    summary=[dict(architecture=k[0],head=k[1],metric=k[2],**paired({s:np.mean(v) for s,v in values.items()})) for k,values in groups.items()]
    (a.out/'diagnostic.json').write_text(json.dumps(dict(code_sha=a.code_sha,selection='post-training diagnostic prompted by predefined and worst-case qualitative examples; no model modification or test access',summary=summary,rows=rows),indent=2))
    for r in summary:print(json.dumps(r),flush=True)


if __name__=='__main__':main()

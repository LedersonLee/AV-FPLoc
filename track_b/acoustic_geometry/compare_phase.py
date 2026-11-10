"""Matched phase removal contrasts; no tuning or new evaluation observations."""
import argparse
import json
from pathlib import Path
import numpy as np
from .prepare_targets import read
from .summarize import paired


def main():
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--code-sha',required=True)
    a=p.parse_args();a.out.mkdir(exist_ok=False,parents=True)
    roots=[a.study/'representation_v1',a.study/'phase_zero_v1'];configs=[read(r/'protocol.json') for r in roots]
    for key in ['steps','seeds','heads','scene_ids','features_manifest_sha256','targets_manifest_sha256','lr','weight_decay','batch_size']:
        assert configs[0][key]==configs[1][key]
    assert not configs[0]['phase_zero'] and configs[1]['phase_zero'];groups={}
    for head in configs[0]['heads']:
      for seed in configs[0]['seeds']:
        arm=f'spatial_{head}_{seed}'
        rows=[{(x['scene'],x['condition']):x for x in read(r/arm/'metrics.json')} for r in roots]
        assert rows[0].keys()==rows[1].keys()
        for (scene,condition),normal in rows[0].items():
            ablated=rows[1][(scene,condition)]
            for metric in ['mae','rmse','direction_correlation','categorical_nll','crps_m']:
                if normal.get(metric) is not None:
                    groups.setdefault((head,condition,metric),{}).setdefault(scene,[]).append(ablated[metric]-normal[metric])
    summary=[dict(head=k[0],condition=k[1],metric=k[2],contrast='phase_zero_minus_full',**paired({s:np.mean(v) for s,v in d.items()})) for k,d in groups.items()]
    (a.out/'summary.json').write_text(json.dumps(dict(code_sha=a.code_sha,validation='same IDs, seeds, optimizer settings, update count, data and GT hashes; only declared phase-zero intervention differs',summary=summary),indent=2))


if __name__=='__main__':main()

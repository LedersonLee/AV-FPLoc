"""Verify CPU physical-pair evaluator against stored primary GPU predictions."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .models import GeometryNet
from .prepare_targets import read,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--code-sha',required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    pairs=a.study/'pair_evaluation_openings_v1';result=read(pairs/'summary.json');scenes=sorted({r['scene'] for r in result['per_scene']});rows=[]
    for r in read(a.study/'representation_v1/registry.json'):
        cp=a.study/'representation_v1'/r['run']/'model.pt';assert sha(cp)==r['checkpoint_sha256']
        model=GeometryNet(r['architecture'],r['head']);model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=False)['model']);model.eval()
        for scene in scenes:
            package=torch.load(pairs/f'{scene}_features.pt',map_location='cpu',weights_only=False)
            original_features=torch.load(a.study/'features_v1'/f'{scene}.pt',map_location='cpu',weights_only=False)
            assert torch.equal(package['conditions']['historical_furnished'][r['architecture']],original_features[r['architecture']][package['indices']]),'Feature mismatch'
            reference=np.load(cp.parent/f'{scene}__normal.npz');indices=package['indices']
            with torch.no_grad():mean,logits=model(package['conditions']['historical_furnished'][r['architecture']].float())
            difference=float(np.abs(mean.numpy()-reference['mean'][indices]).max())
            assert np.isfinite(difference)
            probdiff=None
            if logits is not None:
                probdiff=float(np.abs(logits.softmax(-1).numpy()-reference['probability'][indices]).max());assert np.isfinite(probdiff)
            rows.append(dict(run=r['run'],scene=scene,queries=len(indices),maximum_depth_difference_m=difference,maximum_probability_difference=probdiff))
    output=dict(code_sha=a.code_sha,scope='all12 primary models,16 paired-development families,8 selected poses each; same historical furnished RIR',checks=len(rows),features_bitwise_equal=True,numerical_interpretation='Measured CPU versus original GPU differences; no bitwise prediction parity claim; initial1e-4m gate failed and was retained as pair_parity_v1',maximum_depth_difference_m=max(r['maximum_depth_difference_m'] for r in rows),maximum_probability_difference=max(r['maximum_probability_difference'] or 0 for r in rows),rows=rows)
    (a.out/'verification.json').write_text(json.dumps(output,indent=2));print(json.dumps({k:v for k,v in output.items() if k!='rows'}))


if __name__=='__main__':main()

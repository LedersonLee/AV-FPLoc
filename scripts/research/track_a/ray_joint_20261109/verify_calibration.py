"""Recompute fit-only normalization and OOF uncertainty scales, independently of stored summaries."""
from pathlib import Path
import json
import numpy as np, torch
from oracle_capacity import represent
P=Path('/output');OLD=Path('/workspace/experiments/directional_tribranch_20261109');pr=json.loads((P/'PROTOCOL.json').read_text())
def main():
 data={s:dict(np.load(P/'audio'/s/'features.npz')) for s in pr['training_scenes']};obs={s:dict(np.load(OLD/'data'/s/'observations.npz')) for s in data};checks=0;maxerr=0.
 for folder in sorted((P/'calibration').iterdir()):
  r=json.loads((folder/'complete.json').read_text());fit=set(r['outer_fit_scenes']);held=set(r['outer_held_scenes']);assert fit.isdisjoint(held);assert fit|held==set(pr['training_scenes']);seen=[];errs=[]
  for inner in r['inner_fits']:
   ck=torch.load(folder/f"inner{inner['inner']}.pt",weights_only=False,map_location='cpu');a=ck['fit_scenes'];b=ck['validation_scenes'];assert set(a).isdisjoint(b);assert set(a+b)==fit;seen+=b;key='raw' if ck['group']=='joint' else ck['kind'];x=np.concatenate([data[s][key].astype('float32') for s in a]);axes=(0,) if x.ndim==2 else (0,2,3);np.testing.assert_array_equal(x.mean(axes,keepdims=True),ck['mu']);np.testing.assert_array_equal(x.std(axes,keepdims=True).clip(.01),ck['std']);v=np.concatenate([obs[s]['visual'] for s in a]);np.testing.assert_array_equal(v.mean((0,1),keepdims=True),ck['vm']);np.testing.assert_array_equal(v.std((0,1),keepdims=True).clip(.01),ck['vs']);checks+=4
   for sid in b:
    oof=np.load(folder/f'{sid}_oof.npz');e=abs(oof['prediction']-np.log1p(represent(obs[sid]['target'],'raw36')));np.testing.assert_array_equal(e,oof['absolute_error']);errs.append(e);checks+=1
  assert sorted(seen)==sorted(fit)
  e=np.concatenate(errs);stored=np.load(folder/'scales.npz');np.testing.assert_array_equal(e.mean(0).clip(.02),stored['sigma']);np.testing.assert_array_equal(np.quantile(e,.9,axis=0),stored['q90']);checks+=2
  for row in r['held_records']:
   sid=row['scene'];assert sid in held;group=row['group'];kind=row['kind'];seed=row['seed'];fold=row['fold'];model=f'{kind}_k36_seed{seed}_fold{fold}' if group=='acoustic' else f'{kind}_seed{seed}_fold{fold}';pred=np.load(P/group/model/f'{sid}.npz')['prediction'];err=abs(pred-np.log1p(represent(obs[sid]['target'],'raw36')));delta=abs(float((err<=stored['q90']).mean())-row['interval90_coverage']);maxerr=max(maxerr,delta);assert delta==0;checks+=1
 result=dict(calibration_blocks=24,inner_models=48,checks=checks,normalization_fit_only=True,inner_validation_exactly_partitions_outer_fit=True,outer_held_excluded=True,recomputed_scales_exact=True,coverage_max_error=maxerr);(P/'calibration_verification.json').write_text(json.dumps(result,indent=2));print(result)
if __name__=='__main__':main()

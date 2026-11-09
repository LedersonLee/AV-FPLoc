"""Require all27 matched fits before comparing encoder update boundaries."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch

def read(p):return json.loads(p.read_text())
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 paths={r:a.root/('foundation_matched_v1' if r=='joint' else f'foundation_{r}_v1') for r in ['joint','frozen','partial']}
 for path in paths.values():assert len(read(path/'summary.json'))==9,'Await all scheduled fits'
 a.out.mkdir(parents=True,exist_ok=False);allrows=[];validation=[];summary=[]
 base=read(paths['joint']/'protocol.json')
 for regime,path in paths.items():
  protocol=read(path/'protocol.json')
  for k in ['data','steps','batch','lr','weight_decay','visual_aux_weight','acoustic_aux_weight','map_anchor_weight','skip_sampling_steps','pretrained_sha256','visual_cache_protocol_sha256','lowlight_cache_protocol_sha256']:
   assert protocol[k]==base[k],f'Matched setting differs: {regime}/{k}'
  for mode in base['modes']:
   for seed in base['seeds']:
    rec=read(path/f'{mode}_{seed}.json');assert rec['history'][-1]['step']==999
    init=torch.load(path/f'{mode}_{seed}_initial.pt',weights_only=True,map_location='cpu');original=torch.load(paths['joint']/f'{mode}_{seed}_initial.pt',weights_only=True,map_location='cpu')
    assert init.keys()==original.keys() and all(torch.equal(init[k],original[k]) for k in init)
    state=torch.load(path/f'{mode}_{seed}.pt',weights_only=True,map_location='cpu')
    if regime!='joint':
     active=set(rec['active_parameters']);frozen=[k for k in init if k not in active]
     assert all(torch.equal(init[k],state[k]) for k in frozen),'Frozen tensors differ after full fitting'
    else:frozen=[]
    validation.append(dict(regime=regime,mode=mode,seed=seed,initial_equal=True,frozen_tensors_checked=len(frozen),execution_code_sha=rec['execution_code_sha'],checkpoint_sha256=hashlib.sha256((path/f'{mode}_{seed}.pt').read_bytes()).hexdigest()))
    for arm,metrics in rec['val']['summary'].items():summary.append(dict(regime=regime,mode=mode,seed=seed,arm=arm,**metrics))
    allrows.extend([dict(regime=regime,mode=mode,seed=seed,**r) for r in rec['val']['rows']])
 rng=np.random.default_rng(20261109);paired=[];scenes=base['data']['scenes']['val']
 for mode in base['modes']:
  for left,right in [('frozen','joint'),('partial','joint'),('partial','frozen')]:
   for arm in ['normal','audio_shuffle','low_light']:
    for metric in ['joint','fine','raw20','nll']:
     d=np.array([np.mean([r[metric] for r in allrows if r['regime']==left and r['mode']==mode and r['arm']==arm and r['scene']==s])-np.mean([r[metric] for r in allrows if r['regime']==right and r['mode']==mode and r['arm']==arm and r['scene']==s]) for s in scenes]);scale=1 if metric=='nll' else 100;b=d[rng.integers(len(d),size=(10000,len(d)))].mean(1)*scale
     paired.append(dict(mode=mode,left=left,right=right,arm=arm,metric=metric,delta=float(d.mean()*scale),ci95=np.quantile(b,[.025,.975]).tolist(),units='nats' if metric=='nll' else 'pp',unit='scene, seeds averaged'))
 for name,obj in [('per_seed',summary),('validation',validation),('paired',paired)]:(a.out/f'{name}.json').write_text(json.dumps(obj,indent=2))
 print('COMPLETE',flush=True)
if __name__=='__main__':main()

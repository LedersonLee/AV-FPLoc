"""Inference-only raw/learned FP swaps at identical frozen visual weights."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
from track_b.models.foundation import FoundationStructureMatcher
from track_b.models.structure import visible_rays
from track_b.scripts.train_structure import scene_to_gpu,attach_visual_features,evaluate,full_descriptors

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.4)
 runs=['matched_frozen_v1','matched_joint_v1','learned_fp_w1000_v1','fixed_fp_w1000_v1'];base=read(a.root/runs[0]/'protocol.json');scenes=[];manifest={x['scene']:x['sha256'] for x in read(a.root/'features_v1/manifest.json')}
 for sid in base['data']['scenes']['val']:
  s=scene_to_gpu(torch.load(a.cache/f'{sid}.pt',weights_only=True));f=a.root/'features_v1'/f'{sid}.pt';assert sha(f)==manifest[sid];attach_visual_features(s,torch.load(f,weights_only=True));scenes.append(s)
 rows=[];drifts=[];replay=0
 protocol=dict(runs=runs,seeds=[42,43,44],intervention='same frozen visual checkpoint; only candidate FP descriptor switches raw versus learned',test_accessed=False,evidence='Exploratory reused development; six scenes; no training or selection',checkpoint_hashes={f'{r}/{seed}':sha(a.root/r/f'visual_{seed}.pt') for r in runs for seed in [42,43,44]})
 (a.out/'protocol.json').write_text(json.dumps(protocol,indent=2))
 for run in runs:
  for seed in [42,43,44]:
   m=FoundationStructureMatcher().cuda().eval();m.load_state_dict(torch.load(a.root/run/f'visual_{seed}.pt',weights_only=True));original=read(a.root/run/f'visual_{seed}.json');isfixed=read(a.root/run/'protocol.json').get('fixed_fp',False)
   with torch.no_grad():
    for s in scenes:
     delta=full_descriptors(m,s,True)-full_descriptors(m,s,False);mask=visible_rays(s['hfov_deg'],device='cuda');drifts.append(dict(run=run,seed=seed,scene=s['scene'],visible_normalized_rmse=float(((delta.square()*mask).sum(-1)/mask.sum()).mean().sqrt()),max_abs=float(delta.abs().max())))
   for residual in [False,True]:
    result=evaluate(m,scenes,'visual',residual,interventions=False,diversity=False)
    if residual != isfixed:
     expected={(r['scene'],r['query']):r for r in original['val']['rows']}
     for row in result['rows']:
      old=expected[row['scene'],row['query']]
      for k in ['joint','r1','fine','raw20','nll']:assert np.isclose(row[k],old[k],atol=2e-5,rtol=1e-5),(run,seed,k,row,old)
      replay+=1
    rows.extend(dict(run=run,seed=seed,descriptor='learned' if residual else 'raw',**r) for r in result['rows'])
   print('DONE',run,seed,flush=True)
 summary=[dict(run=run,descriptor=desc,**{k:float(np.mean([r[k] for r in rows if r['run']==run and r['descriptor']==desc])) for k in ['joint','r1','fine','raw20','nll']}) for run in runs for desc in ['raw','learned']]
 rng=np.random.default_rng(20261109);paired=[]
 for run in runs:
  d=np.array([np.mean([r['joint'] for r in rows if r['run']==run and r['descriptor']=='raw' and r['scene']==s['scene']])-np.mean([r['joint'] for r in rows if r['run']==run and r['descriptor']=='learned' and r['scene']==s['scene']]) for s in scenes]);boot=d[rng.integers(0,len(d),(10000,len(d)))].mean(1)*100;paired.append(dict(run=run,raw_minus_learned_pp=float(d.mean()*100),ci95_pp=np.quantile(boot,[.025,.975]).tolist()))
 for name,value in [('predictions',rows),('descriptor_drift',drifts),('summary',dict(results=summary,paired=paired,replay_cases=replay))]:(a.out/f'{name}.json').write_text(json.dumps(value,indent=2))
 print(json.dumps(summary));print('COMPLETE',flush=True)
if __name__=='__main__':main()

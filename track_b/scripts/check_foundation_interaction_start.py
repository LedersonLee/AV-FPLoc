"""Replay the new model's serialized initial state against archived independent F1."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from track_b.models.foundation_interaction import FoundationInteractionMatcher
from track_b.scripts.train_structure import scene_to_gpu,attach_visual_features
from track_b.scripts.train_foundation_interaction import evaluate,sha,read

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--foundation',type=Path,required=True);ap.add_argument('--run',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--seed',type=int,default=42);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);scenes=[]
 for sid in read(a.cache/'protocol.json')['scenes']['val']:
  s=scene_to_gpu(torch.load(a.cache/f'{sid}.pt',weights_only=True));attach_visual_features(s,torch.load(a.foundation/'features_v1'/f'{sid}.pt',weights_only=True));package=torch.load(a.foundation/'features_lowlight_v3'/f'{sid}.pt',weights_only=True);assert package['query_ids']==s['query_ids'];s['lowlight_features']=package['visual_features'].cuda();scenes.append(s)
 m=FoundationInteractionMatcher().cuda().eval();path=a.run/f'attention_{a.seed}_initial.pt';m.load_state_dict(torch.load(path,weights_only=True));result=evaluate(m,scenes);reference={}
 for condition in ['normal','lowlight']:
  run='independent_fusion_anchor'+('_lowlight' if condition=='lowlight' else '')+'_v1'
  for r in read(a.foundation/run/'predictions.json'):
   if r['seed']==a.seed:reference[condition,r['method'],r['scene'],r['query']]=r
 mismatches=[]
 for r in result['rows']:
  condition='lowlight' if r['arm']=='low_light' else 'normal';method='fixed_audio_shuffle' if r['arm']=='audio_shuffle' else 'fixed';old=reference[condition,method,r['scene'],r['query']]
  delta={k:dict(current=r[k],reference=old[k]) for k in ['joint','r1','fine','raw20','nll'] if not np.isclose(r[k],old[k],atol=2e-5,rtol=1e-5)}
  if delta:mismatches.append(dict(scene=r['scene'],query=r['query'],arm=r['arm'],differences=delta))
 report=dict(seed=a.seed,initial_checkpoint_sha256=sha(path),checks=len(result['rows']),mismatches=mismatches,summary=result['summary'],interpretation='Full bank, seed42, normal/shuffled/brightness; not a claim of independent evaluation')
 (a.out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report));assert not mismatches;print('COMPLETE',flush=True)
if __name__=='__main__':main()

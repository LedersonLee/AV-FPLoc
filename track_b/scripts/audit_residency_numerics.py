"""Preserve failed bitwise check; compare streaming and resident repeat noise."""
import argparse,json
from pathlib import Path
import torch

def read(p):return json.loads(p.read_text())
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 paths={k:a.root/name for k,name in [('gpu','residency_gpu_smoke_v1'),('cpu','residency_cpu_smoke_v1'),('gpu_repeat','residency_gpu_repeat_v1')]};results=[]
 for other in ['cpu','gpu_repeat']:
  record=dict(reference='gpu',other=other)
  for phase,suffix in [('aux','_pretrained'),('final','')]:
   left=torch.load(paths['gpu']/f'visual_42{suffix}.pt',weights_only=True,map_location='cpu');right=torch.load(paths[other]/f'visual_42{suffix}.pt',weights_only=True,map_location='cpu')
   differences={k:float((left[k]-right[k]).abs().max()) for k in left if not torch.equal(left[k],right[k])}
   record[phase]=dict(bitwise_equal=not differences,max_abs_difference=max(differences.values(),default=0),changed_tensor_count=len(differences),changed_tensor_maxima=differences)
  left,right=[read(paths[k]/'visual_42.json') for k in ['gpu',other]]
  record['predictions']={}
  for part in ['train','val']:
   ll,rr=left[part]['rows'],right[part]['rows'];assert len(ll)==len(rr) and all((x['scene'],x['query'],x['arm'])==(y['scene'],y['query'],y['arm']) for x,y in zip(ll,rr))
   record['predictions'][part]=dict(cases=len(ll),metric_mismatch_counts={k:sum(x[k]!=y[k] for x,y in zip(ll,rr)) for k in ['joint','r1','fine','raw20','diverse20']},max_nll_difference=max(abs(x['nll']-y['nll']) for x,y in zip(ll,rr)))
  record['max_loss_difference']=max(abs(x['loss']-y['loss']) for x,y in zip(left['history'],right['history']))
  results.append(record)
 a.out.write_text(json.dumps(dict(comparisons=results,interpretation='Bitwise failure remains a failure; this diagnostic describes observed short-run numeric differences, not full-run deterministic equivalence.'),indent=2));print(json.dumps(results));print('COMPLETE')
if __name__=='__main__':main()

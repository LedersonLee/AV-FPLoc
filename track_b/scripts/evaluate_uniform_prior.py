"""Exact random-pose baseline on the full valid bank; GT is evaluation-only."""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
import torch

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def inclusion(n,k,draws=20):
 """Probability at least one of k successes in draws uniform distinct cells."""
 draws=min(n,draws)
 if k==0:return 0.
 if n-k<draws:return 1.
 return float(-np.expm1(sum(math.log1p(-k/(n-i)) for i in range(draws))))
def main():
 p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--code-sha',required=True);a=p.parse_args()
 protocol=read(a.cache/'protocol.json');manifest={r['scene']:r for r in read(a.cache/'manifest.json')};torch.set_num_threads(2);rows=[]
 for sid in protocol['scenes']['val']:
  path=a.cache/f'{sid}.pt';assert sha(path)==manifest[sid]['sha256'];scene=torch.load(path,weights_only=True,map_location='cpu');poses=scene['candidates'][:,:3];n=len(poses)
  for q,gt in enumerate(scene['gt']):
   dist=(poses[:,:2]-gt[:2]).norm(dim=-1);dyaw=poses[:,2]-gt[2];yaw=torch.atan2(dyaw.sin(),dyaw.cos()).abs();ok=(dist<=1+1e-6)&(yaw<=math.pi/6+1e-6);k=int(ok.sum())
   rows.append(dict(scene=sid,query=scene['query_ids'][q],candidates=n,success_cells=k,joint=k/n,r1=float((dist<=1+1e-6).sum())/n,fine=float((dist<=.1+1e-6).sum())/n,random20=inclusion(n,k),nll=math.log(n),entropy=math.log(n),oracle_joint=float(k>0),oracle_fine=float((dist<=.1+1e-6).any())))
  print('DONE',sid,flush=True)
 keys=[(r['scene'],r['query']) for r in rows];assert len(keys)==len(set(keys))
 a.out.mkdir(parents=True,exist_ok=False)
 fields=['joint','r1','fine','random20','nll','entropy','oracle_joint','oracle_fine'];summary={k:float(np.mean([r[k] for r in rows])) for k in fields}
 setup=dict(code_sha=a.code_sha,cache_protocol_sha256=sha(a.cache/'protocol.json'),cache_manifest_sha256=sha(a.cache/'manifest.json'),scene_ids=protocol['scenes']['val'],queries=len(rows),definition='Uniform random pose among all bank cells; exact expectation. Random20 samples distinct cells without replacement; not score-tie topk or diverse-NMS.',gt_use='evaluation only; no filtering or ranking',training='none',evidence='Exploratory reused development cohort',test_accessed=False)
 for name,obj in [('protocol',setup),('rows',rows),('summary',summary)]:(a.out/f'{name}.json').write_text(json.dumps(obj,indent=2))
 print(json.dumps(summary),flush=True);print('COMPLETE',flush=True)
if __name__=='__main__':main()

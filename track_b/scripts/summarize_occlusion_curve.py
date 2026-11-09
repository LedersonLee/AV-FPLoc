"""Complete declared occlusion curve; no choosing a favorable severity."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();runs={}
 for variant in ['anchor','frozen']:
  for level in [0,25,50,75]:
   name=f'independent_fusion_{variant}'+(f'_occlusion{level}' if level else '')+'_v1';p=a.root/name
   assert (p/'summary.json').exists(),f'Await complete curve: {name}'
   runs[variant,level]=(p,read(p/'summary.json'),read(p/'predictions.json'))
 a.out.mkdir(parents=True,exist_ok=False);curve=[];paired=[];negative_transfer=[];hashes={};rng=np.random.default_rng(20261109)
 for (variant,level),(path,summary,rows) in runs.items():
  assert len(rows)==6912
  if level:
   assert read(path/'selection.json')==read(runs[variant,0][0]/'selection.json')
   protocol=read(a.root/f'features_occlusion{level}_v1/protocol.json');assert protocol['center_occlusion']==level/100 and protocol['brightness']==1.
  for method,metrics in summary['summary'].items():curve.append(dict(variant=variant,occlusion_percent=level,method=method,**metrics))
  scenes=sorted({r['scene'] for r in rows});assert len(scenes)==6
  for method in ['visual','fixed','calibrated']:
   for metric in ['joint','r1','fine','raw20','diverse20']:
    baseline=runs[variant,0][2];d=np.array([np.mean([r[metric] for r in rows if r['scene']==s and r['method']==method])-np.mean([r[metric] for r in baseline if r['scene']==s and r['method']==method]) for s in scenes]);boot=d[rng.integers(0,6,(10000,6))].mean(1)*100;paired.append(dict(variant=variant,occlusion_percent=level,method=method,metric=metric,shift_minus_normal_pp=float(d.mean()*100),ci95_pp=np.quantile(boot,[.025,.975]).tolist(),unit='six scenes; seeds averaged'))
  for left,right,metric,scale in [('fixed','acoustic','joint',100),('calibrated','fixed','nll',1)]:
   delta=np.array([np.mean([r[metric] for r in rows if r['scene']==s and r['method']==left])-np.mean([r[metric] for r in rows if r['scene']==s and r['method']==right]) for s in scenes]);boot=delta[rng.integers(0,6,(10000,6))].mean(1)*scale;negative_transfer.append(dict(variant=variant,occlusion_percent=level,left=left,right=right,metric=metric,delta=float(delta.mean()*scale),ci95=np.quantile(boot,[.025,.975]).tolist(),units='pp' if scale==100 else 'nats'))
  for p in path.glob('*.json'):hashes[str(p.relative_to(a.root))]=sha(p)
 for name,data in [('negative_transfer',negative_transfer),('curve',curve),('paired_scene_bootstrap',paired),('external_hashes',hashes)]:(a.out/f'{name}.json').write_text(json.dumps(data,indent=2))
 print(json.dumps([r for r in curve if r['method'] in ['visual','fixed']]));print('COMPLETE')
if __name__=='__main__':main()

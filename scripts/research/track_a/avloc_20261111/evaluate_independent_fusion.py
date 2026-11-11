"""Evaluate fixed fusion of independently trained A0/A1 using fresh outputs.
Immutable local source snapshot, pinned Docker image, new output paths, stop on error.
"""
import argparse,json,os,shutil,subprocess,time
from pathlib import Path
from predict_hash import sha


def run(cmd,**kwargs):return subprocess.run(cmd,check=True,text=True,**kwargs)

def main():
 p=argparse.ArgumentParser()
 for k in ['workspace','study','out']:p.add_argument('--'+k,type=Path,required=True)
 p.add_argument('--seed',type=int,required=True);a=p.parse_args();tag=f'independent_equal_seed{a.seed}';head='F'
 image='sha256:e51cb712100f25d0018aa65e371e63deaa51d12601c465ce698c0c446800c318'
 source=a.workspace/'AV-FPLoc/scripts/research/track_a/avloc_20261111';inputs={}
 for arm in ['A0','A1']:
  path=a.study/'predictions'/f'{arm}_seed{a.seed}'/'manifest.json';m=json.loads(path.read_text())
  assert m['complete'] and m['arm']==arm and m['seed']==a.seed and m['queries']==9965 and m['split']=='val'
  inputs[arm]=sha(path)
 a.out.mkdir(parents=True,exist_ok=False);snapshot=a.out/'source';snapshot.mkdir()
 for path in source.glob('*.py'):shutil.copyfile(path,snapshot/path.name)
 hashes={p.name:sha(p) for p in snapshot.glob('*.py')}
 pred=f'/study/predictions/{tag}';common=['--predictions',pred,'--data','/study/data_v1'];stages=[('fusion',False,['fusion_predictions.py','--mode','independent_equal','--first',f'/study/predictions/A0_seed{a.seed}','--second',f'/study/predictions/A1_seed{a.seed}','--out',pred])]
 # Both localization grids precede costly temperature sweeps.
 for grid in [20,10]:
  name=f'{tag}_{grid}cm';ev=f'/study/evaluation/{name}'
  stages.append((f'evaluate{grid}',False,['evaluate_fields.py',*common,'--banks','/study/banks_v1','--out',ev,'--head',head,'--grid-cm',str(grid)]))
  stages.append((f'analyze{grid}',False,['analyze_rays.py',*common,'--evaluation',ev,'--geometry',f'/study/observability/geometry{grid}','--head',head,'--out',f'/study/analysis/{name}']))
 for grid in [20,10]:
  name=f'{tag}_{grid}cm';stages.append((f'calibrate{grid}',False,['calibrate_rays.py',*common,'--banks','/study/banks_v1','--native',f'/study/evaluation/{name}','--out',f'/study/calibration/{name}','--head',head,'--grid-cm',str(grid)]))
 for _,_,args in stages:
  output=args[args.index('--out')+1];assert not (a.study/Path(output).relative_to('/study')).exists(),('Existing output requires review',output)
 plan=dict(control=tag,input_manifest_sha256=inputs,image_id=image,source_sha256=hashes,stages=stages,policy='fixed independent expert predictions; no retries, overwrite or model training',test_accessed=False)
 (a.out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
 for label,gpu,args in stages:
  for name,digest in hashes.items():assert sha(snapshot/name)==digest
  output=args[args.index('--out')+1];dest=a.study/Path(output).relative_to('/study');assert not dest.exists()
  cmd=['docker','run','--rm','--name',f'avloc-{tag.lower()}-eval-{label}-20261112','--network','none','--cpus','4','--memory','12g','--shm-size','1g','--user',f'{os.getuid()}:{os.getgid()}','-e','PYTHONDONTWRITEBYTECODE=1','-e','CUBLAS_WORKSPACE_CONFIG=:4096:8','-e','OMP_NUM_THREADS=2','-e','OPENBLAS_NUM_THREADS=2','-v',f'{snapshot}:/source:ro','-v',f'{a.study}:/study:rw']
  if gpu:cmd+=['--gpus','all']
  cmd += [image,'python','/source/'+args[0],*args[1:]]
  print('START',label,time.time(),flush=True)
  with (a.out/(label+'.log')).open('x') as f:run(cmd,stdout=f,stderr=subprocess.STDOUT)
  receipt=dest/('manifest.json' if label=='fusion' else 'summary.json');d=json.loads(receipt.read_text());assert d['complete'] and d['queries']==9965
  (a.out/(label+'_receipt.json')).write_text(json.dumps(dict(status='complete',output=output,summary_sha256=sha(receipt),seconds_utc=time.time()))+'\n');print('COMPLETE',label,flush=True)
 (a.out/'complete.json').write_text(json.dumps(dict(status='independent_fusion_evaluation_complete',arm='independent_equal',seed=a.seed,full_research_complete=False,test_accessed=False))+'\n')
if __name__=='__main__':main()

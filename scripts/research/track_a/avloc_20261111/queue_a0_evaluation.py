"""Wait for the live A0 run, then evaluate its completed checkpoint without training.
Immutable local source snapshot, pinned Docker image, new output paths, stop on error.
"""
import argparse,json,os,shutil,subprocess,time
from pathlib import Path
from predict_hash import sha


def run(cmd,**kwargs):return subprocess.run(cmd,check=True,text=True,**kwargs)

def main():
 p=argparse.ArgumentParser()
 for k in ['workspace','study','out']:p.add_argument('--'+k,type=Path,required=True)
 p.add_argument('--await-container',required=True);a=p.parse_args()
 state=json.loads(run(['docker','inspect',a.await_container],capture_output=True).stdout)[0]
 assert state['State']['Running'],'Must attach to actual live training, never assume a stale receipt is live'
 source=a.workspace/'AV-FPLoc/scripts/research/track_a/avloc_20261111';training=a.study/'runs/A0_seed42';protocol=json.loads((training/'protocol.json').read_text())
 assert protocol['arm']=='A0' and protocol['seed']==42
 for name,digest in protocol['source_sha256'].items():assert sha(source/name)==digest,('Training source mismatch',name)
 image=state['Image'];container_id=state['Id'];a.out.mkdir(parents=True,exist_ok=False);snapshot=a.out/'source';snapshot.mkdir()
 for path in source.glob('*.py'):shutil.copyfile(path,snapshot/path.name)
 hashes={p.name:sha(p) for p in snapshot.glob('*.py')}
 pred='/study/predictions/A0_seed42';common=['--predictions',pred,'--data','/study/data_v1'];stages=[('predict',True,['predict.py','--run','/study/runs/A0_seed42','--data','/study/data_v1','--out',pred,'--batch','4','--workers','2'])]
 # Both localization grids precede costly temperature sweeps.
 for grid in [20,10]:
  name=f'A0_seed42_{grid}cm';ev=f'/study/evaluation/{name}'
  stages.append((f'evaluate{grid}',False,['evaluate_fields.py',*common,'--banks','/study/banks_v1','--out',ev,'--head','V','--grid-cm',str(grid)]))
  stages.append((f'analyze{grid}',False,['analyze_rays.py',*common,'--evaluation',ev,'--geometry',f'/study/observability/geometry{grid}','--head','V','--out',f'/study/analysis/{name}']))
 for grid in [20,10]:
  name=f'A0_seed42_{grid}cm';stages.append((f'calibrate{grid}',False,['calibrate_rays.py',*common,'--banks','/study/banks_v1','--native',f'/study/evaluation/{name}','--out',f'/study/calibration/{name}','--head','V','--grid-cm',str(grid)]))
 for _,_,args in stages:
  output=args[args.index('--out')+1];assert not (a.study/Path(output).relative_to('/study')).exists(),('Existing output requires review',output)
 plan=dict(await_container_id=container_id,image_id=image,source_sha256=hashes,training_code_commit=protocol['code_commit'],stages=stages,policy='require normal container exit and training_complete; no retries, overwrite or model training',test_accessed=False)
 (a.out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');print('WAIT LIVE',container_id,flush=True)
 result=run(['docker','wait',container_id],capture_output=True);assert result.stdout.strip()=='0',result.stdout
 complete=json.loads((training/'complete.json').read_text());assert complete['status']=='training_complete';best=json.loads((training/'best.json').read_text());assert (training/best['path']).is_file()
 for label,gpu,args in stages:
  for name,digest in hashes.items():assert sha(snapshot/name)==digest
  output=args[args.index('--out')+1];dest=a.study/Path(output).relative_to('/study');assert not dest.exists()
  cmd=['docker','run','--rm','--name',f'avloc-a0-eval-{label}-20261111','--network','none','--cpus','4','--memory','12g','--shm-size','1g','--user',f'{os.getuid()}:{os.getgid()}','-e','PYTHONDONTWRITEBYTECODE=1','-e','CUBLAS_WORKSPACE_CONFIG=:4096:8','-e','OMP_NUM_THREADS=2','-e','OPENBLAS_NUM_THREADS=2','-v',f'{snapshot}:/source:ro','-v',f'{a.study}:/study:rw']
  if gpu:cmd+=['--gpus','all']
  cmd += [image,'python','/source/'+args[0],*args[1:]]
  print('START',label,time.time(),flush=True)
  with (a.out/(label+'.log')).open('x') as f:run(cmd,stdout=f,stderr=subprocess.STDOUT)
  receipt=dest/('manifest.json' if label=='predict' else 'summary.json');d=json.loads(receipt.read_text());assert d['complete'] and d['queries']==9965
  (a.out/(label+'_receipt.json')).write_text(json.dumps(dict(status='complete',output=output,summary_sha256=sha(receipt),seconds_utc=time.time()))+'\n');print('COMPLETE',label,flush=True)
 (a.out/'complete.json').write_text(json.dumps(dict(status='A0_evaluation_queue_complete',full_research_complete=False,test_accessed=False))+'\n')
if __name__=='__main__':main()

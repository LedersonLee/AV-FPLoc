"""Evaluate a completed expert run using immutable sources and fresh output paths.
Immutable local source snapshot, pinned Docker image, new output paths, stop on error.
"""
import argparse,json,os,shutil,subprocess,time
from pathlib import Path
from predict_hash import sha


def run(cmd,**kwargs):return subprocess.run(cmd,check=True,text=True,**kwargs)

def evaluation_stages(tag, head, arm):
    pred=f'/study/predictions/{tag}';common=['--predictions',pred,'--data','/study/data_v1'];stages=[('predict',True,['predict.py','--run',f'/study/runs/{tag}','--data','/study/data_v1','--out',pred,'--batch','4','--workers','2'])]
    # Both localization grids precede costly temperature sweeps.
    for grid in [20,10]:
     name=f'{tag}_{grid}cm';ev=f'/study/evaluation/{name}'
     stages.append((f'evaluate{grid}',False,['evaluate_fields.py',*common,'--banks','/study/banks_v1','--out',ev,'--head',head,'--grid-cm',str(grid)]))
     stages.append((f'analyze{grid}',False,['analyze_rays.py',*common,'--evaluation',ev,'--geometry',f'/study/observability/geometry{grid}','--head',head,'--out',f'/study/analysis/{name}']))
    for grid in [20,10]:
     name=f'{tag}_{grid}cm';stages.append((f'calibrate{grid}',False,['calibrate_rays.py',*common,'--banks','/study/banks_v1','--native',f'/study/evaluation/{name}','--out',f'/study/calibration/{name}','--head',head,'--grid-cm',str(grid)]))
    if arm == 'A3':
        stages.append(('router',False,['analyze_router.py',*common,'--out',f'/study/analysis/{tag}_router']))
        uniform=f'/study/predictions/{tag}_uniform'
        stages.append(('uniform_export',False,['fusion_predictions.py','--mode','A3_uniform','--first',pred,'--out',uniform]))
        for grid in [20,10]:
            ev=f'/study/evaluation/{tag}_uniform_{grid}cm'
            stages.append((f'uniform_evaluate{grid}',False,['evaluate_fields.py','--predictions',uniform,'--data','/study/data_v1','--banks','/study/banks_v1','--out',ev,'--head','F','--grid-cm',str(grid)]))
    return stages


def main():
 p=argparse.ArgumentParser()
 for k in ['workspace','study','out']:p.add_argument('--'+k,type=Path,required=True)
 p.add_argument('--await-container',help='Optional live completed-run predecessor; event wait, never restart')
 p.add_argument('--arm',choices=['A0','A1','A2','A2_disabled','A3'],required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args();tag=f'{a.arm}_seed{a.seed}';head={'A0':'V','A1':'A','A2':'AV','A2_disabled':'AV','A3':'F'}[a.arm]
 image='sha256:e51cb712100f25d0018aa65e371e63deaa51d12601c465ce698c0c446800c318'
 if a.await_container:
  state=json.loads(run(['docker','inspect',a.await_container],capture_output=True).stdout)[0]
  assert state['State']['Running'],'Await mode requires a live predecessor at scheduling'
  assert f'/study/runs/{tag}' in state['Config']['Cmd'] and state['Image']==image
  assert not a.out.exists(),'Existing evaluation output requires review'
  print('WAIT LIVE',state['Id'],flush=True)
  result=run(['docker','wait',state['Id']],capture_output=True)
  assert result.stdout.strip()=='0',result.stdout
 source=a.workspace/'AV-FPLoc/scripts/research/track_a/avloc_20261111';training=a.study/'runs'/tag;protocol=json.loads((training/'protocol.json').read_text())
 assert protocol['arm']==a.arm and protocol['seed']==a.seed
 assert json.loads((training/'complete.json').read_text())['status']=='training_complete'
 for name,digest in protocol['source_sha256'].items():assert sha(source/name)==digest,('Training source mismatch',name)
 a.out.mkdir(parents=True,exist_ok=False);snapshot=a.out/'source';snapshot.mkdir()
 for path in source.glob('*.py'):shutil.copyfile(path,snapshot/path.name)
 hashes={p.name:sha(p) for p in snapshot.glob('*.py')}
 stages=evaluation_stages(tag,head,a.arm)
 for _,_,args in stages:
  output=args[args.index('--out')+1];assert not (a.study/Path(output).relative_to('/study')).exists(),('Existing output requires review',output)
 plan=dict(completed_run=tag,complete_sha256=sha(training/'complete.json'),image_id=image,source_sha256=hashes,training_code_commit=protocol['code_commit'],stages=stages,policy='require training_complete and unchanged training sources; no retries, overwrite or model training',test_accessed=False)
 (a.out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
 complete=json.loads((training/'complete.json').read_text());assert complete['status']=='training_complete';best=json.loads((training/'best.json').read_text());assert (training/best['path']).is_file()
 for label,gpu,args in stages:
  for name,digest in hashes.items():assert sha(snapshot/name)==digest
  output=args[args.index('--out')+1];dest=a.study/Path(output).relative_to('/study');assert not dest.exists()
  cmd=['docker','run','--rm','--name',f'avloc-{tag.lower()}-eval-{label}-20261112','--network','none','--cpus','4','--memory','12g','--shm-size','1g','--user',f'{os.getuid()}:{os.getgid()}','-e','PYTHONDONTWRITEBYTECODE=1','-e','CUBLAS_WORKSPACE_CONFIG=:4096:8','-e','OMP_NUM_THREADS=2','-e','OPENBLAS_NUM_THREADS=2','-v',f'{snapshot}:/source:ro','-v',f'{a.study}:/study:rw']
  if gpu:cmd+=['--gpus','all']
  cmd += [image,'python','/source/'+args[0],*args[1:]]
  print('START',label,time.time(),flush=True)
  with (a.out/(label+'.log')).open('x') as f:run(cmd,stdout=f,stderr=subprocess.STDOUT)
  receipt=dest/('manifest.json' if label in ['predict','uniform_export'] else 'summary.json');d=json.loads(receipt.read_text());assert d['complete'] and d['queries']==9965
  (a.out/(label+'_receipt.json')).write_text(json.dumps(dict(status='complete',output=output,summary_sha256=sha(receipt),seconds_utc=time.time()))+'\n');print('COMPLETE',label,flush=True)
 (a.out/'complete.json').write_text(json.dumps(dict(status='completed_expert_evaluation_complete',arm=a.arm,seed=a.seed,full_research_complete=False,test_accessed=False))+'\n')
if __name__=='__main__':main()

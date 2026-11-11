"""Sequential local training queue, never starts A3 or restarts interrupted runs."""
import argparse,hashlib,json,subprocess,time
from pathlib import Path

def run(args,**kwargs):return subprocess.run(args,check=True,text=True,**kwargs)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--study',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--await-container',required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 source=a.workspace/'AV-FPLoc/scripts/research/track_a/avloc_20261111';initial=a.study/'runs/A0_seed42';protocol=json.loads((initial/'protocol.json').read_text());base=protocol['code_commit'];closure=protocol['source_sha256'];image=run(['docker','image','inspect','--format','{{.Id}}','avfploc:cu124-clean'],capture_output=True).stdout.strip()
 def verify_source():
  for name,digest in closure.items():assert sha(source/name)==digest,('Training source changed',name)
 verify_source();plan=[(arm,seed) for seed in [42,43] for arm in ['A0','A1','A2'] if (arm,seed)!=('A0',42)]+[('A2_disabled',s) for s in [42,43]]
 (a.out/'plan.json').write_text(json.dumps(dict(await_container=a.await_container,training_source_commit=base,source_sha256=closure,image_id=image,plan=plan,A3_included=False,policy='wait existing A0; no restart, no overwrite, stop on failure; full100epochs/patience20',test_accessed=False),indent=2)+'\n')
 # The caller must provide an actually live container; do not rely on a lock/status file.
 state=run(['docker','inspect','--format','{{.State.Running}}',a.await_container],capture_output=True).stdout.strip();assert state=='true'
 print('WAIT LIVE',a.await_container,flush=True);wait=run(['docker','wait',a.await_container],capture_output=True);assert wait.stdout.strip()=='0',wait.stdout
 def completed(path):
  d=json.loads((path/'complete.json').read_text());assert d['status']=='training_complete';assert (path/'best.json').is_file();return d
 completed(initial)
 for arm,seed in plan:
  verify_source();dest=a.study/f'runs/{arm}_seed{seed}';assert not dest.exists(),('Existing output requires review, will not restart or overwrite',str(dest));name=f'avloc-{arm.lower()}-seed{seed}-queued-20261111';log=a.out/f'{arm}_seed{seed}.log'
  command=['docker','run','--rm','--name',name,'--gpus','all','--network','none','--cpus','8','--memory','20g','--shm-size','2g','--user','1000:1000','-e','PYTHONDONTWRITEBYTECODE=1','-e','CUBLAS_WORKSPACE_CONFIG=:4096:8','-e','PYTORCH_KERNEL_CACHE_PATH=/tmp','-v',f'{source}:/source:ro','-v',f'{a.workspace}/experiments/avloc_20261111/weights:/weights:ro','-v',f'{a.study}:/study:rw',image,'python','/source/train.py','--data','/study/data_v1','--noise','/study/esc50_v1','--weights','/weights/resnet50-11ad3fa6.pth','--out',f'/study/runs/{arm}_seed{seed}','--arm',arm,'--seed',str(seed),'--batch','8','--workers','4','--code-sha',base]
  print('START',arm,seed,time.time(),flush=True)
  with log.open('x') as f:run(command,stdout=f,stderr=subprocess.STDOUT)
  completed(dest);print('COMPLETE',arm,seed,time.time(),flush=True)
 (a.out/'complete.json').write_text(json.dumps(dict(status='expert_training_queue_complete',A3_started=False,benchmark_complete=False))+'\n')
if __name__=='__main__':main()

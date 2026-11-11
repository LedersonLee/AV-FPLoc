"""Resume the reviewed A0 boolean-correlation failure without overwriting outputs."""
import argparse, json, os, shutil, subprocess, time
from pathlib import Path
from predict_hash import sha

def main():
 p=argparse.ArgumentParser()
 for k in ['previous','out','study']:p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args(); old=json.loads((a.previous/'plan.json').read_text())
 assert json.loads((a.study/'runs/A0_seed42/complete.json').read_text())['status']=='training_complete'
 # The first two successful stages are reused only with their exact receipts.
 reused={}
 for label,filename in [('predict','manifest.json'),('evaluate20','summary.json')]:
  r=json.loads((a.previous/(label+'_receipt.json')).read_text())
  path=a.study/Path(r['output']).relative_to('/study')/filename
  assert sha(path)==r['summary_sha256']
  d=json.loads(path.read_text());assert d['complete'] and d['queries']==9965
  reused[label]=r
 a.out.mkdir(parents=True,exist_ok=False);snapshot=a.out/'source'
 shutil.copytree(a.previous/'source',snapshot)
 for name,digest in old['source_sha256'].items():assert sha(snapshot/name)==digest
 shutil.copyfile(Path(__file__).with_name('analyze_rays.py'),snapshot/'analyze_rays.py')
 hashes={f.name:sha(f) for f in snapshot.glob('*.py')}
 stages=old['stages'][2:]
 assert stages[0][0]=='analyze20'
 args=stages[0][2];args[args.index('--out')+1]='/study/analysis/A0_seed42_20cm_v2'
 for label,gpu,args in stages:
  assert not gpu
  assert not (a.study/Path(args[args.index('--out')+1]).relative_to('/study')).exists()
 plan=dict(parent_plan_sha256=sha(a.previous/'plan.json'),reused=reused,source_sha256=hashes,image_id=old['image_id'],stages=stages,test_accessed=False,reason='Fix boolean np.ptp failure in post-hoc correlation; preserve failed directory and all prior results')
 (a.out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
 for label,gpu,args in stages:
  for name,digest in hashes.items():assert sha(snapshot/name)==digest
  dest=a.study/Path(args[args.index('--out')+1]).relative_to('/study');assert not dest.exists()
  cmd=['docker','run','--rm','--name','avloc-a0-recovery-'+label+'-20261112','--network','none','--cpus','4','--memory','12g','--user',f'{os.getuid()}:{os.getgid()}','-e','PYTHONDONTWRITEBYTECODE=1','-e','OMP_NUM_THREADS=2','-e','OPENBLAS_NUM_THREADS=2','-v',f'{snapshot.resolve()}:/source:ro','-v',f'{a.study.resolve()}:/study:rw',old['image_id'],'python','/source/'+args[0],*args[1:]]
  print('START',label,time.time(),flush=True)
  with (a.out/(label+'.log')).open('x') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
  receipt=dest/'summary.json';d=json.loads(receipt.read_text());assert d['complete'] and d['queries']==9965
  (a.out/(label+'_receipt.json')).write_text(json.dumps(dict(status='complete',summary_sha256=sha(receipt),output=str(dest)))+'\n')
  print('COMPLETE',label,flush=True)
 (a.out/'complete.json').write_text(json.dumps(dict(status='A0_evaluation_recovery_complete',full_research_complete=False))+'\n')
if __name__=='__main__':main()

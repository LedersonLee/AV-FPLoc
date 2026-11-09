"""Run the frozen factorial plan after feature validation; preserve failed attempts."""
import argparse,fcntl,hashlib,json,subprocess,sys
from pathlib import Path
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2));tmp.replace(p)
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--code-sha',required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
 plan_path=a.root/'plan_v1/plan.json';plan=read(plan_path);fv=read(a.root/'feature_validation_v1/validation.json')
 assert fv['scenes']==131 and fv['queries']==8384 and fv['pilot_exact_scenes']==22
 assert fv['feature_protocol_sha256']==sha(a.root/'features_v1/protocol.json')
 assert plan['cache_protocol_sha256']==sha(a.root/'cache_v1/protocol.json')
 a.out.mkdir(parents=True,exist_ok=a.resume);lock=(a.out/'runner.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 repo=Path(__file__).resolve().parents[2]
 dependencies=['scripts/train_structure.py','models/structure.py','models/foundation.py','models/dense.py','train_regimes.py','data/folds.py','data/scene_residency.py','data/training_subset.py']
 hashes={name:sha(repo/'track_b'/name) for name in dependencies}
 config=dict(code_sha=a.code_sha,plan_sha256=sha(plan_path),feature_verification_sha256=sha(a.root/'feature_validation_v1/validation.json'),dependency_hashes=hashes)
 cp=a.out/'runner_protocol.json'
 if a.resume:assert read(cp)==config,'Resume source/data/plan mismatch'
 else:save(cp,config)
 state_path=a.out/'state.json';state=read(state_path) if a.resume and state_path.exists() else {'attempts':[]}
 for condition in plan['conditions']:
  completed={}
  for stage in condition['stages']:
   assert all(sha(repo/'track_b'/name)==digest for name,digest in hashes.items()),'Training dependency changed during batch'
   key=condition['name']+'_'+stage['stage'];prior=[x for x in state['attempts'] if x['key']==key]
   if prior and prior[-1].get('exit_code')==0:
    path=Path(prior[-1]['output']);summary=read(path/'summary.json');assert len(summary)==3
    for seed in plan['seeds']:
     rec=read(path/f"{stage['mode']}_{seed}.json")
     assert rec['execution_code_sha']==a.code_sha and rec['history'][-1]['step']==stage['aux_steps']+stage['dense_steps']-1
     assert sha(path/f"{stage['mode']}_{seed}_initial.pt")==rec['initial_state_sha256']
    completed[stage['stage']]=path;print('SKIP_COMPLETED',key,flush=True);continue
   # A released exclusive lock plus no completed exit record permits a new
   # attempt directory; prior partial artifacts are never reused/overwritten.
   path=a.out/f'{key}_attempt{len(prior)+1}'
   cmd=[sys.executable,'-m','track_b.scripts.train_structure','--cache',str(a.root/'cache_v1'),'--out',str(path),'--modes',stage['mode'],'--seeds',*map(str,plan['seeds']),'--train-scene-count',str(condition['train_scene_count']),'--scene-residency','cpu','--save-initial-state','--code-sha',a.code_sha,'--aux-steps',str(stage['aux_steps']),'--dense-steps',str(stage['dense_steps']),'--aux-weight',str(stage['aux_weight']),'--fp-anchor-weight',str(stage['fp_anchor_weight']),'--skip-sampling-steps',str(stage['skip_sampling_steps'])]
   if stage['foundation_features']:cmd+=['--visual-cache',str(a.root/'features_v1')]
   if stage['pretrained_stage']:cmd+=['--pretrained-dir',str(completed[stage['pretrained_stage']])]
   record=dict(key=key,output=str(path),command=cmd,exit_code=None);state['attempts'].append(record);save(state_path,state);print('START',key,flush=True)
   with (a.out/f'{path.name}.log').open('x') as log:result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=False)
   record['exit_code']=result.returncode;save(state_path,state)
   if result.returncode:raise RuntimeError(f'{key} exited{result.returncode}; preserved at{path}')
   assert len(read(path/'summary.json'))==3
   completed[stage['stage']]=path;print('DONE',key,flush=True)
 print('COMPLETE',flush=True)
if __name__=='__main__':main()

"""Check completed scaling cells against the frozen plan before interpretation."""
import argparse,hashlib,json
from pathlib import Path
import torch

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def equal_state(a,b):
 x=torch.load(a,weights_only=True,map_location='cpu');y=torch.load(b,weights_only=True,map_location='cpu')
 assert x.keys()==y.keys() and all(torch.equal(x[k],y[k]) for k in x),f'Initial tensor mismatch: {a}, {b}'
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--condition');a=p.parse_args()
 plan=read(a.root/'plan_v1/plan.json');runner=read(a.run/'runner_protocol.json');state=read(a.run/'state.json')
 assert runner['plan_sha256']==sha(a.root/'plan_v1/plan.json')
 conditions=[c for c in plan['conditions'] if not a.condition or c['name']==a.condition];assert conditions
 rows=[];initials={};continuations=0
 for c in conditions:
  paths={}
  query_keys={(sid,q) for sid in c['val_scenes'] for q in torch.load(a.root/'cache_v1'/f'{sid}.pt',weights_only=True,map_location='cpu')['query_ids']}
  for stage in c['stages']:
   key=c['name']+'_'+stage['stage'];attempts=[r for r in state['attempts'] if r['key']==key]
   assert attempts and attempts[-1]['exit_code']==0,f'Await completed stage: {key}'
   # Executor paths are container-local; only the preserved basename is relocated.
   path=a.run/Path(attempts[-1]['output']).name;paths[stage['stage']]=path;protocol=read(path/'protocol.json')
   assert protocol['data']['scenes']=={'train':c['train_scenes'],'val':c['val_scenes']}
   assert protocol['cache_protocol_sha256']==plan['cache_protocol_sha256']
   assert protocol['lr']==.0003 and protocol['encoder_regime']=='joint' and protocol['arm']=='supervised' and not protocol['fixed_fp']
   assert protocol['seeds']==plan['seeds'] and protocol['scene_residency']=='cpu' and protocol['batch_size']==8
   for k in ['aux_steps','dense_steps','aux_weight','fp_anchor_weight']:assert protocol[k]==stage[k],(key,k)
   assert protocol['skip_sampling_steps']==stage['skip_sampling_steps'] if stage['pretrained_stage'] else stage['skip_sampling_steps']==0
   summary=read(path/'summary.json');assert len(summary)==len(plan['seeds'])
   for seed in plan['seeds']:
    prefix=f"{stage['mode']}_{seed}";rec=read(path/f'{prefix}.json');initial=path/f'{prefix}_initial.pt'
    assert rec['execution_code_sha']==runner['code_sha'] and rec['initial_state_sha256']==sha(initial)
    assert rec['history'][-1]['step']==stage['aux_steps']+stage['dense_steps']-1
    expected={(sid,q['query']) for sid in c['val_scenes'] for q in rec['val']['rows'] if q['scene']==sid and q['arm']=='normal'}
    normal=[r for r in rec['val']['rows'] if r['arm']=='normal'];assert expected==query_keys and len(expected)==len(normal)==len(c['val_scenes'])*64
    if stage['pretrained_stage']:
     prior=paths[stage['pretrained_stage']]/f'{prefix}_pretrained.pt';equal_state(initial,prior);assert protocol['pretrained_sha256'][prefix]==sha(prior);continuations+=1
    else:
     ik=(stage['mode'],seed)
     if ik in initials:equal_state(initial,initials[ik])
     else:initials[ik]=initial
    rows.append(dict(condition=c['name'],stage=stage['stage'],seed=seed,execution_code_sha=rec['execution_code_sha'],checkpoint_sha256=sha(path/f'{prefix}.pt'),initial_sha256=sha(initial),val_cases=len(normal)))
 a.out.mkdir(parents=True,exist_ok=False);result=dict(conditions=[c['name'] for c in conditions],fits=len(rows),continuation_initial_checks=continuations,all_cells_checked=len(conditions)==len(plan['conditions']),rows=rows)
 (a.out/'validation.json').write_text(json.dumps(result,indent=2));print('COMPLETE',len(rows),flush=True)
if __name__=='__main__':main()

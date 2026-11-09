"""Compare all four predeclared scaling cells after provenance verification."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--fit-validation',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 plan=read(a.root/'plan_v1/plan.json');names=[c['name'] for c in plan['conditions']]
 fit=read(a.fit_validation/'validation.json');assert fit['all_cells_checked'] and fit['conditions']==names and fit['fits']==36
 assert fit['continuation_initial_checks']==12
 data={};registries={};baseline_keys=None;strata=None
 methods=['visual','acoustic','fixed','fixed_audio_shuffle']
 for name in names:
  reg=a.root/f'{name}_registered_v1';run=a.root/f'{name}_evaluation_v1';v=read(reg/'validation.json');setup=read(run/'protocol.json')
  assert v['condition']==name and v['plan_sha256']==sha(a.root/'plan_v1/plan.json')
  assert v['predictions_sha256']==sha(run/'predictions.json') and v['evaluation_protocol_sha256']==sha(run/'protocol.json')
  assert v['rows']==14592 and v['unimodal_replay_checks']==7296
  assert setup['fixed_only'] and setup['extended_metrics'] and read(run/'selection.json')=={}
  hashes={f"{'visual' if r['stage']=='visual_joint' else 'acoustic'}_{r['seed']}":r['checkpoint_sha256'] for r in fit['rows'] if r['condition']==name and r['stage'] in ['visual_joint','acoustic']}
  assert setup['checkpoint_hashes']==hashes
  rows=read(run/'predictions.json');keys=[tuple(r[k] for k in ['scene','seed','query','method']) for r in rows]
  assert len(keys)==len(set(keys))==14592
  if baseline_keys is None:baseline_keys=set(keys);strata=v['strata']
  else:assert set(keys)==baseline_keys and strata==v['strata']
  data[name]=rows;registries[name]=sha(reg/'validation.json')
 # These match the frozen plan: diversity at fixed updates, updates at fixed
 # diversity, and expected draws/query matched across diversity levels.
 contrasts=[('n112_budget1','n16_budget1','data_at_budget1'),('n112_budget7','n16_budget7','data_at_budget7'),('n16_budget7','n16_budget1','budget_at_n16'),('n112_budget7','n112_budget1','budget_at_n112'),('n112_budget7','n16_budget1','matched_draws_per_query')]
 metrics=['joint','r1','fine','raw20','diverse20','nll','strict_05_15','strict_02_10']
 means={(name,method,scene,k):np.mean([r[k] for r in rows if r['method']==method and r['scene']==scene]) for name,rows in data.items() for method in methods for scene in strata['all19'] for k in metrics}
 rng=np.random.default_rng(20261109);paired=[]
 for left,right,label in contrasts:
  for stratum,scenes in strata.items():
   for method in methods:
    for metric in metrics:
     d=np.array([means[left,method,s,metric]-means[right,method,s,metric] for s in scenes]);scale=1 if metric=='nll' else 100;boot=d[rng.integers(len(d),size=(10000,len(d)))].mean(1)*scale
     paired.append(dict(contrast=label,left=left,right=right,stratum=stratum,method=method,metric=metric,delta=float(d.mean()*scale),ci95=np.quantile(boot,[.025,.975]).tolist(),units='nats' if metric=='nll' else 'pp',unit='paired scene; seeds averaged',multiplicity='uncorrected; secondary metrics and strata descriptive'))
 a.out.mkdir(parents=True,exist_ok=False)
 (a.out/'paired.json').write_text(json.dumps(paired,indent=2))
 (a.out/'validation.json').write_text(json.dumps(dict(conditions=names,matched_records_per_cell=len(baseline_keys),registries=registries,fit_validation_sha256=sha(a.fit_validation/'validation.json'),plan_sha256=sha(a.root/'plan_v1/plan.json'),evidence='Exploratory reused Val; no independent Test',test_accessed=False),indent=2))
 print('COMPLETE',len(paired),flush=True)
if __name__=='__main__':main()

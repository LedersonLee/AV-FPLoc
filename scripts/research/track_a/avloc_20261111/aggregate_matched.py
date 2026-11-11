"""Strict matched-cohort comparison; ray methods can be added only after complete evaluation."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from paired_summary import paired,align,matched_keys

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
 p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--ray',nargs=3,action='append',default=[],metavar=('NAME','SEED','EVALUATION_DIRECTORY'));a=p.parse_args();manifest=read(a.study/'data_v1/manifest.json');expected=[]
 for r in manifest['records']:
  if r['split']=='val':expected.extend((r['scene'],int(q)) for q in np.load(a.study/'data_v1'/r['scene']/'labels.npz')['query_ids'])
 assert len(expected)==9965 and len({s for s,q in expected})==19;methods={};sources=[]
 for seed in [42,43]:
  root=a.study/f'baselines/fields_seed{seed}';summary=read(root/'summary.json');assert summary['complete'] and summary['queries']==9965 and summary['scenes']==19 and summary['grid_cm']==20;sources.append(dict(kind='baseline',seed=seed,summary_sha256=sha(root/'summary.json')));rows=[]
  for rec in summary['records']:
   path=root/rec['file'];assert sha(path)==rec['sha256'];rows.extend(read(path))
  rows=align(rows,expected)
  for name in summary['methods']:methods.setdefault(name,{})[seed]=[r['methods'][name] for r in rows]
 root=a.study/'baselines/geoq_replay_v1';assert read(root/'manifest.json')['complete']
 for seed in [42,43]:
  path=root/f'seed{seed}_queries.json';rows=align(read(path),expected);methods.setdefault('B2_GEOQ',{})[seed]=rows;methods.setdefault('B2_Support',{})[seed]=[dict(joint=r['support_joint']) for r in rows];sources.append(dict(kind='GEOQ',seed=seed,queries_sha256=sha(path),manifest_sha256=sha(root/'manifest.json')))
 for name,seed,path in a.ray:
  seed=int(seed);root=Path(path);summary=read(root/'summary.json');assert summary['complete'] and summary['queries']==9965 and summary['grid_cm']==20;rows=[]
  for sid in sorted({s for s,q in expected}):rows.extend(read(root/f'{sid}.json'))
  assert seed not in methods.get(name,{});methods.setdefault(name,{})[seed]=align(rows,expected);sources.append(dict(kind='ray',name=name,seed=seed,summary_sha256=sha(root/'summary.json'),scene_files_sha256={sid:sha(root/f'{sid}.json') for sid in sorted({s for s,q in expected})}))
 scenes=np.array([s for s,q in expected]);results={};keys=['r1','joint','fine','top5_nms','top20_nms','top5_raw','top20_raw','ray_mae','in_fov_mae','out_fov_mae']
 for name,seeds in methods.items():
  record=dict(seeds=sorted(seeds),micro={},macro={},per_scene={})
  for key in keys:
   if not all(all(key in r and r[key] is not None for r in rows) for rows in seeds.values()):continue
   values=np.array([[r[key] for r in seeds[s]] for s in sorted(seeds)]).mean(0);per={sid:float(values[scenes==sid].mean()) for sid in sorted(set(scenes))};record['micro'][key]=float(values.mean());record['macro'][key]=float(np.mean(list(per.values())));record['per_scene'][key]=per
  baseline=np.array([r['joint'] for r in methods['visual'][42]]);record['paired_joint_vs_visual']=paired([[r['joint'] for r in seeds[s]] for s in sorted(seeds)],baseline,scenes);results[name]=record
 b2={s:np.array([r['joint'] for r in rows]) for s,rows in methods['b2'].items()};contrasts={}
 for name in methods:
  if set(methods[name])==set(b2):contrasts[name+'_vs_B2']=paired([[r['joint'] for r in methods[name][s]] for s in sorted(b2)],[b2[s] for s in sorted(b2)],scenes)
 detailed={}
 for model_name,baseline_name in [('fixed','visual'),('b2','fixed'),('B2_GEOQ','fixed'),('B2_GEOQ','b2'),('B2_GEOQ','B2_Support')]:
  for metric in ['r1','joint','fine']:
   mm=methods[model_name];bb=methods[baseline_name]
   if not all(metric in r for rr in [*mm.values(),*bb.values()] for r in rr):continue
   detailed[f'{model_name}_vs_{baseline_name}_{metric}']=paired([[r[metric] for r in mm[s]] for s in sorted(mm)],[[r[metric] for r in bb[s]] for s in sorted(bb)],scenes)
 a.out.mkdir(parents=True,exist_ok=False);result=dict(scope='matched20cm/10deg full Val19 reused development; seed-mean query estimates and paired scene bootstrap',complete_for_listed_methods=True,full_research_complete=False,queries=9965,scenes=19,results=results,paired_contrasts=contrasts,paired_metric_contrasts=detailed,sources=sources,source_sha256={n:sha(Path(__file__).with_name(n)) for n in ['aggregate_matched.py','paired_summary.py']},test_accessed=False);(a.out/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
 lines=['# Matched Val19 comparison (development)','', 'Not independent validation; pending AVLoc arms must not be interpreted as missing failures.','', '| Method | Seeds | R@1m | Joint | R@0.1m | Macro joint |','|---|---|---:|---:|---:|---:|']
 for name,r in results.items():
  show=lambda k:f"{r['micro'][k]*100:.3f}%" if k in r['micro'] else 'N/A'
  lines.append(f"| {name} | {r['seeds']} | {show('r1')} | {show('joint')} | {show('fine')} | {r['macro']['joint']*100:.3f}% |")
 (a.out/'comparison.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
if __name__=='__main__':main()

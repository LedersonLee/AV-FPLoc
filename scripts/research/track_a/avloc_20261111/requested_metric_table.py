"""Reaggregate saved final-pose errors at requested thresholds; never reselect poses."""
import argparse,json
from pathlib import Path
import numpy as np
from paired_summary import align
from predict_hash import sha

METRICS=['P≤0.1m','P≤0.5m','P≤1.0m','P≤1.0m & Y≤30°','Y≤10°','Y≤30°','Y≤50°']

def summarize(rows):
 p=np.array([r['position_error_m'] for r in rows]);y=np.array([r['yaw_error_deg'] for r in rows])
 assert len(rows)==9965 and np.isfinite(p).all() and np.isfinite(y).all() and (p>=0).all() and ((y>=0)&(y<=180)).all()
 flags=[p<=.1,p<=.5,p<=1,(p<=1)&(y<=30),y<=10,y<=30,y<=50]
 for name,f in [('fine',flags[0]),('r1',flags[2]),('joint',flags[3])]:np.testing.assert_array_equal(f,[r[name] for r in rows])
 return [float(f.mean()*100) for f in flags]


def main():
 p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--include-arm',action='append',choices=['A2_disabled','A3'],default=[]);a=p.parse_args();arms=['A0','A1','A2']+list(dict.fromkeys(a.include_arm))
 manifest=json.loads((a.study/'predictions/A2_seed42/manifest.json').read_text());expected=[]
 for r in manifest['records']:
  with np.load(a.study/'predictions/A2_seed42'/r['file']) as z:expected.extend((r['scene'],int(q)) for q in z['query_ids'])
 assert len(set(expected))==9965 and len({s for s,q in expected})==19
 sources={};tables={}
 def load(path):sources[str(path.relative_to(a.study))]=sha(path);return json.loads(path.read_text())
 def ray(arm,grid):
  assert load(a.study/f'runs/{arm}_seed42/complete.json')['status']=='training_complete'
  root=a.study/f'evaluation/{arm}_seed42_{grid}cm';summary=load(root/'summary.json');assert summary['complete'] and summary['queries']==9965 and summary['grid_cm']==grid
  return align([r for sid in sorted({s for s,q in expected}) for r in load(root/f'{sid}.json')],expected)
 base={}
 for seed in [42,43]:
  root=a.study/f'baselines/fields_seed{seed}';summary=load(root/'summary.json');assert summary['complete'] and summary['queries']==9965
  base[seed]=align([r for sid in sorted({s for s,q in expected}) for r in load(root/f'{sid}.json')],expected)
 entries=[]
 for key,label,seeds in [('visual','Existing Visual',[42]),('acoustic','FP-RIR Cosine',[42]),('fixed','Fixed Fusion',[42]),('b2','B2',[42,43])]:
  stats=[summarize([r['methods'][key] for r in base[seed]]) for seed in seeds];entries.append({'method':label,'seeds':seeds,'values':np.mean(stats,axis=0).tolist(),'per_seed':dict(zip(map(str,seeds),stats))})
 stats=[summarize(align(load(a.study/f'baselines/geoq_replay_v1/seed{seed}_queries.json'),expected)) for seed in [42,43]]
 entries.append({'method':'B2+GEOQ','seeds':[42,43],'values':np.mean(stats,axis=0).tolist(),'per_seed':dict(zip(['42','43'],stats))})
 for arm in arms:entries.append({'method':arm,'seeds':[42],'values':summarize(ray(arm,20))})
 tables['20cm']=entries;tables['10cm']=[{'method':arm,'seeds':[42],'values':summarize(ray(arm,10))} for arm in arms]
 result={'complete':True,'queries':9965,'scenes':19,'metrics':METRICS,'tables':tables,'source_sha256':sha(__file__),'sources':sources,'test_accessed':False,'scope':'Reused Val19 micro-averaged success percentages; B2/GEOQ average seeds42/43. Other methods seed42 or frozen cache label42. Inclusive thresholds with original floating-point errors and no epsilon, exactly reproducing stored metrics. Angular errors circular. No new inference/training.'}
 a.out.mkdir(parents=True,exist_ok=False);(a.out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
 lines=['# Requested final-pose metrics — 2026-11-12','','All values percent; higher is better. Val19:9,965 queries,19 scenes; reused development. P=position error; Y=circular yaw error. All thresholds inclusive. Angular-only columns do not constrain position. B2/GEOQ:mean of42/43; AVLoc-style arms:seed42. Frozen baseline cache label42 is not a new training replicate.','']
 for grid,rows in tables.items():
  lines += ['## '+grid+' XY /10° yaw','', '| Method | Seeds | '+' | '.join(METRICS)+' |','|---|---|'+ '|'.join(['---:']*7)+'|']
  for row in rows:lines.append('| '+row['method']+' | '+','.join(map(str,row['seeds']))+' | '+' | '.join(f'{v:.2f}' for v in row['values'])+' |')
  lines+=['']
 lines+=['The20cm table is the matched primary comparison. The10cm table changes pose sampling for the same frozen networks; it must not imply model superiority over20cm baselines. Only explicitly included, completed and evaluated arms appear; absence does not indicate zero performance. Seed43 A0 was interrupted and is excluded. No method was selected or retrained for these additional thresholds. Existing fine/r1/joint flags were reproduced exactly from stored errors for every query.','', 'Training code: `62c58da20ed26ba95863638bbfd4926829999eec`. Script: `requested_metric_table.py --study <study-root> --out <fresh-output>`. Source/output hashes in [summary](requested_metrics.json). Single seed and reused validation limit generalization; no Test17 access.']
 (a.out/'report.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
if __name__=='__main__':main()

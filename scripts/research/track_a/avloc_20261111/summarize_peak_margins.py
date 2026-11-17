"""Describe saved score margins on matched development queries; no selection."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from paired_summary import align
from predict_hash import sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 manifest=json.loads((a.study/'predictions/A3_seed42/manifest.json').read_text());expected=[];sources={};methods={}
 assert manifest['complete'] and manifest['queries']==9965
 for rec in manifest['records']:
  pp=a.study/'predictions/A3_seed42'/rec['file'];assert sha(pp)==rec['sha256']
  with np.load(pp) as z:expected.extend((rec['scene'],int(q)) for q in z['query_ids'])
 def load(root,method=None):
  summary=json.loads((root/'summary.json').read_text());assert summary['complete'] and summary['queries']==9965
  rows=[]
  for sid in sorted({s for s,q in expected}):
   p=root/f'{sid}.json';sources[str(p.relative_to(a.study))]=sha(p)
   rows.extend(json.loads(p.read_text()))
  rows=align(rows,expected)
  return [x['methods'][method] for x in rows] if method else rows
 for m in ['visual','acoustic','fixed','b2']:methods[m]=load(a.study/'baselines/fields_seed42',m)
 for m in ['A0','A1','A2','A2_disabled','A3']:methods[m]=load(a.study/f'evaluation/{m}_seed42_20cm')
 result={}
 for name,rows in methods.items():
  correct=np.array([x['joint'] for x in rows],bool);stats={}
  for key in ['raw_peak_margin','distinct_peak_margin','entropy_normalized','gt_region_mass']:
   values=np.array([x[key] for x in rows],float);assert np.isfinite(values).all()
   stats[key]={group:dict(count=int(mask.sum()),mean=float(values[mask].mean()),median=float(np.median(values[mask])),q10=float(np.quantile(values[mask],.1)),q90=float(np.quantile(values[mask],.9))) for group,mask in [('all',np.ones(len(rows),bool)),('success',correct),('failure',~correct)]}
   stats[key]['spearman_joint']=float(spearmanr(values,correct.astype(float)).statistic)
  result[name]=stats
 a.out.parent.mkdir(parents=True,exist_ok=True)
 with a.out.open('x') as f:json.dump(dict(complete=True,queries=9965,scenes=19,seed=42,grid_cm=20,methods=result,sources=sources,source_sha256=sha(__file__),test_accessed=False,scope='Reused development; descriptive within-method native-score margins. Different score units prevent interpreting raw cross-method magnitudes as confidence or information. GT masses and success strata are post-hoc diagnostics, never inference inputs.'),f,indent=2)
 print('COMPLETE',len(result),'methods, matched9965queries')
if __name__=='__main__':main()

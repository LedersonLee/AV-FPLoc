"""Paired scene summary for the predefined frozen-distribution 2x2 diagnostic."""
import argparse,json
from pathlib import Path
from paired_summary import align,paired
from predict_hash import sha
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 pred=json.loads((a.study/'predictions/A0_seed42/manifest.json').read_text());assert pred['complete'] and pred['queries']==9965
 expected=[]
 for rec in pred['records']:
  path=a.study/'predictions/A0_seed42'/rec['file'];assert sha(path)==rec['sha256'];expected.extend((rec['scene'],int(q)) for q in np.load(path)['query_ids'])
 assert len(expected)==9965 and len({s for s,q in expected})==19
 tags=['A0','fused_fov','visual_inside_audio_outside','independent_equal'];methods={};sources={};metrics={}
 for tag in tags:
  root=a.study/'evaluation'/f'{tag}_seed42_20cm';summary=json.loads((root/'summary.json').read_text());assert summary['complete'] and summary['queries']==9965 and summary['grid_cm']==20 and summary['temperature']==1
  rows=[];hashes={}
  for sid in sorted({s for s,q in expected}):
   path=root/f'{sid}.json';rows.extend(json.loads(path.read_text()));hashes[sid]=sha(path)
  methods[tag]=align(rows,expected);sources[tag]=dict(summary_sha256=sha(root/'summary.json'),rows_sha256=hashes);metrics[tag]=summary['metric_means']
 contrasts={};scene_ids=[s for s,q in expected]
 for model,base in [('visual_inside_audio_outside','A0'),('fused_fov','A0'),('independent_equal','fused_fov'),('independent_equal','visual_inside_audio_outside')]:
  contrasts[model+'_vs_'+base]={metric:paired([r[metric] for r in methods[model]],[r[metric] for r in methods[base]],scene_ids) for metric in ['joint','r1','fine','top20_nms']}
 a.out.mkdir(parents=True,exist_ok=False)
 result=dict(complete=True,queries=9965,scenes=19,metrics=metrics,contrasts=contrasts,sources=sources,source_sha256={n:sha(Path(__file__).with_name(n)) for n in ['summarize_fusion_decomposition.py','paired_summary.py']},scope='Exploratory fixed-checkpoint 2x2 inference diagnostic; reused Val19, seed42; not independently selected or training ablation',test_accessed=False)
 (a.out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:{m:{s:v[s] for s in ['micro_delta','macro_delta','macro_ci95','repair_count_seed_mean','harm_count_seed_mean']} for m,v in c.items()} for k,c in contrasts.items()},indent=2))
if __name__=='__main__':main()

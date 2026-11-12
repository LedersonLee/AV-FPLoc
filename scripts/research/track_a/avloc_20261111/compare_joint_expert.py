"""Matched A2 seed42 contrasts: fixed checkpoints, seeds averaged inside scenes."""
import argparse,json
from pathlib import Path
import numpy as np
from paired_summary import align,paired
from predict_hash import sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();sources={};methods={}
 m=json.loads((a.study/'predictions/A2_seed42/manifest.json').read_text());assert m['complete'] and m['queries']==9965 and m['seed']==42
 expected=[]
 for rec in m['records']:
  path=a.study/'predictions/A2_seed42'/rec['file'];assert sha(path)==rec['sha256'];expected.extend((rec['scene'],int(q)) for q in np.load(path)['query_ids'])
 assert len(expected)==9965 and len({s for s,q in expected})==19
 for tag in ['A2','A0','independent_equal','visual_inside_audio_outside']:
  root=a.study/'evaluation'/f'{tag}_seed42_20cm';summary=json.loads((root/'summary.json').read_text());assert summary['complete'] and summary['queries']==9965 and summary['grid_cm']==20
  rows=[];hashes={}
  for sid in sorted({s for s,q in expected}):
   path=root/f'{sid}.json';rows.extend(json.loads(path.read_text()));hashes[sid]=sha(path)
  methods[tag]=[align(rows,expected)];sources[tag]=dict(summary_sha256=sha(root/'summary.json'),rows_sha256=hashes)
 methods['B2_GEOQ']=[];sources['B2_GEOQ']={}
 for seed in [42,43]:
  path=a.study/f'baselines/geoq_replay_v1/seed{seed}_queries.json';methods['B2_GEOQ'].append(align(json.loads(path.read_text()),expected));sources['B2_GEOQ'][str(seed)]=sha(path)
 contrasts={};scenes=[s for s,q in expected]
 for name,rows in methods.items():
  if name=='A2':continue
  contrasts[name]={metric:paired([[r[metric] for r in rs] for rs in methods['A2']],[[r[metric] for r in rs] for rs in rows],scenes) for metric in ['r1','joint','fine']}
 a.out.mkdir(parents=True,exist_ok=False);result=dict(complete=True,queries=9965,scenes=19,contrasts=contrasts,sources=sources,source_sha256={n:sha(Path(__file__).with_name(n)) for n in ['compare_joint_expert.py','paired_summary.py']},scope='Single seed A2 versus seed42 ray controls and seed42/43 mean B2+GEOQ; reused Val19; not isolated causal interaction',test_accessed=False)
 (a.out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({n:{k:{x:v[x] for x in ['micro_delta','macro_delta','macro_ci95','repair_count_seed_mean','harm_count_seed_mean']} for k,v in rows.items()} for n,rows in contrasts.items()},indent=2))
if __name__=='__main__':main()

"""Paired grid-resolution intervention using exactly the same frozen predictions."""
import argparse,json
from pathlib import Path
import numpy as np
from paired_summary import align,paired
from predict_hash import sha


def main():
 p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 result={'complete':True,'seed':42,'queries':9965,'scenes':19,'methods':{},'test_accessed':False,
         'source_sha256':{n:sha(Path(__file__).with_name(n)) for n in ['compare_ray_grids.py','paired_summary.py']},
         'scope':'Same frozen prediction manifest per arm; 10cm minus20cm grid intervention. Reused Val19, single seed. No comparison to20cm baseline implies model superiority.'}
 for arm in ['A0','A1','A2']:
  pred=a.study/f'predictions/{arm}_seed42';manifest=json.loads((pred/'manifest.json').read_text());assert manifest['complete'] and manifest['queries']==9965
  expected=[]
  for r in manifest['records']:
   path=pred/r['file'];assert sha(path)==r['sha256']
   with np.load(path) as z:expected.extend((r['scene'],int(q)) for q in z['query_ids'])
  assert len(expected)==9965 and len(set(expected))==9965 and len({s for s,q in expected})==19
  grids={};summaries={};sources={}
  for grid in [20,10]:
   root=a.study/f'evaluation/{arm}_seed42_{grid}cm';summary=json.loads((root/'summary.json').read_text())
   assert summary['complete'] and summary['queries']==9965 and summary['grid_cm']==grid
   assert summary['prediction_manifest_sha256']==sha(pred/'manifest.json')
   rows=[];hashes={}
   for sid in sorted({s for s,q in expected}):
    path=root/f'{sid}.json';rows.extend(json.loads(path.read_text()));hashes[sid]=sha(path)
   grids[grid]=align(rows,expected);summaries[str(grid)]=summary['metric_means'];sources[str(grid)]={'summary_sha256':sha(root/'summary.json'),'rows_sha256':hashes}
  result['methods'][arm]={'metrics':summaries,'sources':sources,'prediction_manifest_sha256':sha(pred/'manifest.json'),
   'contrasts':{k:paired([r[k] for r in grids[10]],[r[k] for r in grids[20]],[s for s,q in expected]) for k in ['r1','joint','fine','top20_nms']}}
 a.out.mkdir(parents=True,exist_ok=False);(a.out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({arm:{k:{x:v[x] for x in ['micro_delta','macro_delta','macro_ci95']} for k,v in r['contrasts'].items()} for arm,r in result['methods'].items()},indent=2))
if __name__=='__main__':main()

"""Validate full diagnostic cohort and paired perfect-profile comparison, no model claim."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from paired_summary import paired,align

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();s=json.loads((a.input/'summary.json').read_text());assert s['complete'] and s['queries']==9965;manifest=json.loads((a.data/'manifest.json').read_text());expected=[]
 for r in manifest['records']:
  if r['split']=='val':expected.extend((r['scene'],int(q)) for q in np.load(a.data/r['scene']/'labels.npz')['query_ids'])
 rows=[]
 for r in s['records']:
  p=a.input/r['file'];assert sha(p)==r['sha256'];rows.extend(json.loads(p.read_text()))
 rows=align(rows,expected);scenes=[r['scene'] for r in rows];assert len(set(scenes))==19
 for name in ['omnidirectional','visual_fov']:
  values=[r[name] for r in rows];assert np.isclose(np.mean([v['perfect_profile_joint'] for v in values]),s['results'][name]['perfect_profile_joint'])
  for v in values:
   assert v['minimum_profile_mae_m']>=-1e-10
   if v['joint_geometry_margin_m'] is not None and v['joint_geometry_margin_m']>1e-10:assert v['perfect_profile_joint']
 delta=paired([r['omnidirectional']['perfect_profile_joint'] for r in rows],[r['visual_fov']['perfect_profile_joint'] for r in rows],scenes)
 result=dict(status='pass',queries=len(rows),scenes=19,grid_cm=s['grid_cm'],profile_joint_36_minus_11=delta,summary_sha256=sha(a.input/'summary.json'),source_sha256=sha(__file__),interpretation='Geometry information potential under perfect GT profiles, not evidence that RIR encodes/reconstructs them. Pairwise comparison changes direction count only under this diagnostic. Equal-scale result is not a learned-model upper bound.',test_accessed=False)
 a.out.write_text(json.dumps(result,indent=2)+'\n');print(delta['macro_delta'],delta['macro_ci95'])
if __name__=='__main__':main()

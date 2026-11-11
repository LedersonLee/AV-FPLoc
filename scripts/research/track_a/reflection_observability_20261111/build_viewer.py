"""Standalone interactive evidence viewer; only reduced display envelopes embedded."""
import argparse,json
from pathlib import Path
import numpy as np
from geometry import paths
from metrics import measure

def envelope(raw,scale=None):
 raw=raw[:,:3840].astype(float);scale=scale or max(abs(raw[:,48:]).max(),1e-25);bins=raw.reshape(6,480,8)
 return dict(lo=np.round(bins.min(-1)/scale,5).tolist(),hi=np.round(bins.max(-1)/scale,5).tolist(),scale=scale,bin_ms=8/48)
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();cases=[]
 controlled=json.loads((a.root/'controlled_analysis.json').read_text())['records']
 for r in controlled:
  if r['variant']!='base':continue
  pose=[r['center'][0],r['center'][2],0];base=np.load(a.root/f"controlled/{r['kind']}_d{r['distance']}_base/rir.npy");pp=paths(pose,np.array(r['base_segments']));
  for p in pp:p['metrics']=measure(base,p['times_ms'])
  wave={v:envelope(np.load(a.root/f"controlled/{r['kind']}_d{r['distance']}_{v}/rir.npy"),float(abs(base[:,48:3840]).max())) for v in ['base','repeat','removed','moved']}
  cases.append(dict(name=f"Controlled · {r['kind']} · {r['distance']}m",kind='controlled',segments=r['base_segments'],architecture=[],hypotheses=[dict(pose=pose,paths=pp)],wave=wave,note='Intervention identifies a wall-dependent response, not an isolated first-order path. A local window maximum is not a detected wall. Production-settings timing failure is retained.'))
 sources=sorted((a.root/'hssd').glob('*.json'))
 for ix in [0,6,11]:
  r=json.loads(sources[ix].read_text());q=r['queries'][0];raw=np.roll(np.load(q['raw_path']),q['roll'],0)
  for h in q['hypotheses']:
   for p in h['paths']:p['metrics']=measure(raw,p['times_ms'])
  cases.append(dict(name=f"HSSD · {r['scene']} · query {q['query_id']}",kind='hssd',segments=r['segments'],architecture=r['architecture'],hypotheses=q['hypotheses'],wave={'observed':envelope(raw)},note='Deterministic first query of scenes 1/7/12. Candidate geometry uses pure-visual Top20; no true pose is drawn or used. Gray native architectural slices audit FP support.'))
 data=json.dumps(cases,separators=(',',':'));template=Path(__file__).with_name('viewer_template.html').read_text();a.out.write_text(template.replace('__DATA__',data));print(a.out,a.out.stat().st_size)
if __name__=='__main__':main()

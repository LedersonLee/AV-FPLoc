import json,argparse,time
from pathlib import Path
import numpy as np
from geometry import paths
from metrics import measure,average

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();out=[]
 for dest in sorted((a.root/'controlled').iterdir()):
  if not (dest/'receipt.json').exists():continue
  r=json.loads((dest/'receipt.json').read_text());raw=np.load(dest/'rir.npy');pose=[r['center'][0],r['center'][2],0]
  pred=paths(pose,np.array(r['segments']).reshape(-1,2,2)) if r['segments'] else []
  basepred=paths(pose,np.array(r['base_segments']));front=[x for x in basepred if x['wall']==0][0]
  evaluated=[]
  for path in pred:
   if min(path['times_ms'])<1 or max(path['times_ms'])>80:continue
   evaluated.append(dict(**path,metrics={str(h):measure(raw,path['times_ms'],h) for h in [.25,.5,1.]}))
  old=measure(raw,front['times_ms']);frontmoved=paths(pose,np.array(r['base_segments'])+np.array([0,-1])) if r['kind']=='single' else []
  # Only front segment translated; prediction independent of received samples.
  moved=np.array(r['base_segments']);moved[0,:,1]-=1
  newfront=[x for x in paths(pose,moved) if x['wall']==0][0]
  oldpower=raw[np.arange(6)[:,None],np.rint(np.array(front['times_ms'])*48).astype(int)[:,None]+np.arange(-24,25)]**2
  newpower=raw[np.arange(6)[:,None],np.rint(np.array(newfront['times_ms'])*48).astype(int)[:,None]+np.arange(-24,25)]**2
  out.append(dict(**r,paths=evaluated,old_front_times_ms=front['times_ms'],moved_front_times_ms=newfront['times_ms'],old_front=old,new_front=measure(raw,newfront['times_ms']),old_front_absolute_power=float(oldpower.mean()),new_front_absolute_power=float(newpower.mean())))
 contrasts=[]
 for kind in ['single','room','corridor']:
  for d in [.3,2.,6.]:
   v={r['variant']:r for r in out if r['kind']==kind and r['distance']==d};b=v['base'];rep=v['repeat'];rm=v['removed'];mv=v['moved'];eps=1e-30
   contrasts.append(dict(kind=kind,distance=d,expected_shift_ms=float(np.mean(mv['moved_front_times_ms'])-np.mean(b['old_front_times_ms'])),base_peak_error_ms=b['old_front']['peak_error_ms'],moved_peak_error_ms=mv['new_front']['peak_error_ms'],removal_db=float(10*np.log10((b['old_front_absolute_power']+eps)/(rm['old_front_absolute_power']+eps))),repeat_change_db=float(10*np.log10((b['old_front_absolute_power']+eps)/(rep['old_front_absolute_power']+eps))),move_new_vs_old_db=float(10*np.log10((mv['new_front_absolute_power']+eps)/(mv['old_front_absolute_power']+eps))),old_after_move_db=float(10*np.log10((b['old_front_absolute_power']+eps)/(mv['old_front_absolute_power']+eps)))))
 (a.root/'controlled_analysis.json').write_text(json.dumps(dict(records=out,contrasts=contrasts),indent=2));print(json.dumps(contrasts,indent=2))
if __name__=='__main__':main()

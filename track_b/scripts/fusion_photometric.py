"""Frozen independent fusion under exactly the interaction brightness intervention."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from track_b.models.structure import StructureMatcher,visible_rays
from track_b.eval.metrics import pose_statistics

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def descriptors(m,s):
 f=m.floorplan(s['floorplan']);cal={'origin':s['origin'],'spacing':s['spacing']}
 return torch.cat([m.fp_descriptors(f,p,cal) for p in s['candidates'].split(8192)])
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--run',type=Path,required=True);ap.add_argument('--fusion-reference',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--device',choices=['cpu','cuda'],default='cpu');a=ap.parse_args();a.out.mkdir(exist_ok=False,parents=True);torch.set_num_threads(2);protocol=read(a.run/'protocol.json');reference={(r['scene'],r['seed'],r['query']):r for r in read(a.fusion_reference/'predictions.json') if r['method']=='fixed'};rows=[];replayed=0;mismatches=[]
 (a.out/'protocol.json').write_text(json.dumps(dict(arms=['normal','low_light'],brightness=.2,weights=[.85,.15],temperature=1,device=a.device,checkpoint_sha256={f'{m}_{seed}':sha(a.run/f'{m}_{seed}.pt') for m in ('visual','acoustic') for seed in protocol['seeds']},cache_manifest_sha256=sha(a.cache/'manifest.json'),evidence='Exploratory six reused development scenes; no fit or selection'),indent=2))
 with torch.no_grad():
  for seed in protocol['seeds']:
   models={}
   for mode in ('visual','acoustic'):
    m=StructureMatcher().eval();m.load_state_dict(torch.load(a.run/f'{mode}_{seed}.pt',weights_only=True,map_location='cpu'));models[mode]=m.to(a.device)
   for sid in protocol['data']['scenes']['val']:
    s=torch.load(a.cache/f'{sid}.pt',weights_only=True,map_location='cpu');s={k:v.to(a.device) if isinstance(v,torch.Tensor) else v for k,v in s.items()};fp={mode:descriptors(m,s) for mode,m in models.items()};weight=.25+.75*s['candidates'][:,19:35];mask=visible_rays(s['hfov_deg'],device=a.device);poses=s['candidates'][:,:3]
    for ids in torch.arange(len(s['gt']),device=a.device).split(8):
     acoustic=models['acoustic'].acoustic(s['spectrum'][ids].float());sa=models['acoustic'].match(acoustic,fp['acoustic'],weight)
     for arm in ('normal','low_light'):
      rgb=s['rgb'][ids].float()/255
      if arm=='low_light':rgb=rgb*.2
      visual=models['visual'].visual(rgb);sv=models['visual'].match(visual,fp['visual'],weight*mask)
      for method,field in [('visual',sv),('acoustic',sa),('fixed_fusion',.85*sv+.15*sa)]:
       for j,q in enumerate(ids.tolist()):
        metric=pose_statistics(field[j],poses,s['gt'][q],diverse=False);row=dict(scene=sid,seed=seed,query=s['query_ids'][q],arm=arm,method=method,**metric);rows.append(row)
        if arm=='normal' and method=='fixed_fusion':
         old=reference[sid,seed,row['query']];differences={k:dict(current=metric[k],reference_gpu=old[k]) for k in ('joint','r1','fine','raw20','nll') if not np.isclose(metric[k],old[k],atol=2e-5,rtol=1e-5)}
         if differences:mismatches.append(dict(scene=sid,seed=seed,query=row['query'],differences=differences))
         replayed+=1
    print('SCENE',seed,sid,flush=True)
 summary=[dict(method=m,arm=arm,**{k:float(np.mean([r[k] for r in rows if r['method']==m and r['arm']==arm])) for k in ('joint','r1','fine','raw20','nll','entropy')}) for m in ('visual','acoustic','fixed_fusion') for arm in ('normal','low_light')]
 (a.out/'predictions.json').write_text(json.dumps(rows));result=dict(summary=summary,normal_replay_checks=replayed,device=a.device,replay_mismatches=mismatches,interpretation='Matched brightness severity; no physical night capture claim; same acoustic input in both arms.');(a.out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));print('COMPLETE',flush=True)
if __name__=='__main__':main()

"""Inspect a declared CPU/GPU disagreement without selecting a new model."""
import argparse,json
from pathlib import Path
import torch
from track_b.models.structure import StructureMatcher,visible_rays
from track_b.scripts.fusion_photometric import descriptors

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--run',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--device',default='cpu');ap.add_argument('--batch-size',type=int,default=8);a=ap.parse_args();torch.set_num_threads(2);sid='108294783_176710410';query=1298;seed=44;reports=[]
 for dtype in (torch.float32,torch.float64):
  s=torch.load(a.cache/f'{sid}.pt',weights_only=True,map_location='cpu');s={k:(v.to(device=a.device,dtype=dtype) if v.is_floating_point() else v.to(a.device)) if isinstance(v,torch.Tensor) else v for k,v in s.items()};idx=s['query_ids'].index(query);start=(idx//a.batch_size)*a.batch_size;stop=min(start+a.batch_size,len(s['gt']));offset=idx-start;rgb=s['rgb'][start:stop].to(dtype)/255;spec=s['spectrum'][start:stop].to(dtype);fields={};w=.25+.75*s['candidates'][:,19:35]
  with torch.no_grad():
   for mode in ('visual','acoustic'):
    m=StructureMatcher().to(device=a.device,dtype=dtype).eval();m.load_state_dict(torch.load(a.run/f'{mode}_{seed}.pt',weights_only=True,map_location=a.device));fp=descriptors(m,s)
    if mode=='visual':observed=m.visual(rgb);weight=w*visible_rays(s['hfov_deg'],device=a.device)
    else:observed=m.acoustic(spec);weight=w
    fields[mode]=m.match(observed,fp,weight)[offset]
   field=.85*fields['visual']+.15*fields['acoustic'];best=field.topk(5);items=[]
   for rank,candidate in enumerate(best.indices.tolist()):
    d=s['candidates'][candidate,:3]-s['gt'][idx];distance=d[:2].norm();d64=s['candidates'][candidate,:3].double()-s['gt'][idx].double();distance64=d64[:2].norm();yaw=torch.atan2(d[2].sin(),d[2].cos()).abs()
    items.append(dict(rank=rank,candidate=candidate,score=float(field[candidate]),position_distance_native=float(distance),position_distance_float64=float(distance64),yaw_degrees=float(torch.rad2deg(yaw)),position_pass_native=bool(distance<=1+1e-6),position_pass_float64=bool(distance64<=1+1e-6),pose=s['candidates'][candidate,:3].tolist(),gt=s['gt'][idx].tolist()))
  reports.append(dict(dtype=str(dtype),device=a.device,batch_size=stop-start,target_batch_offset=offset,top5=items,top1_top2_margin=float(best.values[0]-best.values[1])))
 a.out.write_text(json.dumps(dict(scene=sid,query=query,seed=seed,scope='one predeclared mismatch case; float64 uses stored float32 coordinates, not unrounded original labels',reports=reports),indent=2));print(json.dumps(reports),flush=True)
if __name__=='__main__':main()

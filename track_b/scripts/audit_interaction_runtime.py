"""Trace actual observation-cache/checkpoint inference; labels are not an input."""
import argparse,hashlib,json,sys
from pathlib import Path
import torch
from track_b.models.interaction import InteractionMatcher
from track_b.models.dense import rasterize

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--scene',type=Path,required=True);ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();torch.set_num_threads(2);events=[]
 def hook(event,args):
  if event=='open' and isinstance(args[0],(str,bytes)):events.append(str(args[0]))
 sys.addaudithook(hook);s=torch.load(a.scene,weights_only=True,map_location='cpu');m=InteractionMatcher('attention').eval();m.load_state_dict(torch.load(a.checkpoint,weights_only=True,map_location='cpu'));load_opens=list(events);events.clear()
 inputs=(s['rgb'][:1].float()/255,s['spectrum'][:1].float(),s['floorplan'],{'origin':s['origin'],'spacing':s['spacing'],'hfov_deg':s['hfov_deg']},s['candidates'])
 with torch.no_grad():
  field=m(*inputs);s.pop('gt');s.pop('target');s.pop('target_hit');again=m(*inputs);torch.testing.assert_close(field,again,atol=0,rtol=0)
  raster=rasterize(field,s['valid'],36);assert torch.isfinite(raster).sum()==field.numel();assert field.shape[-1]==len(s['candidates'])
 forward_opens=list(events);assert not forward_opens,f'Unexpected forward file I/O: {forward_opens}'
 suspicious=[p for p in load_opens if any(t in p.lower() for t in ('/rir/fp/','/track_a/','/track-a/','b2','geoq'))];assert not suspicious
 result=dict(scene=s['scene'],queries=1,candidates=len(s['candidates']),checkpoint_sha256=sha(a.checkpoint),scene_cache_sha256=sha(a.scene),label_removal_bitwise_equal=True,full_raster_finite_count=int(torch.isfinite(raster).sum()),load_open_paths=load_opens,forward_open_paths=forward_opens,forbidden_reference_paths=suspicious,scope='Actual completed attention checkpoint, one held-out development scene, one full-bank query, CPU audit; no inference file opens after inputs loaded. Broader entrypoints/other scenes not traced here.')
 a.out.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()

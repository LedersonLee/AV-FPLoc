"""Profile exact full-bank backward for all three equal-parameter controls."""
import argparse,json,time
from pathlib import Path
import torch
from track_b.models.interaction import InteractionMatcher
from track_b.models.structure import auxiliary_loss
from track_b.models.dense import dense_loss
from track_b.scripts.train_structure import scene_to_gpu

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();torch.set_num_threads(2)
 manifest=json.loads((a.cache/'manifest.json').read_text());row=max((r for r in manifest if r['split']=='train'),key=lambda r:r['candidates']);scene=scene_to_gpu(torch.load(a.cache/f"{row['scene']}.pt",weights_only=True));results=[]
 for mode in ('attention','concat','separable'):
  torch.manual_seed(42);m=InteractionMatcher(mode).cuda();opt=torch.optim.AdamW(m.parameters(),lr=.0003);cal={'origin':scene['origin'],'spacing':scene['spacing']};torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
  field,v,au=m(scene['rgb'][:8].float()/255,scene['spectrum'][:8].float(),scene['floorplan'],cal,scene['candidates'],return_aux=True)
  loss=dense_loss(field,scene['candidates'][:,:3],scene['gt'][:8])+10*auxiliary_loss(v,au,scene['target'][:8]);loss.backward();opt.step();torch.cuda.synchronize()
  grad={name:sum(float(p.grad.abs().sum()) for p in branch.parameters() if p.grad is not None) for name,branch in [('visual',m.visual),('acoustic',m.acoustic),('floorplan',m.floorplan),('visual_retrieval',m.v_retrieval),('acoustic_retrieval',m.a_retrieval),('joint_head',m.joint_head)]};assert all(x>0 for x in grad.values())
  results.append(dict(mode=mode,parameters=sum(p.numel() for p in m.parameters()),scene=row['scene'],candidates=len(scene['candidates']),batch=8,seconds=time.perf_counter()-start,peak_vram_bytes=torch.cuda.max_memory_allocated(),gradient_sums=grad,loss=float(loss),scope='one first-step full-bank backward; no warm steady-state or whole-cache claim'));print(json.dumps(results[-1]),flush=True);del m,opt,field,v,au,loss;torch.cuda.empty_cache()
 a.out.write_text(json.dumps(results,indent=2))
if __name__=='__main__':main()

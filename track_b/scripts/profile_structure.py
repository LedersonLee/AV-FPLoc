"""Full-bank backward resource check on largest chosen training scene."""
import argparse,json,time
from pathlib import Path
import torch
from track_b.models.structure import StructureMatcher,auxiliary_loss
from track_b.models.dense import dense_loss
from track_b.scripts.train_structure import scene_to_gpu,full_descriptors,scores,observe


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();torch.set_num_threads(2);torch.manual_seed(42)
    manifest=json.loads((a.cache/'manifest.json').read_text());row=max((r for r in manifest if r['split']=='train'),key=lambda r:r['candidates'])
    scene=scene_to_gpu(torch.load(a.cache/f"{row['scene']}.pt",weights_only=True));model=StructureMatcher().cuda();opt=torch.optim.AdamW(model.parameters(),lr=.0003)
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter();history=[]
    for i in range(5):
        ids=torch.arange(8,device='cuda');v,b=observe(model,scene,ids,'fusion');fp=full_descriptors(model,scene,True);s=scores(model,v,b,fp,scene,'fusion')
        loss=dense_loss(s,scene['candidates'][:,:3],scene['gt'][ids])+10*auxiliary_loss(v,b,scene['target'][ids]);opt.zero_grad();loss.backward()
        grad={name:sum(float(p.grad.abs().sum()) for p in branch.parameters() if p.grad is not None) for name,branch in [('visual',model.visual),('acoustic',model.acoustic),('floorplan',model.floorplan)]}
        assert all(x>0 for x in grad.values());opt.step();history.append(float(loss))
    torch.cuda.synchronize();result=dict(scene=row['scene'],candidates=len(fp),batch=8,steps=5,seconds=time.perf_counter()-start,peak_vram_bytes=torch.cuda.max_memory_allocated(),
        gradient_sums=grad,losses=history,parameter_count=sum(p.numel() for p in model.parameters()),scope='one largest training scene resident; full cache residency adds memory')
    a.out.write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()

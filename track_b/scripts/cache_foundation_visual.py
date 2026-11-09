"""Cache fixed upstream RGB features, never historical HSSD visual-head weights."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import cv2
import torch
from torch.nn import functional as F

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--source',type=Path,required=True);ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--device',choices=['cpu','cuda'],default='cuda');ap.add_argument('--smoke',action='store_true');a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
 expected='6f82ff2bc543ac02ddff4aa31fa363676a8305dd3ccf04e80e2af115a044cb6d';digest=sha(a.checkpoint);assert digest==expected,'Unexpected upstream checkpoint; stop for provenance review'
 sys.path.insert(0,str(a.source));from depth_anything_v2.dpt import DepthAnythingV2
 m=DepthAnythingV2(encoder='vitl',features=256,out_channels=[256,512,1024,1024]);m.load_state_dict(torch.load(a.checkpoint,weights_only=True,map_location='cpu'),strict=True);m=m.to(a.device).eval();m.requires_grad_(False);data=json.loads((a.cache/'protocol.json').read_text());selected={'train':data['scenes']['train'][:1]} if a.smoke else data['scenes'];protocol=dict(data=data,selected_scenes=selected,smoke=a.smoke,device=a.device,backbone='DepthAnythingV2 ViT-L upstream metric Hypersim checkpoint; historical HSSD-trained UnLoc head NOT loaded',checkpoint_sha256=digest,source_hashes={str(p.relative_to(a.source)):sha(p) for p in sorted((a.source/'depth_anything_v2').rglob('*.py'))},preprocessing='upstream image2tensor: BGR->RGB, cubic resize shorter side518/multiple14, ImageNet normalization',features='last DINOv2 block reshape=True, adaptive pool4x16, fp16 cache',training_boundary='upstream encoder frozen; future adapter trainable; not full-foundation joint tuning',evidence='Exploratory development; upstream HSSD asset-family exposure not independently excluded; no Test observations');(a.out/'protocol.json').write_text(json.dumps(protocol,indent=2));manifest=[]
 with torch.no_grad():
  for split,sids in selected.items():
   assert split in ('train','val')
   for sid in sids:
    inputs=json.loads((a.cache/f'{sid}_inputs.json').read_text());inputs=inputs[:1] if a.smoke else inputs;features=[];shapes=[]
    if a.device=='cuda':torch.cuda.synchronize()
    start=time.perf_counter()
    for r in inputs:
     p=a.root/r['rgb_path'];assert sha(p)==r['rgb_sha256'];raw=cv2.imread(str(p));assert raw is not None;image,_=m.image2tensor(raw);image=image.to(a.device);z=m.pretrained.get_intermediate_layers(image,reshape=True)[0];feature=F.adaptive_avg_pool2d(z,(4,16)).half().cpu()[0];assert feature.shape==(1024,4,16) and torch.isfinite(feature).all();features.append(feature);shapes.append(list(image.shape))
    if a.device=='cuda':torch.cuda.synchronize()
    dest=a.out/f'{sid}.pt';torch.save(dict(scene=sid,split=split,query_ids=[r['query_id'] for r in inputs],visual_features=torch.stack(features)),dest);row=dict(scene=sid,split=split,queries=len(features),seconds=time.perf_counter()-start,cache_bytes=dest.stat().st_size,sha256=sha(dest),input_manifest_sha256=sha(a.cache/f'{sid}_inputs.json'),transformed_image_shapes=sorted({tuple(x) for x in shapes}));manifest.append(row);(a.out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(row),flush=True)
 print('COMPLETE',flush=True)
if __name__=='__main__':main()

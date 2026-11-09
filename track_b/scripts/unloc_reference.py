"""Isolated pretrained UnLoc comparator; never imported by a proposed scorer.

Uses the user's existing upstream checkout and checkpoint, preserving all files.
It shares candidate IDs/cohort with Track B but retains upstream exposure caveats.
"""
import argparse,hashlib,json,math,os,sys,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from track_b.data.rays import boundary_rays
from track_b.eval.metrics import pose_statistics


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def angular_prediction(depth,scale):
    angles=np.deg2rad(np.arange(-5,6)*10);column=np.tan(angles)*len(depth)*3/8+(len(depth)-1)/2
    return np.interp(column,np.arange(len(depth)),depth)/np.cos(angles),np.interp(column,np.arange(len(scale)),scale)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--unloc',type=Path,required=True);ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);sys.path.insert(0,str(a.unloc.resolve()));original=Path.cwd();os.chdir(a.unloc)
    from modules.depth_net import UnLocDepthNet
    model=UnLocDepthNet().cuda().eval();ckpt=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    weights={k[len('encoder.'):]:v for k,v in ckpt['state_dict'].items() if k.startswith('encoder.')};model.load_state_dict(weights,strict=True);del weights,ckpt;os.chdir(original)
    protocol=json.loads((a.cache/'protocol.json').read_text());meta=dict(checkpoint_sha256=sha(a.checkpoint),depth_anything_sha256=sha(a.root/'UnLoc/pretrained/depth_anything_v2_metric_hypersim_vitl.pth'),
        external_sources={str(p.relative_to(a.unloc)):sha(p) for p in (a.unloc/'modules').glob('*.py')},
        exposure='HSSD visual teacher Train116/Val19 exposure; development comparator, not pristine held-out',
        method='UnLoc checkpoint, shared 0.25m/36 candidate bank; footprint boundary rays0.1m step/max100m; original clamped Laplace and unclamped control',
        main_pipeline_dependency=False,test_accessed=False)
    (a.out/'provenance.json').write_text(json.dumps(meta,indent=2));rows=[];timings=[]
    with torch.inference_mode():
      for sid in protocol['scenes']['val']:
        scene=torch.load(a.cache/f'{sid}.pt',weights_only=True);poses=scene['candidates'][:,:3].cuda();queries=json.loads((a.cache/f'{sid}_inputs.json').read_text())
        m=json.loads((a.root/f'2DFP/{sid}/floorplan_metadata.json').read_text());free=np.asarray(Image.open(a.root/f'2DFP/{sid}/map.png'))==255
        origin=[m['world_bounds_m'][k] for k in ('x_min','z_min')];start=time.perf_counter()
        ranges,hits=boundary_rays(free,m['map_spacing_m'],origin,poses[::36,:2].cpu().numpy(),36,36,max_range=100.)
        rays=torch.tensor(ranges[:,np.arange(-5,6)%36],device='cuda');setup=time.perf_counter()-start
        timings.append(dict(scene=sid,candidates=len(poses),geometry_setup_seconds=setup,geometry_cache_bytes=rays.numel()*rays.element_size(),censored_fraction=float(1-hits.mean())))
        for i,q in enumerate(queries):
            image=np.asarray(Image.open(a.root/q['rgb_path']).convert('RGB'),dtype=np.float32)/255
            x=torch.tensor(((image-np.array([.485,.456,.406]))/np.array([.229,.224,.225])).transpose(2,0,1),device='cuda',dtype=torch.float32)[None]
            mask=torch.ones((1,480,640),dtype=torch.uint8,device='cuda');torch.cuda.synchronize();start=time.perf_counter()
            with torch.autocast('cuda',dtype=torch.float16):mu,b,_,_=model(x,mask)
            torch.cuda.synchronize();elapsed=time.perf_counter()-start
            d,s=angular_prediction(mu[0].float().cpu().numpy(),b[0].float().cpu().numpy());d=torch.tensor(d,device='cuda');s=torch.tensor(s,device='cuda').clamp_min(1e-6)
            normalized=(rays-d).abs()/s
            for variant in ('clamped','unclamped'):
                cost=normalized.clamp(max=5) if variant=='clamped' else normalized
                score=-cost.sum(-1)/40
                rows.append(dict(scene=sid,query=q['query_id'],variant=variant,rgb_forward_seconds=elapsed,**pose_statistics(score,poses,scene['gt'][i].cuda())))
        print('SCENE_DONE',sid,flush=True);(a.out/'predictions.json').write_text(json.dumps(rows,indent=2))
    summary={variant:{k:float(np.mean([r[k] for r in rows if r['variant']==variant])) for k in ('joint','r1','fine','raw20','diverse20','nll','entropy')} for variant in ('clamped','unclamped')}
    (a.out/'summary.json').write_text(json.dumps(dict(summary=summary,timings=timings),indent=2));print(json.dumps(summary),flush=True)

if __name__=='__main__':main()

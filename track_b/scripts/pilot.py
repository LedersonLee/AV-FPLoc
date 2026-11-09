"""Small from-scratch full-field development pilot; never accesses test data.

This deliberately measures feasibility/underfitting, not final architecture claims.
"""
import argparse, csv, hashlib, json, math, time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from track_b.data.geometry import candidate_bank,canonical_rir
from track_b.models.dense import DenseScorer,dense_loss,topk


def jread(p):return json.loads(p.read_text())

def materialize(root,scene,split,n):
    sch=jread(root/f'RIR/manifests/positive_pairs_v1/schedules/{split}/{scene}.json')
    meta=jread(root/f'2DFP/{scene}/floorplan_metadata.json');free=np.array(Image.open(root/f'2DFP/{scene}/map.png'))==255
    origin=np.array([meta['world_bounds_m'][k] for k in ('x_min','z_min')]);spacing=meta['map_spacing_m']
    bank,_=candidate_bank(free,spacing,origin,.5,12)
    # Exact metric raster subsampling, 0.10m feature grid. Bank remains derived from original FP.
    stride=round(.1/spacing);small=free[::stride,::stride]
    fp=torch.tensor(np.stack([small,~small]),dtype=torch.float32)[None].cuda()
    p=torch.tensor(bank,dtype=torch.float32).cuda();cal={'origin':torch.tensor(origin,dtype=torch.float32).cuda(),'spacing':spacing*stride}
    eligible=[q for q in sch['queries'] if (root/q['observed_rir_path']).is_file() and (root/q['image_path']).is_file()]
    ids=np.linspace(0,len(eligible)-1,min(n,len(eligible))).round().astype(int)
    data=[]
    for i in ids:
        q=eligible[i];im=np.asarray(Image.open(root/q['image_path']).convert('RGB').resize((64,48)),dtype=np.float32).copy()/255
        r=canonical_rir(np.load(root/q['observed_rir_path']),q['world_yaw_deg'],q['base_acoustic_yaw_deg'])
        audio=np.zeros((6,8192),np.float32);audio[:,:min(8192,r.shape[1])]=r[:,:8192]
        audio/=max(float(np.sqrt(np.mean(audio**2))),1e-8)
        gt=torch.tensor([q['centre_xyz_m'][0],q['centre_xyz_m'][2],math.radians(q['world_yaw_deg'])],dtype=torch.float32).cuda()[None]
        data.append((torch.tensor(im.transpose(2,0,1)).cuda()[None],torch.tensor(audio).cuda()[None],gt,q['sample_index']))
    return dict(scene=scene,fp=fp,p=p,cal=cal,data=data)


def metrics(s,p,gt):
    d=p-gt;dist=d[:,:2].norm(dim=-1);angle=torch.atan2(d[:,2].sin(),d[:,2].cos()).abs();ok=(dist<=1)&(angle<=math.pi/6+1e-6)
    idx=s.argmax();raw=topk(s,p);div=topk(s,p,radius=1.)
    return dict(joint=float(ok[idx]),r1=float(dist[idx]<=1),fine=float(dist[idx]<=.1),raw20=float(ok[raw].any()),diverse20=float(ok[div].any()),
        oracle_joint=float(ok.any()),oracle_fine=float((dist<=.1).any()),nll=float(dense_loss(s[None],p,gt)),entropy=float(-(s.softmax(0)*s.log_softmax(0)).sum()))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--audit',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--steps',type=int,default=120);args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
    split=jread(args.audit/'split_manifest.json')['development']
    order=lambda s:hashlib.sha256(s.encode()).hexdigest()
    train_ids=sorted(split['train'],key=order)[:4];val_ids=sorted(split['val'],key=order)[:3]
    protocol=dict(train=train_ids,val=val_ids,queries_per_scene=16,steps=args.steps,seeds=[42,43,44],modes=['visual','acoustic','concat','noninteraction','joint'],
        step_m=.5,yaw_bins=12,fp_spacing=.1,rgb_size=[64,48],rir_samples=8192,sample_rate=48000,chunk=512,
        selection='SHA256(scene ID) ordering; uniform schedule query subsample before any performance',evidence='Exploratory; existing development scenes, tiny from-scratch feasibility',
        capacity_warning='Identical allocated modules; active paths differ, not a capacity-matched causal test',regime='joint from scratch; frozen/partial comparisons pending',
        test_used=False,torch=torch.__version__)
    (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2));print('MATERIALIZING',flush=True)
    train=[materialize(args.root,s,'train',16) for s in train_ids];val=[materialize(args.root,s,'val',16) for s in val_ids]
    results=[]
    for seed in protocol['seeds']:
      for mode in protocol['modes']:
        torch.manual_seed(seed);rng=np.random.default_rng(seed);model=DenseScorer(mode=mode).cuda();opt=torch.optim.Adam(model.parameters(),lr=.001)
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter();history=[]
        model.train()
        for step in range(args.steps):
            scene=train[int(rng.integers(len(train)))];rgb,rir,gt,_=scene['data'][int(rng.integers(len(scene['data'])))];opt.zero_grad()
            score=model(rgb,rir,scene['fp'],scene['cal'],scene['p']);loss=dense_loss(score,scene['p'],gt)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            if step%20==0:history.append(dict(step=step,loss=float(loss)))
        torch.cuda.synchronize();elapsed=time.perf_counter()-start;peak=torch.cuda.max_memory_allocated();model.eval();pred=[]
        with torch.no_grad():
          for scene in val:
            for i,(rgb,rir,gt,qid) in enumerate(scene['data']):
              for intervention in ('normal','shuffle','zero'):
                audio=rir if intervention=='normal' else scene['data'][(i+1)%len(scene['data'])][1] if intervention=='shuffle' else torch.zeros_like(rir)
                s=model(rgb,audio,scene['fp'],scene['cal'],scene['p'])[0]
                pred.append(dict(scene=scene['scene'],query=qid,intervention=intervention,**metrics(s,scene['p'],gt)))
        record=dict(mode=mode,seed=seed,steps=args.steps,training_seconds=elapsed,peak_vram_bytes=peak,history=history,
                    parameter_count=sum(p.numel() for p in model.parameters()),predictions=pred)
        path=args.out/f'{mode}_{seed}.json';path.write_text(json.dumps(record,indent=2));torch.save(model.state_dict(),args.out/f'{mode}_{seed}.pt')
        result={k:v for k,v in record.items() if k not in ('history','predictions')}
        for intervention in ('normal','shuffle','zero'):
            rr=[r for r in pred if r['intervention']==intervention]
            result[intervention]={k:float(np.mean([r[k] for r in rr])) for k in ('joint','r1','fine','raw20','diverse20','oracle_joint','oracle_fine','nll','entropy')}
        results.append(result);(args.out/'summary.json').write_text(json.dumps(results,indent=2));print(json.dumps(result),flush=True)
        del model,opt
    print('COMPLETE',flush=True)

if __name__=='__main__':main()

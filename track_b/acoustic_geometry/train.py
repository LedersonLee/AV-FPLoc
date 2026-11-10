"""Fixed-budget, scene-balanced representation fits; no test-based selection."""
import argparse
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from .models import GeometryNet,masked_loss
from .prepare_targets import read,sha


def donors(poses):
    xy=poses[:,:2].numpy();result=[]
    for i in range(len(xy)):
        choices=[(i+k)%len(xy) for k in range(1,len(xy)) if np.linalg.norm(xy[(i+k)%len(xy)]-xy[i])>.5]
        if not choices:raise ValueError('No different-position donor >0.5m')
        # Closest heading among permissible positions avoids confounding yaw replacement.
        angles=poses[choices,2].numpy()-float(poses[i,2])
        result.append(choices[np.argmin(np.abs(np.arctan2(np.sin(angles),np.cos(angles))))])
    return torch.tensor(result)


@torch.no_grad()
def evaluate(model,scenes,device,phase_zero=False):
    model.eval();rows=[];predictions={}
    for scene in scenes:
      x=scene['echo' if model.architecture=='echo' else 'spatial'].float()
      if phase_zero:x[:,6:]=0
      donor=donors(scene['poses'])
      for condition in ['normal','same_scene_different_position']:
        xx=x if condition=='normal' else x[donor]
        means=[];logits=[]
        for batch in xx.split(16):
            m,l=model(batch.to(device));means.append(m.cpu())
            if l is not None:logits.append(l.cpu())
        m=torch.cat(means);target=scene['depth'];mask=scene['mask'];err=m-target
        row=dict(scene=scene['scene'],condition=condition,mae=float(err.abs()[mask].mean()),
                 rmse=float(err.square()[mask].mean().sqrt()),valid=int(mask.sum()))
        correlations=[]
        for d in range(36):
            good=mask[:,d];a=m[good,d];b=target[good,d]
            if len(a)>2 and a.std()>1e-7 and b.std()>1e-7:
                correlations.append(float(torch.corrcoef(torch.stack([a,b]))[0,1]))
        row['direction_correlation']=float(np.mean(correlations)) if correlations else None
        save=dict(mean=m.numpy(),target=target.numpy(),mask=mask.numpy(),donor=donor.numpy())
        if logits:
            l=torch.cat(logits);lp=l.log_softmax(-1);p=lp.exp()
            label=(target*model.bins/20).long().clamp(0,model.bins-1)
            row['categorical_nll']=float(-lp.gather(-1,label[...,None])[...,0][mask].mean())
            cdf=p.cumsum(-1);observed=(model.centres[None,None]>=target[...,None]).float()
            row['crps_m']=float(((cdf-observed).square().sum(-1)*20/model.bins)[mask].mean())
            lo=(cdf<.05).sum(-1).clamp_max(model.bins-1)*20/model.bins
            hi=((cdf<.95).sum(-1).clamp_max(model.bins-1)+1)*20/model.bins
            row['coverage90']=float(((target>=lo)&(target<=hi))[mask].float().mean())
            row['width90_m']=float((hi-lo)[mask].mean());save['probability']=p.numpy()
        rows.append(row);predictions[scene['scene']+'__'+condition]=save
    return rows,predictions


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--features',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--targets',type=Path,required=True)
    p.add_argument('--steps',type=int,default=1500);p.add_argument('--seeds',nargs='+',type=int,default=[42,43,44])
    p.add_argument('--architectures',nargs='+',default=['echo','spatial'])
    p.add_argument('--heads',nargs='+',default=['regression','categorical'])
    p.add_argument('--code-sha',required=True);p.add_argument('--device',default='cuda')
    p.add_argument('--phase-zero',action='store_true');p.add_argument('--smoke-scenes',type=int)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    assert (a.features/'COMPLETE').exists()
    assert (a.targets/'COMPLETE').exists()
    target_manifest={r['scene']:r for r in read(a.targets/'manifest.json')}
    manifest=read(a.features/'manifest.json');scenes={part:[] for part in ['train','val']}
    for row in manifest:
        if row['split'] not in scenes:continue
        if a.smoke_scenes and len(scenes[row['split']])>=a.smoke_scenes:continue
        path=a.features/(row['scene']+'.pt');assert sha(path)==row['sha256']
        s=torch.load(path,map_location='cpu',weights_only=False)
        target_path=a.targets/(row['scene']+'.npz')
        assert sha(target_path)==target_manifest[row['scene']]['target_sha256']
        target=np.load(target_path);torch.testing.assert_close(s['poses'],torch.from_numpy(target['poses']))
        refs=read(a.targets/(row['scene']+'_queries.json'))
        assert s['query_ids']==[q['query_id'] for q in refs]
        s['depth']=torch.from_numpy(target['depth']);s['mask']=torch.from_numpy(target['mask'])
        scenes[row['split']].append(s)
    config=vars(a).copy();config={k:str(v) if isinstance(v,Path) else v for k,v in config.items()}
    config.update(lr=1e-4,weight_decay=5e-4,batch_size=16,selection='final fixed step only',
        scene_ids={k:[s['scene'] for s in v] for k,v in scenes.items()},features_manifest_sha256=sha(a.features/'manifest.json'),targets_manifest_sha256=sha(a.targets/'manifest.json'))
    (a.out/'protocol.json').write_text(json.dumps(config,indent=2))
    # Empirical prior estimated from train labels only, evaluated on inner validation.
    sums=torch.zeros(36);counts=torch.zeros(36)
    for s in scenes['train']:sums+=(s['depth']*s['mask']).sum(0);counts+=s['mask'].sum(0)
    prior=sums/counts.clamp_min(1)
    prior_rows=[dict(scene=s['scene'],mae=float((s['depth']-prior).abs()[s['mask']].mean()),
                     rmse=float((s['depth']-prior).square()[s['mask']].mean().sqrt())) for s in scenes['val']]
    (a.out/'train_prior.json').write_text(json.dumps(prior_rows,indent=2))
    registry=[]
    for architecture in a.architectures:
      for head in a.heads:
       for seed in a.seeds:
        torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
        rng=np.random.default_rng(seed);model=GeometryNet(architecture,head).to(a.device)
        optimizer=torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=5e-4)
        run=a.out/f'{architecture}_{head}_{seed}';run.mkdir()
        start=time.perf_counter();history=[]
        if a.device=='cuda':torch.cuda.reset_peak_memory_stats()
        for step in range(a.steps):
            model.train();s=scenes['train'][int(rng.integers(len(scenes['train'])))];ids=rng.integers(len(s['depth']),size=16)
            x=s['echo' if architecture=='echo' else 'spatial'][ids].float().to(a.device)
            if a.phase_zero and architecture=='spatial':x[:,6:]=0
            y=s['depth'][ids].to(a.device);mask=s['mask'][ids].to(a.device)
            mean,logits=model(x);loss=masked_loss(mean,logits,y,mask)
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
            if step%100==0 or step==a.steps-1:
                event=dict(step=step+1,loss=float(loss),seconds=time.perf_counter()-start)
                history.append(event);print(run.name,json.dumps(event),flush=True)
        if a.device=='cuda':torch.cuda.synchronize()
        train_seconds=time.perf_counter()-start
        torch.save(dict(model=model.state_dict(),architecture=architecture,head=head,seed=seed,code_sha=a.code_sha),run/'model.pt')
        rows,pred=evaluate(model,scenes['val'],a.device,a.phase_zero)
        (run/'metrics.json').write_text(json.dumps(rows,indent=2));(run/'history.json').write_text(json.dumps(history,indent=2))
        for name,arrays in pred.items():np.savez_compressed(run/(name+'.npz'),**arrays)
        record=dict(run=run.name,seed=seed,architecture=architecture,head=head,steps=a.steps,code_sha=a.code_sha,
                    parameters=sum(p.numel() for p in model.parameters()),encoder_parameters=sum(p.numel() for p in model.encoder.parameters()),
                    train_seconds=train_seconds,checkpoint_sha256=sha(run/'model.pt'),
                    peak_gpu_bytes=torch.cuda.max_memory_allocated() if a.device=='cuda' else None)
        registry.append(record);(a.out/'registry.json').write_text(json.dumps(registry,indent=2))
    (a.out/'COMPLETE').write_text(a.code_sha+'\n')


if __name__=='__main__':main()

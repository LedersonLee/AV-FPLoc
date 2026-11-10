"""Controlled transfer into unchanged independent Track B full-bank matching.

Inner validation only. Both map descriptors and acoustic16 readout are trainable.
The same independently trained visual scorer is shared by all acoustic arms.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from .models import masked_loss
from .prepare_targets import read,sha
from .transfer_model import TransferAcoustic
from .train import donors
from track_b.models.foundation import FoundationStructureMatcher
from track_b.models.structure import visible_rays
from track_b.models.dense import dense_loss


def tensor_hash(module):
    h=hashlib.sha256()
    for name,t in module.state_dict().items():h.update(name.encode());h.update(t.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def load_data(a):
    split=read(a.targets/'protocol.json')['scenes'];parts={k:split[k] for k in ['train','val']}
    if a.smoke_scenes:parts={k:v[:a.smoke_scenes] for k,v in parts.items()}
    caches={r['scene']:r for r in read(a.scale/'cache_v1/manifest.json')}
    visual={r['scene']:r for r in read(a.scale/'features_v1/manifest.json')}
    audio={r['scene']:r for r in read(a.study/'features_v1/manifest.json')}
    targets={r['scene']:r for r in read(a.targets/'manifest.json')}
    data={};source=[]
    for part,ids in parts.items():
      data[part]=[]
      for scene in ids:
        cp=a.scale/'cache_v1'/f'{scene}.pt';vp=a.scale/'features_v1'/f'{scene}.pt'
        ap=a.study/'features_v1'/f'{scene}.pt';tp=a.targets/f'{scene}.npz'
        assert sha(cp)==caches[scene]['cache_sha256'];assert sha(vp)==visual[scene]['sha256']
        assert sha(ap)==audio[scene]['sha256'];assert sha(tp)==targets[scene]['target_sha256']
        c=torch.load(cp,map_location='cpu',weights_only=True);v=torch.load(vp,map_location='cpu',weights_only=True)
        au=torch.load(ap,map_location='cpu',weights_only=False);t=np.load(tp)
        assert c['query_ids']==v['query_ids']==au['query_ids']
        torch.testing.assert_close(c['gt'].double(),torch.from_numpy(t['poses']),atol=2e-6,rtol=1e-6)
        s={k:c[k] for k in ['scene','gt','target','candidates','floorplan','origin','spacing','hfov_deg','spectrum','query_ids']}
        s.update(visual=v['visual_features'],geometry_input=au['spatial' if a.architecture=='spatial' else 'echo'],
                 depth=torch.from_numpy(t['depth']),mask=torch.from_numpy(t['mask']),poses=au['poses'])
        data[part].append(s)
        source.append(dict(scene=scene,split=part,cache_sha256=caches[scene]['cache_sha256'],visual_sha256=visual[scene]['sha256'],audio_sha256=audio[scene]['sha256'],target_sha256=targets[scene]['target_sha256']))
    return data,source


def map_input(s,device):
    return dict(candidates=s['candidates'].to(device),floorplan=s['floorplan'].to(device),origin=s['origin'].to(device),spacing=s['spacing'])


def descriptors(model,scene):
    feature=model.floorplan(scene['floorplan'])
    calibration={k:scene[k] for k in ['origin','spacing']}
    return torch.cat([model.fp_descriptors(feature,p,calibration) for p in scene['candidates'].split(8192)])


def score(model,observed,fp,candidates,visual=False):
    weight=.25+.75*candidates[:,19:35]
    if visual:weight=weight*visible_rays(device=observed.device)
    return model.match(observed,fp,weight)


def make_model(seed,arm,architecture,checkpoint=None):
    torch.manual_seed(seed);model=FoundationStructureMatcher()
    if arm not in ['legacy','visual']:
        # Separate seed keeps FP/scale and common16-readout identical across arms.
        torch.manual_seed(seed+9000);model.acoustic=TransferAcoustic(architecture)
        model.acoustic.configure(arm,checkpoint['model'] if checkpoint else None)
    if arm=='visual':
        for p in model.acoustic.parameters():p.requires_grad=False
    else:
        for p in model.visual.parameters():p.requires_grad=False
    return model


def prediction(model,s,ids,arm,device,aux=False):
    if arm=='visual':return model.visual(s['visual'][ids].float().to(device))
    x=s['spectrum' if arm=='legacy' else 'geometry_input'][ids].float().to(device)
    return model.acoustic(x) if arm=='legacy' else model.acoustic(x,auxiliary=aux)


def fit(model,arm,seed,data,a,run):
    model.to(a.device);optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=3e-4,weight_decay=1e-4)
    rng=np.random.default_rng(seed);warmup=a.warmup if arm in ['legacy','visual'] else 0
    local=a.local_steps+(a.warmup if arm=='scratch_total' else 0)
    initial_encoder=tensor_hash(model.acoustic.geometry.encoder) if arm not in ['visual','legacy'] else None
    initial_readout=tensor_hash(model.acoustic.readout) if arm not in ['visual','legacy'] else None
    initial_map=tensor_hash(model.floorplan)+':'+tensor_hash(model.fp_head)
    history=[];start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    for step in range(warmup+local):
        if step==warmup:rng=np.random.default_rng(seed)
        model.train();s=data[int(rng.integers(len(data)))];ids=torch.from_numpy(rng.integers(len(s['gt']),size=8))
        output=prediction(model,s,ids,arm,a.device,aux=arm=='joint')
        if arm=='joint':observed,geodepth=output
        else:observed=output
        target=s['target'][ids].to(a.device)
        weight=visible_rays(device=a.device) if arm=='visual' else torch.ones(16,device=a.device)
        aux=((observed-target).square()*weight).sum(-1).mean()/weight.sum()
        ce=aux*0;anchor=aux*0;geometry=aux*0
        if step<warmup:loss=aux
        else:
            mi=map_input(s,a.device);fp=descriptors(model,mi)
            field=score(model,observed,fp,mi['candidates'],arm=='visual')
            ce=dense_loss(field,mi['candidates'][:,:3],s['gt'][ids].to(a.device))
            # Keep the established observation-matching auxiliary for every arm.
            # New geometry loss adds supervision, never an inference feature.
            anchor=(fp-mi['candidates'][:,3:19]).square().mean()
            loss=ce+(1000 if arm=='visual' else 10)*aux+(1000 if arm=='visual' else 0)*anchor
            if arm=='joint':
                geometry=masked_loss(geodepth,None,s['depth'][ids].to(a.device),s['mask'][ids].to(a.device))
                loss=loss+geometry
        assert torch.isfinite(loss),'nonfinite transfer objective'
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
        if step%100==0 or step==warmup+local-1:
            row=dict(step=step+1,loss=float(loss),ce=float(ce),fp_aux=float(aux),geometry_aux=float(geometry),elapsed_s=time.perf_counter()-start)
            history.append(row);print(run.name,json.dumps(row),flush=True)
    torch.cuda.synchronize();elapsed=time.perf_counter()-start
    final_encoder=tensor_hash(model.acoustic.geometry.encoder) if initial_encoder else None
    if arm=='frozen':assert initial_encoder==final_encoder,'Frozen encoder was changed'
    if initial_encoder and arm!='frozen':assert initial_encoder!=final_encoder,'Trainable encoder did not change'
    torch.save(dict(model=model.state_dict(),arm=arm,seed=seed,architecture=a.architecture,code_sha=a.code_sha),run/'model.pt')
    (run/'history.json').write_text(json.dumps(history,indent=2))
    return dict(arm=arm,seed=seed,source_sha=a.code_sha,local_steps=local,local_warmup_steps=warmup,
                geometry_pretrain_steps=1500 if arm in ['frozen','finetuned','joint'] else 0,
                initial_encoder_hash=initial_encoder,final_encoder_hash=final_encoder,initial_readout_hash=initial_readout,initial_map_hash=initial_map,
                seconds=elapsed,peak_gpu_bytes=torch.cuda.max_memory_allocated(),parameters=sum(p.numel() for p in model.parameters()),
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),checkpoint_sha256=sha(run/'model.pt'))


@torch.no_grad()
def evaluate(model,visual,arm,data,a,run):
    model.eval();visual.eval();rows=[];cost=[]
    for s in data:
        torch.cuda.synchronize();start=time.perf_counter();mi=map_input(s,a.device)
        afp=descriptors(model,mi);vfp=descriptors(visual,mi);poses=mi['candidates'][:,:3]
        donor=donors(s['poses']);n=len(s['gt']);chosen={}
        for ids in torch.arange(n).split(8):
            av=prediction(model,s,ids,arm,a.device);vv=prediction(visual,s,ids,'visual',a.device)
            replacement=prediction(model,s,donor[ids],arm,a.device)
            sv=score(visual,vv,vfp,mi['candidates'],True)
            sa=score(model,av,afp,mi['candidates']);ss=score(model,replacement,afp,mi['candidates'])
            gt=s['gt'][ids].to(a.device);dxy=(poses[None,:,:2]-gt[:,None,:2]).square().sum(-1).sqrt()
            dyaw=(poses[None,:,2]-gt[:,None,2]);dyaw=torch.atan2(dyaw.sin(),dyaw.cos()).abs()
            ok=(dxy<=1+1e-6)&(dyaw<=math.pi/6+1e-6)
            nearest=(dxy.square()/.35**2+dyaw.square()/math.radians(15)**2).argmin(-1)
            for condition,field in [('visual',sv),('acoustic',sa),('fixed_fusion',.85*sv+.15*sa),('shuffle_acoustic',ss),('shuffle_fusion',.85*sv+.15*ss)]:
                lp=field.log_softmax(-1);idx=field.argmax(-1);top=field.topk(20,-1).indices
                entropy=-(lp.exp()*lp).sum(-1)
                for j,q in enumerate(ids.tolist()):
                    rows.append(dict(scene=s['scene'],query=s['query_ids'][q],condition=condition,
                        r1=float(dxy[j,idx[j]]<=1+1e-6),joint=float(ok[j,idx[j]]),fine=float(dxy[j,idx[j]]<=.1+1e-6),
                        raw20=float(ok[j,top[j]].any()),nll=float(-lp[j,nearest[j]]),entropy=float(entropy[j]),
                        normalized_entropy=float(entropy[j]/math.log(len(poses))),position_error_m=float(dxy[j,idx[j]]),
                        yaw_error_deg=float(dyaw[j,idx[j]]*180/math.pi),candidate_count=len(poses),predicted_index=int(idx[j]),
                        oracle_joint=float(ok[j].any())))
        torch.cuda.synchronize();cost.append(dict(scene=s['scene'],candidates=len(poses),queries=n,evaluation_seconds=time.perf_counter()-start))
    (run/'metrics.json').write_text(json.dumps(rows));(run/'evaluation_cost.json').write_text(json.dumps(cost,indent=2))
    return {condition:{k:float(np.mean([r[k] for r in rows if r['condition']==condition])) for k in ['r1','joint','fine','nll','raw20']} for condition in sorted({r['condition'] for r in rows})}


def main():
    p=argparse.ArgumentParser()
    for k in ['study','scale','targets','out']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--architecture',choices=['spatial','echo'],default='spatial');p.add_argument('--seeds',nargs='+',type=int,default=[42,43,44])
    p.add_argument('--arms',nargs='+',default=['legacy','scratch','scratch_total','frozen','finetuned','joint'])
    p.add_argument('--warmup',type=int,default=1500);p.add_argument('--local-steps',type=int,default=1000)
    p.add_argument('--smoke-scenes',type=int);p.add_argument('--device',default='cuda');p.add_argument('--code-sha',required=True)
    a=p.parse_args();a.out.mkdir(exist_ok=False,parents=True);torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
    data,source=load_data(a)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    config.update(scenes={k:[s['scene'] for s in v] for k,v in data.items()},batch_size=8,lr=.0003,weight_decay=.0001,
        geometry_aux_weight=1.,acoustic_fp_aux_weight=10.,visual_fp_aux_weight=1000.,visual_anchor_weight=1000.,
        primary_architecture='Spatial regression fixed before localization outcomes; not selected by best depth MAE',
        scorer='unchanged FoundationStructureMatcher map descriptors/match, original full bank and masks; independent .85/.15 fusion',
        evidence='Exploratory inner family validation; historical19 holdout not evaluated',
        budgets='scratch local1000 controls local updates; scratch_total2500 controls total optimizer updates; pretrained1500+local1000; FLOPs differ')
    (a.out/'protocol.json').write_text(json.dumps(config,indent=2));(a.out/'inputs.json').write_text(json.dumps(source,indent=2));registry=[]
    for seed in a.seeds:
        vr=a.out/f'visual_{seed}';vr.mkdir();visual=make_model(seed,'visual',a.architecture)
        record=fit(visual,'visual',seed,data['train'],a,vr);registry.append(record);(a.out/'registry.json').write_text(json.dumps(registry,indent=2))
        visual.eval()
        for param in visual.parameters():param.requires_grad=False
        for arm in a.arms:
            run=a.out/f'{arm}_{seed}';run.mkdir()
            cp=a.study/'representation_v1'/f'{a.architecture}_regression_{seed}'/'model.pt'
            ck=torch.load(cp,map_location='cpu',weights_only=False) if arm in ['frozen','finetuned','joint'] else None
            model=make_model(seed,arm,a.architecture,ck)
            record=fit(model,arm,seed,data['train'],a,run)
            record['pretrained_checkpoint_sha256']=sha(cp) if ck else None
            record['summary']=evaluate(model,visual,arm,data['val'],a,run)
            registry.append(record);(a.out/'registry.json').write_text(json.dumps(registry,indent=2))
            print('COMPLETED',arm,seed,json.dumps(record['summary']),flush=True)
            del model
        del visual
    (a.out/'COMPLETE').write_text(a.code_sha+'\n')


if __name__=='__main__':main()

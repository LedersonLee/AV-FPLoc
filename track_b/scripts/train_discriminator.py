"""Scheduled new backend fits on OOF candidates; development is evaluation only."""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from track_b.models.discrimination import GeometryDiscriminator
from track_b.scripts.train_structure import read, sha


def load(folder):
    records=[]
    for path in sorted(folder.glob('*.pt')):
        records.extend(torch.load(path,weights_only=True,map_location='cpu'))
    return records


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--code-sha',required=True)
    args=p.parse_args();torch.set_num_threads(2)
    folds=[args.root/f'fold{i}_candidates_v1' for i in (0,1)]
    dev=args.root/'development_candidates_v1'
    protocols=[read(r/'protocol.json') for r in folds]; dp=read(dev/'protocol.json')
    assert all(p['role']=='oof' for p in protocols) and dp['role']=='development'
    held=[set(p['candidate_scenes']) for p in protocols]
    assert not held[0]&held[1] and held[0]|held[1]==set(dp['training_scenes'])
    assert all(set(p['training_scenes']).isdisjoint(p['candidate_scenes']) for p in protocols)
    assert not (held[0]|held[1])&set(dp['candidate_scenes'])
    args.out.mkdir(parents=True,exist_ok=False)
    setup=dict(code_sha=args.code_sha,steps=1000,batch_queries=32,lr=.001,weight_decay=.0001,
        loss='mean candidate BCE, all-negative queries retained',seeds=[42,43,44],
        policies=['raw20','diverse20'],arms=['geometry','score_statistics_only'],
        architecture='133-64-32-1 GELU; input ablation zeroes first128 standardized dimensions',
        initialization='same seed and sampling order for paired arms',
        selection='scheduled final step, no dev selection or early stopping',
        normalization='OOF training candidate feature mean/std only; std floor .001',
        protocol_hashes={str(r):sha(r/'protocol.json') for r in folds+[dev]},
        limitation='independent F1 backend; half-Train proposals vs full-Train development scorer; reused Val; frozen upstream exposure uncertain',test_accessed=False)
    (args.out/'protocol.json').write_text(json.dumps(setup,indent=2))
    train=sum([load(r) for r in folds],[]); evaluation=load(dev)
    rows=[];registry=[]
    for seed in setup['seeds']:
      for policy in setup['policies']:
        tr=[r for r in train if r['seed']==seed and r['policy']==policy]
        ev=[r for r in evaluation if r['seed']==seed and r['policy']==policy]
        assert len(tr)==1024 and len(ev)==384
        assert len({(r['scene'],r['query']) for r in tr})==len(tr)
        x=torch.stack([r['features'] for r in tr]); y=torch.stack([r['labels'] for r in tr]).float()
        mean=x.mean((0,1));std=x.std((0,1)).clamp_min(.001)
        x=(x-mean)/std; ex=(torch.stack([r['features'] for r in ev])-mean)/std
        for arm in setup['arms']:
          torch.manual_seed(seed);rng=np.random.default_rng(seed);model=GeometryDiscriminator()
          opt=torch.optim.AdamW(model.parameters(),lr=setup['lr'],weight_decay=setup['weight_decay'])
          xx,ee=x.clone(),ex.clone()
          if arm=='score_statistics_only':xx[:,:,:128]=0;ee[:,:,:128]=0
          start=time.perf_counter();history=[]
          for step in range(setup['steps']):
            ids=torch.tensor(rng.integers(len(tr),size=setup['batch_queries']))
            opt.zero_grad();loss=F.binary_cross_entropy_with_logits(model(xx[ids]),y[ids]);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
            if step%100==0 or step==setup['steps']-1:history.append(dict(step=step,loss=float(loss)))
          elapsed=time.perf_counter()-start;model.eval()
          with torch.no_grad():out=model(ee);indices=out.argmax(-1)
          for r,k in zip(ev,indices.tolist()):
            for method,idx in [('map',0),('support',int(r['support'].argmax())),(arm,k)]:
              rows.append(dict(seed=seed,policy=policy,arm=arm,method=method,scene=r['scene'],query=r['query'],
                joint=float(r['labels'][idx]),r1=float(r['distance'][idx]<=1+1e-6),fine=float(r['distance'][idx]<=.1+1e-6),
                strict_05_15=float((r['distance'][idx]<=.5+1e-6)&(r['yaw'][idx]<=np.pi/12+1e-6)),
                strict_02_10=float((r['distance'][idx]<=.2+1e-6)&(r['yaw'][idx]<=np.pi/18+1e-6)),
                translation_error_m=float(r['distance'][idx]),yaw_error_deg=float(r['yaw'][idx]*180/np.pi),
                oracle=float(r['labels'].any())))
          name=f'{policy}_{arm}_{seed}'
          torch.save(dict(model=model.state_dict(),mean=mean,std=std,arm=arm),args.out/f'{name}.pt')
          record=dict(run_id=name,code_sha=args.code_sha,seed=seed,policy=policy,arm=arm,
             parameters=sum(p.numel() for p in model.parameters()),train_queries=len(tr),dev_queries=len(ev),
             train_no_positive_queries=int((~y.bool().any(-1)).sum()),train_seconds=elapsed,
             device='CPU',history=history,checkpoint_sha256=sha(args.out/f'{name}.pt'))
          registry.append(record);(args.out/'fits.json').write_text(json.dumps(registry,indent=2))
          print('FIT',name,elapsed,flush=True)
    summary=[];paired=[];rng=np.random.default_rng(20261109)
    for policy in setup['policies']:
      for arm in setup['arms']:
        rr=[r for r in rows if r['policy']==policy and r['arm']==arm]
        for method in ['map','support',arm]:
          selected=[r for r in rr if r['method']==method]
          summary.append(dict(policy=policy,arm=arm,method=method,
             **{k:float(np.mean([r[k] for r in selected])) for k in ['joint','r1','fine','oracle','strict_05_15','strict_02_10']},
             median_translation_m=float(np.median([r['translation_error_m'] for r in selected])),
             median_yaw_deg=float(np.median([r['yaw_error_deg'] for r in selected]))))
        for control in ['map','support']:
          d=np.array([np.mean([r['joint'] for r in rr if r['scene']==s and r['method']==arm])-np.mean([r['joint'] for r in rr if r['scene']==s and r['method']==control]) for s in dp['candidate_scenes']])
          boot=d[rng.integers(len(d),size=(10000,len(d)))].mean(1)*100
          paired.append(dict(policy=policy,arm=arm,control=control,delta_pp=float(100*d.mean()),ci95_pp=np.quantile(boot,[.025,.975]).tolist(),unit='scene; seeds averaged'))
    for name,obj in [('summary',summary),('predictions',rows),('paired',paired)]:
        (args.out/f'{name}.json').write_text(json.dumps(obj,indent=2))
    print(json.dumps(summary),flush=True);print('COMPLETE',flush=True)

if __name__=='__main__':main()

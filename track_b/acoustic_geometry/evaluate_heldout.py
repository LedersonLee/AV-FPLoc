"""Evaluate every pre-frozen model; no optimizer, selection or threshold changes."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from .prepare_targets import read,sha
from .models import GeometryNet
from .train import evaluate as representation_evaluate
from .transfer import make_model,evaluate as transfer_evaluate
from .summarize import paired


def aggregate(rows,keys,metrics):
    grouped={}
    for r in rows:
        for m in metrics:
            if r.get(m) is not None:grouped.setdefault(tuple(r[k] for k in keys)+(m,),{}).setdefault(r['scene'],[]).append(r[m])
    means={k:{s:float(np.mean(v)) for s,v in d.items()} for k,d in grouped.items()}
    summary=[dict(**dict(zip(keys,k[:-1])),metric=k[-1],**paired(v)) for k,v in means.items()]
    return summary,means


def main():
    p=argparse.ArgumentParser()
    for k in ['study','cache','freeze','out']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--code-sha',required=True);p.add_argument('--device',default='cuda')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    assert (a.cache/'COMPLETE').exists();protocol=read(a.cache/'protocol.json');frozen=read(a.freeze)
    assert protocol['freeze_sha256']==sha(a.freeze)
    source={}
    for r in frozen['checkpoints']:
        cp=a.study/r['run']/r['model']/'model.pt';assert sha(cp)==r['sha256'];source[(r['run'],r['model'])]=cp
    corpus=[]
    for r in read(a.cache/'manifest.json'):
        cp=a.cache/f"{r['scene']}.pt";assert sha(cp)==r['sha256'];corpus.append(torch.load(cp,map_location='cpu',weights_only=False))
    assert [s['scene'] for s in corpus]==frozen[protocol['part']]
    config=dict(code_sha=a.code_sha,cache_protocol_sha256=sha(a.cache/'protocol.json'),freeze_sha256=sha(a.freeze),part=protocol['part'],evidence=frozen['historical_exposure'],policy='all frozen models; identical metrics and no fitting')
    (a.out/'protocol.json').write_text(json.dumps(config,indent=2));rep_rows=[];prior_rows=[];transfer_rows=[]
    pri=np.load(a.study/'representation_v1/train_prior_parameters.npz')
    for s in corpus:
        y=s['depth'].numpy();mask=s['mask'].numpy();labels=np.clip((y*4).astype(int),0,79)
        nll=-np.log(pri['probability'][np.arange(36)[None],labels])[mask].mean()
        prior_rows.append(dict(scene=s['scene'],median_mae=float(np.abs(y-pri['median'])[mask].mean()),categorical_nll=float(nll)))
    for input_condition in corpus[0]['conditions']:
        data=[]
        for s in corpus:
            x=dict(s);x.update(s['conditions'][input_condition]);x['geometry_input']=x['spatial'];data.append(x)
        for training in ['representation_v1','phase_zero_v1']:
            for r in read(a.study/training/'registry.json'):
                cp=source[(training,r['run'])];model=GeometryNet(r['architecture'],r['head']).to(a.device)
                model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=False)['model'])
                run=a.out/input_condition/training/r['run'];run.mkdir(parents=True)
                rows,pred=representation_evaluate(model,data,a.device,training=='phase_zero_v1')
                for row in rows:
                    arr=pred[row['scene']+'__'+row['condition']];m=arr['mean'];y=arr['target'];mask=arr['mask'];cor=[]
                    for mm,yy,valid in zip(m,y,mask):
                        if valid.sum()>2 and mm[valid].std()>1e-7 and yy[valid].std()>1e-7:cor.append(float(np.corrcoef(mm[valid],yy[valid])[0,1]))
                    row['within_query_angular_correlation']=float(np.mean(cor)) if cor else None
                    row['scalarized_mae']=float(np.abs(m.mean(-1,keepdims=True)-y)[mask].mean())
                    row['rotated180_mae']=float(np.abs(np.roll(m,18,-1)-y)[mask].mean())
                    rep_rows.append(dict(training=training,architecture=r['architecture'],head=r['head'],seed=r['seed'],input_condition=input_condition,**row))
                (run/'metrics.json').write_text(json.dumps(rows,indent=2))
                for name,arr in pred.items():np.savez_compressed(run/(name+'.npz'),**arr)
                del model;print('REP',input_condition,training,r['run'],flush=True)
        tc=read(a.study/'transfer_v1/protocol.json')
        for seed in tc['seeds']:
            visual=make_model(seed,'visual',tc['architecture']).to(a.device)
            visual.load_state_dict(torch.load(source[('transfer_v1',f'visual_{seed}')],map_location='cpu',weights_only=False)['model'])
            for arm in tc['arms']:
                model=make_model(seed,arm,tc['architecture']).to(a.device)
                model.load_state_dict(torch.load(source[('transfer_v1',f'{arm}_{seed}')],map_location='cpu',weights_only=False)['model'])
                run=a.out/input_condition/'transfer_v1'/f'{arm}_{seed}';run.mkdir(parents=True)
                transfer_evaluate(model,visual,arm,data,SimpleNamespace(device=a.device),run)
                rows=read(run/'metrics.json')
                for scene in frozen[protocol['part']]:
                    for condition in sorted({r['condition'] for r in rows}):
                        group=[r for r in rows if r['scene']==scene and r['condition']==condition];assert len(group)==64
                        metrics={k:float(np.mean([r[k] for r in group])) for k in ['r1','joint','fine','raw20','nll','entropy','normalized_entropy','position_error_m','yaw_error_deg','oracle_joint']}
                        transfer_rows.append(dict(scene=scene,arm=arm,seed=seed,input_condition=input_condition,condition=condition,**metrics))
                del model;print('TRANSFER',input_condition,arm,seed,flush=True)
            del visual
    rs,rm=aggregate(rep_rows,['training','architecture','head','input_condition','condition'],['mae','rmse','direction_correlation','categorical_nll','crps_m','coverage90','width90_m','within_query_angular_correlation','scalarized_mae','rotated180_mae'])
    ts,tm=aggregate(transfer_rows,['arm','input_condition','condition'],['r1','joint','fine','raw20','nll','entropy','normalized_entropy','position_error_m','yaw_error_deg','oracle_joint'])
    contrasts=[]
    for input_condition in corpus[0]['conditions']:
        for left,right in [('frozen','scratch'),('finetuned','scratch'),('joint','scratch'),('frozen','scratch_total'),('finetuned','scratch_total'),('joint','scratch_total'),('frozen','legacy'),('finetuned','legacy'),('joint','legacy')]:
            for condition in ['acoustic','fixed_fusion']:
                for metric in ['joint','r1','fine','nll']:
                    l=tm[(left,input_condition,condition,metric)];r=tm[(right,input_condition,condition,metric)]
                    contrasts.append(dict(left=left,right=right,input_condition=input_condition,condition=condition,metric=metric,**paired({s:l[s]-r[s] for s in l})))
    result=dict(**config,representation_summary=rs,transfer_summary=ts,transfer_contrasts=contrasts,representation_per_scene=rep_rows,transfer_per_scene=transfer_rows,train_prior=prior_rows)
    (a.out/'summary.json').write_text(json.dumps(result,indent=2));(a.out/'COMPLETE').write_text(a.code_sha+'\n')


if __name__=='__main__':main()

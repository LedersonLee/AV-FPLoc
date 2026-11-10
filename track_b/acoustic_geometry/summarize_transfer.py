"""Validate completed transfer arms and compute family-paired localization effects."""
import argparse
import json
from pathlib import Path
import numpy as np
from .prepare_targets import read,sha
from .summarize import paired


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--code-sha',required=True)
    a=p.parse_args();a.out.mkdir(exist_ok=False,parents=True);assert (a.run/'COMPLETE').exists()
    config=read(a.run/'protocol.json');registry=read(a.run/'registry.json')
    expected={(arm,seed) for arm in config['arms']+['visual'] for seed in config['seeds']}
    assert {(r['arm'],r['seed']) for r in registry}==expected
    conditions={'visual','acoustic','fixed_fusion','shuffle_acoustic','shuffle_fusion'}
    metrics=['joint','r1','fine','raw20','nll','entropy','normalized_entropy','position_error_m','yaw_error_deg']
    values={};rows_out=[];visual_refs={};matching_keys=None
    for seed in config['seeds']:
        subset=[r for r in registry if r['seed']==seed]
        assert len({r['initial_map_hash'] for r in subset})==1
        acoustic=[r for r in subset if r['initial_readout_hash']]
        assert len({r['initial_readout_hash'] for r in acoustic})==1
        pretrained=[r for r in subset if r['arm'] in ['frozen','finetuned','joint']]
        assert len({r['initial_encoder_hash'] for r in pretrained})==1
        assert len({r['pretrained_checkpoint_sha256'] for r in pretrained})==1
    for r in registry:
        arm=r['arm'];seed=r['seed'];path=a.run/f'{arm}_{seed}'
        assert sha(path/'model.pt')==r['checkpoint_sha256']
        if arm=='frozen':assert r['initial_encoder_hash']==r['final_encoder_hash']
        if arm=='visual':continue
        rows=read(path/'metrics.json');assert {x['condition'] for x in rows}==conditions
        keys={(x['scene'],x['query'],x['condition']) for x in rows};assert len(keys)==len(rows)
        if matching_keys is None:matching_keys=keys
        assert keys==matching_keys
        assert {x['scene'] for x in rows}==set(config['scenes']['val'])
        for x in rows:
            key=(seed,x['scene'],x['query'])
            if x['condition']=='visual':
                if key not in visual_refs:visual_refs[key]=x
                else:
                    assert all(abs(x[k]-visual_refs[key][k])<1e-7 for k in metrics+['predicted_index','candidate_count'])
        for condition in sorted(conditions):
            for scene in config['scenes']['val']:
                group=[x for x in rows if x['condition']==condition and x['scene']==scene]
                assert len(group)==64 or config['smoke_scenes']
                entry=dict(arm=arm,seed=seed,scene=scene,condition=condition,queries=len(group))
                for metric in metrics:
                    mean=float(np.mean([x[metric] for x in group]));assert np.isfinite(mean)
                    entry[metric]=mean
                    values.setdefault((arm,condition,metric),{}).setdefault(scene,[]).append(mean)
                rows_out.append(entry)
    means={k:{s:float(np.mean(v)) for s,v in d.items()} for k,d in values.items()}
    summary=[dict(arm=k[0],condition=k[1],metric=k[2],**paired(v)) for k,v in means.items()]
    contrasts=[]
    comparisons=[('frozen','scratch'),('finetuned','scratch'),('joint','scratch'),('frozen','scratch_total'),('finetuned','scratch_total'),('joint','scratch_total'),('finetuned','frozen'),('joint','finetuned'),('frozen','legacy'),('finetuned','legacy'),('joint','legacy')]
    for left,right in comparisons:
        if left not in config['arms'] or right not in config['arms']:continue
        for cond in ['acoustic','fixed_fusion']:
            for metric in ['joint','r1','fine','nll','raw20']:
                l=means[(left,cond,metric)];r=means[(right,cond,metric)]
                contrasts.append(dict(left=left,right=right,condition=cond,metric=metric,**paired({s:l[s]-r[s] for s in l})))
    for arm in config['arms']:
        for left,right in [('fixed_fusion','visual'),('acoustic','shuffle_acoustic'),('fixed_fusion','shuffle_fusion')]:
            for metric in ['joint','r1','fine','nll']:
                l=means[(arm,left,metric)];r=means[(arm,right,metric)]
                contrasts.append(dict(left=left,right=right,arm=arm,metric=metric,**paired({s:l[s]-r[s] for s in l})))
    result=dict(code_sha=a.code_sha,training_protocol_sha256=sha(a.run/'protocol.json'),
        validation=dict(all_arms_complete=True,checkpoint_hashes=True,shared_map_initialization=True,shared_readout_initialization=True,
                        shared_pretrained_encoder=True,frozen_encoder_unchanged=True,identical_visual_predictions=True,query_keys_equal=True),
        inference='No candidate acoustic reference RIR; no GT features or filtering; same complete candidate bank per query',
        evidence='Exploratory inner-validation source families; not independently validated',
        uncertainty='10,000 paired family bootstrap draws, seeds then same-family variants averaged; uncorrected comparisons',
        summary=summary,contrasts=contrasts,per_scene=rows_out,registry=registry)
    (a.out/'summary.json').write_text(json.dumps(result,indent=2))
    lines=['# Geometry encoder transfer — inner validation','',result['evidence'],result['uncertainty'],'',
        '| Arm | Acoustic R@1m % | Acoustic joint % | Fixed fusion joint % | Fixed fusion fine % | Fusion NLL |','|---|---:|---:|---:|---:|---:|']
    lookup={(r['arm'],r['condition'],r['metric']):r for r in summary}
    for arm in config['arms']:
        columns=[]
        for cond,metric in [('acoustic','r1'),('acoustic','joint'),('fixed_fusion','joint'),('fixed_fusion','fine'),('fixed_fusion','nll')]:
            x=lookup[(arm,cond,metric)];scale=1 if metric=='nll' else 100
            columns.append(f"{x['mean']*scale:.3f}")
        lines.append('| '+arm+' | '+' | '.join(columns)+' |')
    visual=lookup[(config['arms'][0],'visual','joint')]
    lines+=['',f"Shared visual joint: {100*visual['mean']:.3f}% (same predictions across every acoustic arm).",'',
        'Same local steps and same total optimizer updates are distinct controls; pretraining and dense-scoring FLOPs differ. Full paired intervals, all seeds, errors, entropy, candidate coverage and costs are in summary.json.']
    (a.out/'results.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()

"""Family-clustered, paired uncertainty; seeds and query rays are not replicates."""
import argparse
import json
from pathlib import Path
import numpy as np
from .prepare_targets import read,sha


def paired(values,seed=105):
    families={}
    for scene,value in values.items():families.setdefault(scene.split('_')[0],[]).append(value)
    x=np.array([np.mean(families[k]) for k in sorted(families)])
    rng=np.random.default_rng(seed);boot=x[rng.integers(len(x),size=(10000,len(x)))].mean(-1)
    return dict(mean=float(x.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),independent_families=len(x),scenes=len(values))


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--code-sha',required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    assert (a.run/'COMPLETE').exists()
    config=read(a.run/'protocol.json');registry=read(a.run/'registry.json')
    expected={(arch,head,seed) for arch in config['architectures'] for head in config['heads'] for seed in config['seeds']}
    assert {(r['architecture'],r['head'],r['seed']) for r in registry}==expected
    values={};per_seed=[]
    for r in registry:
        path=a.run/r['run'];assert sha(path/'model.pt')==r['checkpoint_sha256']
        rows=read(path/'metrics.json')
        assert len(rows)==len(config['scene_ids']['val'])*2
        assert {x['scene'] for x in rows}==set(config['scene_ids']['val'])
        assert len({(x['scene'],x['condition']) for x in rows})==len(rows)
        for row in rows:
            for metric,val in row.items():
                if metric in ['scene','condition','valid'] or val is None:continue
                assert np.isfinite(val)
                key=(r['architecture'],r['head'],row['condition'],metric)
                values.setdefault(key,{}).setdefault(row['scene'],[]).append(val)
        for condition in ['normal','same_scene_different_position']:
            rr=[x for x in rows if x['condition']==condition]
            per_seed.append(dict(run=r['run'],seed=r['seed'],condition=condition,
                mae=paired({x['scene']:x['mae'] for x in rr})['mean']))
    means={k:{s:float(np.mean(v)) for s,v in scene.items()} for k,scene in values.items()}
    summary=[dict(architecture=k[0],head=k[1],condition=k[2],metric=k[3],**paired(v)) for k,v in means.items()]
    contrasts=[];prior={r['scene']:r for r in read(a.run/'train_prior.json')}
    for arch,head in sorted({k[:2] for k in means}):
        normal=means[(arch,head,'normal','mae')];shuffle=means[(arch,head,'same_scene_different_position','mae')]
        contrasts.append(dict(architecture=arch,head=head,contrast='shuffle_minus_normal_mae_m',**paired({s:shuffle[s]-normal[s] for s in normal})))
        contrasts.append(dict(architecture=arch,head=head,contrast='model_minus_train_median_mae_m',**paired({s:normal[s]-prior[s]['median_mae'] for s in normal})))
    out=dict(code_sha=a.code_sha,training_protocol_sha256=sha(a.run/'protocol.json'),evidence='Exploratory, inner scene-family validation; no independent Test',
             uncertainty='10,000 paired source-family bootstrap draws after seed and variant averaging; uncorrected intervals',summary=summary,contrasts=contrasts,per_seed=per_seed,registry=registry)
    (a.out/'summary.json').write_text(json.dumps(out,indent=2))
    lines=['# Inner validation representation results','',out['evidence'],out['uncertainty'],'',
           '| Encoder | Head | MAE m (95% family interval) | Shuffle − normal m | Model − train median m |','|---|---|---|---|---|']
    def cell(x):return f"{x['mean']:.3f} [{x['ci95'][0]:.3f}, {x['ci95'][1]:.3f}]"
    for arch,head in sorted({k[:2] for k in means}):
        v=next(x for x in summary if (x['architecture'],x['head'],x['condition'],x['metric'])==(arch,head,'normal','mae'))
        c=[x for x in contrasts if x['architecture']==arch and x['head']==head]
        lines.append(f'| {arch} | {head} | {cell(v)} | {cell(c[0])} | {cell(c[1])} |')
    lines+=['','Positive shuffle minus normal supports observation-specific sensitivity, not physical mechanism identification. Negative model minus prior indicates better masked depth MAE. Neither establishes localization transfer or furniture invariance. Full metrics, all seeds, costs and prior comparisons are retained in summary.json.']
    (a.out/'results.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()

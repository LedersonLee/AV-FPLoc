"""Summarize fixed held-out comparisons; no performance-based selection."""
import argparse
import json
from pathlib import Path
import numpy as np
from .prepare_targets import read,sha
from .summarize import paired


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--code-sha',required=True)
    a=p.parse_args();assert (a.run/'COMPLETE').exists();a.out.mkdir(parents=True,exist_ok=False);x=read(a.run/'summary.json')
    def scene_means(filters,metric):
        groups={}
        for r in x['representation_per_scene']:
            if all(r[k]==v for k,v in filters.items()):groups.setdefault(r['scene'],[]).append(r[metric])
        return {s:float(np.mean(v)) for s,v in groups.items()}
    contrasts=[];conditions=sorted({r['input_condition'] for r in x['representation_per_scene']})
    prior={r['scene']:r['median_mae'] for r in x['train_prior']}
    for condition in conditions:
        for arch in ['echo','spatial']:
            for head in ['regression','categorical']:
                base=dict(training='representation_v1',architecture=arch,head=head,input_condition=condition)
                normal=scene_means(dict(**base,condition='normal'),'mae')
                alternatives={'shuffle_minus_normal':scene_means(dict(**base,condition='same_scene_different_position'),'mae'),'scalarized_minus_normal':scene_means(dict(**base,condition='normal'),'scalarized_mae'),'rotated180_minus_normal':scene_means(dict(**base,condition='normal'),'rotated180_mae'),'train_median_minus_normal':prior}
                for label,alt in alternatives.items():contrasts.append(dict(input_condition=condition,architecture=arch,head=head,contrast=label,**paired({s:alt[s]-normal[s] for s in normal})))
                if arch=='spatial':
                    phase=scene_means({**base,'training':'phase_zero_v1','condition':'normal'},'mae')
                    contrasts.append(dict(input_condition=condition,architecture=arch,head=head,contrast='phase_zero_minus_full',**paired({s:phase[s]-normal[s] for s in normal})))
                if 'furnished' in conditions and 'stage_only' in conditions and condition=='furnished':
                    stage=scene_means({**base,'input_condition':'stage_only','condition':'normal'},'mae')
                    contrasts.append(dict(input_condition=condition,architecture=arch,head=head,contrast='stage_only_minus_furnished',**paired({s:stage[s]-normal[s] for s in normal})))
    output=dict(code_sha=a.code_sha,evaluation_summary_sha256=sha(a.run/'summary.json'),evidence=x['evidence'],uncertainty='10,000 source-family paired bootstrap draws after seed averaging; uncorrected intervals',contrasts=contrasts)
    (a.out/'contrasts.json').write_text(json.dumps(output,indent=2))
    lines=['# Frozen official Test evaluation' if x['part']=='official_test' else '# Frozen development evaluation','',x['evidence'],'',output['uncertainty']+'. No checkpoint, model or threshold updates followed these results.']
    lookup={(r['architecture'],r['head'],r['input_condition'],r['condition'],r['metric']):r for r in x['representation_summary'] if r['training']=='representation_v1'}
    def val(arch,head,cond,metric):return lookup[(arch,head,cond,'normal',metric)]['mean']
    def interval(r):return f"{r['mean']:.3f} [{r['ci95'][0]:.3f},{r['ci95'][1]:.3f}]"
    for condition in conditions:
        lines+=['','## '+condition,'','| Encoder/head | MAE m (95% CI) | RMSE m | Angular correlation within query | Shuffle − normal MAE m |','|---|---:|---:|---:|---:|']
        for arch in ['echo','spatial']:
            for head in ['regression','categorical']:
                c=next(r for r in contrasts if r['input_condition']==condition and r['architecture']==arch and r['head']==head and r['contrast']=='shuffle_minus_normal')
                lines.append(f"| {arch}/{head} | {interval(lookup[(arch,head,condition,'normal','mae')])} | {val(arch,head,condition,'rmse'):.3f} | {val(arch,head,condition,'within_query_angular_correlation'):.4f} | {interval(c)} |")
        lines+=['','| Categorical encoder | NLL | CRPS m | 90% coverage | 90% width m |','|---|---:|---:|---:|---:|']
        for arch in ['echo','spatial']:lines.append('| '+arch+' | '+' | '.join(f'{val(arch,"categorical",condition,m):.3f}' for m in ['categorical_nll','crps_m','coverage90','width90_m'])+' |')
        tl={(r['arm'],r['condition'],r['metric']):r['mean'] for r in x['transfer_summary'] if r['input_condition']==condition}
        lines+=['','| Arm | Acoustic R@1m % | Acoustic joint % | Fusion R@1m % | Fusion joint % | Fusion fine % | Fusion NLL |','|---|---:|---:|---:|---:|---:|---:|']
        for arm in ['legacy','scratch','scratch_total','frozen','finetuned','joint']:
            values=[100*tl[(arm,c,m)] for c,m in [('acoustic','r1'),('acoustic','joint'),('fixed_fusion','r1'),('fixed_fusion','joint'),('fixed_fusion','fine')]]+[tl[(arm,'fixed_fusion','nll')]]
            lines.append('| '+arm+' | '+' | '.join(f'{v:.3f}' for v in values)+' |')
        lines+=['',f"Shared visual joint: {100*tl[('legacy','visual','joint')]:.3f}%. Full entropy, normalized entropy, errors, rawTop20, oracle coverage, all seeds and paired transfer intervals remain in the aggregate."]
    lines+=['','Stage-only removes all object instances, including doors/windows. This is not a pure furniture intervention. The separate development opening-preserving control is required to interpret object-removal effects. Predictions and targets use the same conditional valid-direction mask. Scene-disjoint evaluation does not erase prior visual/upstream asset exposure.']
    (a.out/'results.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()

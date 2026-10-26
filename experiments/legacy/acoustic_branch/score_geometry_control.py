"""Compare fresh controlled renders with the same frozen FP candidate pools."""
import argparse
from pathlib import Path
import importlib.util
import sys

import numpy as np

from .contracts import atomic_json, sha
from .data import read
from .features import extract, roll_features


def run(root, directory, source, config):
    if read(directory/'status.json')['status']!='complete':raise ValueError('Control render incomplete')
    spec=importlib.util.spec_from_file_location('grid_control',source)
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    c=m.GridScoreConfig(sample_rate_hz=48000,direct_guard_samples=96,usable_samples=6144,nfft=256,hop=64,
        bands=((0,500),(500,1500),(1500,4000)),normalise='sum',distance='symmetric',shape_feature=True,feature='stft_band')
    plan=read(directory/'plan.json')
    features=read(config)['features']
    rows=[]
    normal_rows=[]
    for scene,queries in plan['scenes'].items():
        for q in queries:
            bank=[];coh=[]
            for r in q['candidates']:
                rir=np.roll(np.load(root/r['path']),r['roll'],axis=0)
                bank.append(m.featurise(m.band_energy(rir,c),c))
                coh.append(extract(rir,features)[1])
            bank,coh=np.asarray(bank),np.asarray(coh)
            native=[];native_coh=[]
            for i,r in enumerate(q['candidates']):
                path=directory/'rirs/native_stage_candidates'/scene/f'{q["sample_index"]:05d}_{i}.npy'
                if sha(path)!=read(path.with_suffix('.json'))['sha256']:raise ValueError('Native candidate SHA mismatch')
                rir=np.roll(np.load(path),r['roll'],axis=0)
                native.append(m.featurise(m.band_energy(rir,c),c));native_coh.append(extract(rir,features)[1])
            native,native_coh=np.asarray(native),np.asarray(native_coh)
            inward=directory/'rirs/fp_inward'/scene/f'{q["sample_index"]:05d}.npy'
            if inward.exists():
                if sha(inward)!=read(inward.with_suffix('.json'))['sha256']:raise ValueError('Inward SHA mismatch')
                flipped=m.featurise(m.band_energy(np.load(inward),c),c)
                repeat=m.featurise(m.band_energy(np.load(directory/'rirs/fp_repeat'/scene/f'{q["sample_index"]:05d}.npy'),c),c)
                stage=m.featurise(m.band_energy(np.load(directory/'rirs/native_stage'/scene/f'{q["sample_index"]:05d}.npy'),c),c)
                furnished=m.featurise(m.band_energy(np.load(directory/'rirs/furnished_repeat'/scene/f'{q["sample_index"]:05d}.npy'),c),c)
                # Row permutation matters to this feature; all four fresh
                # controls share the same base and the same query roll.
                # Raw bank slot0 is already rolled, so unroll it for this audit.
                original=m.featurise(m.band_energy(np.load(root/q['candidates'][0]['path']),c),c)
                distance=lambda a,b:float(np.abs(a-b).sum())
                normal_rows.append(dict(scene=scene,sample_index=q['sample_index'],
                    original_to_repeat=distance(original,repeat),repeat_to_inward=distance(repeat,flipped),
                    stage_to_outward=distance(stage,repeat),stage_to_inward=distance(stage,flipped),
                    furnished_to_outward=distance(furnished,repeat),furnished_to_inward=distance(furnished,flipped)))
            for condition in ('fp_repeat','native_stage','furnished_repeat','furnished_original','native_vs_native','furnished_vs_native'):
                if condition=='furnished_original':path=root/q['observed']['path']
                elif condition=='native_vs_native':path=directory/'rirs/native_stage'/scene/f'{q["sample_index"]:05d}.npy'
                elif condition=='furnished_vs_native':path=directory/'rirs/furnished_repeat'/scene/f'{q["sample_index"]:05d}.npy'
                else:path=directory/'rirs'/condition/scene/f'{q["sample_index"]:05d}.npy'
                rir=np.roll(np.load(path),q['observed']['roll'],axis=0)
                if condition!='furnished_original' and sha(path)!=read(path.with_suffix('.json'))['sha256']:
                    raise ValueError('Control SHA mismatch')
                f=m.featurise(m.band_energy(rir,c),c)
                cs=extract(rir,features)[1]
                target,target_coh=(native,native_coh) if condition in ('native_vs_native','furnished_vs_native') else (bank,coh)
                d=np.abs(target-f).sum(axis=tuple(range(1,bank.ndim)))
                cd=np.square(target_coh-cs).mean(axis=tuple(range(1,coh.ndim)))
                win=d==d.min();cw=cd==cd.min()
                spatial=[0]+[i for i in range(1,8) if q['candidates'][i]['kind']!='orientation_hard']
                sd=d[spatial];sw=sd==sd.min()
                rows.append(dict(scene=scene,sample_index=q['sample_index'],condition=condition,
                    stft_top1=float(win[0]/win.sum()),coherence_top1=float(cw[0]/cw.sum()),
                    stft_spatial_top1=float(sw[0]/sw.sum()),
                    stft_positive_distance=float(d[0]),stft_distances=d.tolist()))
    summary={condition:dict(queries=len([r for r in rows if r['condition']==condition]),
        **{k:float(np.mean([r[k] for r in rows if r['condition']==condition])) for k in ('stft_top1','coherence_top1','stft_spatial_top1','stft_positive_distance')})
        for condition in sorted({r['condition'] for r in rows})}
    atomic_json(directory/'results.json',dict(test_accessed=False,summary=summary,rows=rows,
        normal_audit=dict(queries=len(normal_rows),mean={k:float(np.mean([r[k] for r in normal_rows]))
            for k in ('original_to_repeat','repeat_to_inward','stage_to_outward','stage_to_inward',
                      'furnished_to_outward','furnished_to_inward')} if normal_rows else {},records=normal_rows),
        featuriser_sha256=sha(source),plan_sha256=sha(directory/'plan.json'),config_sha256=sha(config),
        caveat='Native stage lacks object-based doors/windows too; stage→furnished is not furniture-only causality. 24 queries, diagnostic not benchmark.'))
    print(summary)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','directory','source','config'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.root,a.directory,a.source,a.config)

"""Matched native-stage vs canonical-extrusion evaluation on pilot VAL only."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
from scipy.spatial.distance import cdist

from .contracts import atomic_json, immutable_json, sha
from .data import read, load_manifest
from .features import extract


def stage_path(output, original):
    relative=Path(original).relative_to('RIR/FP/val')
    return output/'native_stage'/relative


def prepare(root,config,output):
    queries,digest=load_manifest(root,'val',read(config))
    refs={r['path']:r for q in queries for r in q['candidates']}
    scenes=sorted({q['scene_id'] for q in queries})
    tasks=[]
    for scene in scenes:
        paths=sorted(p for p in refs if Path(p).parts[3]==scene)
        for start in range(0,len(paths),12):tasks.append(dict(scene=scene,paths=paths[start:start+12]))
    immutable_json(output/'plan.json',dict(test_accessed=False,split='val',queries=len(queries),
        manifest_sha256=digest,config_sha256=sha(config),refs=refs,tasks=tasks,
        comparison='same observed RIR, same candidate XYZ/yaw; geometry only changes'))
    # Reuse the small completed native-stage candidate control where identity
    # and physical geometry are exactly the same. Never alter shared raw data.
    control=root/'AcousticContrastive/diagnostics/learning_failure_v1/geometry_control'
    reused=0
    for meta in sorted((control/'rirs/native_stage_candidates').glob('*/*.json')):
        m=read(meta);r=m.get('reference')
        if r is None or r['path'] not in refs:continue
        path=meta.with_suffix('.npy');target=stage_path(output,r['path'])
        if sha(path)!=m['sha256']:raise ValueError('Prior control SHA mismatch')
        scene=Path(r['path']).parts[3]
        geometry=root/f'source/hssd-hab/stages/{scene}.glb'
        if sha(geometry)!=m['geometry_sha256']:raise ValueError('Prior stage geometry changed')
        if not target.exists():
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
            atomic_json(target.with_suffix('.json'),dict(sha256=sha(target),reference=r,
                geometry_sha256=m['geometry_sha256'],reused_from=str(path)))
            reused+=1
    return len(refs),reused


def child(a):
    sys.path[:0]=[str(a.helpers),str(a.renderer)]
    from hssd_positive_pairs import make_geometry_simulator
    from render_hssd_paired_rirs import _add_audio_sensor
    from render_hssd_exp4_ring_rirs import _render_ring,_atomic_npy
    from hssd_cpu_rir_contract import require_cpu_device
    require_cpu_device(-1)
    plan=read(a.output/'plan.json');task=plan['tasks'][a.task]
    geometry=a.root/f'source/hssd-hab/stages/{task["scene"]}.glb'
    fingerprint=sha(geometry)
    sim=None
    for path in task['paths']:
        r=plan['refs'][path];target=stage_path(a.output,path)
        if target.exists() and target.with_suffix('.json').exists():
            meta=read(target.with_suffix('.json'))
            if meta['sha256']!=sha(target) or meta['geometry_sha256']!=fingerprint:
                raise ValueError('Native bank identity mismatch')
            continue
        if sim is None:
            sim=make_geometry_simulator(geometry,None)
            settings=SimpleNamespace(gpu_device_id=-1,sample_rate=48000,indirect_rays=20000,ray_depth=50,
                source_rays=512,diffraction=True,max_diffraction_order=10,acoustic_threads=2)
            sensor=_add_audio_sensor(sim,settings)
        base=int(Path(path).stem.rsplit('_',1)[1])
        rir=_render_ring(sim,sensor,np.asarray(r['xyz'],np.float32),base,.05)
        if rir.ndim!=2 or rir.shape[0]!=6 or not np.isfinite(rir).all() or not np.all(np.any(rir!=0,axis=-1)):
            raise ValueError('Invalid stage RIR')
        target.parent.mkdir(parents=True,exist_ok=True);_atomic_npy(target,rir)
        atomic_json(target.with_suffix('.json'),dict(sha256=sha(target),reference=r,geometry_sha256=fingerprint))
    atomic_json(a.output/'tasks'/f'{a.task:04d}.json',dict(status='complete',task=a.task))
    sys.stdout.flush();os._exit(0)


def render(a):
    total,reused=prepare(a.root,a.config,a.output)
    plan=read(a.output/'plan.json')
    def task(i):
        command=[sys.executable,'-u','-m','acoustic_branch.stage_comparison','--mode','child',
            '--root',str(a.root),'--config',str(a.config),'--output',str(a.output),
            '--renderer',str(a.renderer),'--helpers',str(a.helpers),'--task',str(i)]
        log=a.output/'logs'/f'{i:04d}.log';log.parent.mkdir(exist_ok=True)
        with log.open('a') as f:
            for attempt in range(3):
                try:
                    subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,timeout=240,check=True);return i
                except (subprocess.CalledProcessError,subprocess.TimeoutExpired):
                    if attempt==2:raise
    completed=0
    with ThreadPoolExecutor(max_workers=38) as pool:
        for _ in pool.map(task,range(len(plan['tasks']))):
            completed+=1
            atomic_json(a.output/'render_status.json',dict(status='running',completed_tasks=completed,
                total_tasks=len(plan['tasks']),physical_arrays=total,reused_arrays=reused,updated_unix=time.time()))
    atomic_json(a.output/'render_status.json',dict(status='complete',physical_arrays=total,
        reused_arrays=reused,test_accessed=False,updated_unix=time.time()))


def feature_pair(rir,config,module,legacy):
    return module.featurise(module.band_energy(rir,legacy),legacy),extract(rir,config)[1]


def rolled(pair,roll):
    stft,coh=pair
    return np.roll(stft.reshape(6,3,-1),roll,axis=0).reshape(stft.shape),np.roll(coh,roll,axis=1)


def summarize(rows):
    return dict(queries=len(rows),sampled_top1=float(np.mean([r['sampled_credit'] for r in rows])),
        **{f'recall_{t:g}m':float(np.mean([r['xy_error_m']<=t for r in rows])) for t in (.1,.5,1)},
        recall_1m_30deg=float(np.mean([r['xy_error_m']<=1 and r['yaw_error_deg']<=30 for r in rows])),
        median_xy_error_m=float(np.median([r['xy_error_m'] for r in rows])),
        median_yaw_error_deg=float(np.median([r['yaw_error_deg'] for r in rows])))


def evaluate(a):
    if read(a.output/'render_status.json')['status']!='complete':raise ValueError('Stage bank incomplete')
    config=read(a.config);queries,digest=load_manifest(a.root,'val',config)
    plan=read(a.output/'plan.json')
    if plan['manifest_sha256']!=digest:raise ValueError('Comparison manifest changed')
    spec=importlib.util.spec_from_file_location('grid_matched',a.source)
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    legacy=m.GridScoreConfig(sample_rate_hz=48000,direct_guard_samples=96,usable_samples=6144,nfft=256,hop=64,
        bands=((0,500),(500,1500),(1500,4000)),normalise='sum',distance='symmetric',shape_feature=True,feature='stft_band')
    allrefs={r['path']:r for q in queries for r in [q['observed']]+q['candidates']}
    jobs=[('original',p) for p in allrefs]+[('native',p) for p in plan['refs']]
    def load(job):
        kind,p=job;path=a.root/p if kind=='original' else stage_path(a.output,p)
        meta=read(path.with_suffix('.json'))
        if sha(path)!=meta['rir_sha256' if kind=='original' else 'sha256']:raise ValueError('Raw SHA mismatch')
        return job,feature_pair(np.load(path,allow_pickle=False),config['features'],m,legacy)
    with ThreadPoolExecutor(max_workers=12) as pool:cache=dict(pool.map(load,jobs))
    result=dict(test_accessed=False,queries=len(queries),scoring_source_sha256=sha(a.source),
        manifest_sha256=digest,plan_sha256=sha(a.output/'plan.json'),methods={},
        scope='Val 3 scenes, same observed RIR/candidates. Captured-pose bank NOT dense grid. No learning or weight tuning.')
    for kind,label in [('original','extruded_FP'),('native','native_stage')]:
        records={name:[] for name in ('STFT_L1','coherence_L2')}
        for scene in sorted({q['scene_id'] for q in queries}):
            scenequeries=[q for q in queries if q['scene_id']==scene]
            candidates=[q['candidates'][0] for q in scenequeries]
            bank=[rolled(cache[kind,r['path']],r['roll']) for r in candidates]
            observed=[rolled(cache['original',q['observed']['path']],q['observed']['roll']) for q in scenequeries]
            for mode,name,metric in [(0,'STFT_L1','cityblock'),(1,'coherence_L2','sqeuclidean')]:
                ob=np.asarray([v[mode].ravel() for v in observed]);fp=np.asarray([v[mode].ravel() for v in bank])
                distances=cdist(ob,fp,metric=metric)
                ids=distances.argmin(-1)
                for i,(q,j) in enumerate(zip(scenequeries,ids)):
                    o=ob[i]
                    pool=np.asarray([rolled(cache[kind,r['path']],r['roll'])[mode].ravel() for r in q['candidates']])
                    ds=cdist(o[None],pool,metric=metric)[0];win=ds==ds.min()
                    prediction=candidates[j]
                    xy=float(np.linalg.norm((np.array(q['observed']['xyz'])-prediction['xyz'])[[0,2]]))
                    yaw=abs((q['observed']['yaw']-prediction['yaw']+180)%360-180)
                    records[name].append(dict(scene=scene,sample_index=q['sample_index'],xyz=q['observed']['xyz'],
                        sampled_credit=float(win[0]/win.sum()),xy_error_m=xy,yaw_error_deg=yaw,
                        predicted_xyz=prediction['xyz'],predicted_yaw=prediction['yaw'],
                        gt_yaw=q['observed']['yaw'],image_path=q['image_path']))
        result['methods'][label]={name:dict(summary=summarize(rows),
            per_scene={s:summarize([r for r in rows if r['scene']==s]) for s in sorted({r['scene'] for r in rows})},
            records=rows) for name,rows in records.items()}
        atomic_json(a.output/'results.json',result)
    atomic_json(a.output/'status.json',dict(status='complete',test_accessed=False,updated_unix=time.time()))
    print({k:{m:v['summary'] for m,v in methods.items()} for k,methods in result['methods'].items()},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['render','child','evaluate'],required=True)
    for name in ('root','config','output'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('renderer','helpers','source'):p.add_argument('--'+name,type=Path)
    p.add_argument('--task',type=int)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    try:globals()[a.mode](a)
    except Exception as e:
        atomic_json(a.output/'status.json',dict(status='failed',error=repr(e)));raise

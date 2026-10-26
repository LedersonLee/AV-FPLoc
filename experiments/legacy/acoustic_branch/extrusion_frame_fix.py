"""Explicit Y-up stage loading + post-load bounds gate; diagnostic VAL bank."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np

from .contracts import atomic_json,immutable_json,sha
from .data import read,load_manifest
from .stage_comparison import summarize


def corrected_path(output,original):
    return output/'corrected_fp'/Path(original).relative_to('RIR/FP/val')


def child(a):
    sys.path[:0]=[str(a.helpers),str(a.renderer)]
    from hssd_positive_pairs import make_geometry_simulator
    from render_hssd_paired_rirs import _add_audio_sensor
    from render_hssd_exp4_ring_rirs import _render_ring,_atomic_npy
    from hssd_cpu_rir_contract import require_cpu_device
    require_cpu_device(-1)
    plan=read(a.output/'plan.json');task=plan['tasks'][a.task];scene=task['scene']
    stage=a.output/'stages'/f'{scene}.stage_config.json'
    if sha(stage)!=plan['stages'][scene]['config_sha256']:raise ValueError('Stage frame contract changed')
    sim=make_geometry_simulator(stage,None)
    bb=sim.get_active_scene_graph().get_root_node().cumulative_bb
    bounds=[np.array(bb.min).tolist(),np.array(bb.max).tolist()]
    np.testing.assert_allclose(bounds,plan['stages'][scene]['expected_bounds'],atol=1e-4,rtol=0)
    settings=SimpleNamespace(gpu_device_id=-1,sample_rate=48000,indirect_rays=20000,ray_depth=50,
        source_rays=512,diffraction=True,max_diffraction_order=10,acoustic_threads=2)
    sensor=_add_audio_sensor(sim,settings)
    for path in task['paths']:
        r=plan['refs'][path];target=corrected_path(a.output,path)
        if target.exists() and target.with_suffix('.json').exists():
            m=read(target.with_suffix('.json'))
            if m['sha256']!=sha(target) or m['plan_sha256']!=sha(a.output/'plan.json'):raise ValueError('Corrected RIR changed')
            continue
        xyz=np.array(r['xyz'])
        if not np.all(xyz>np.array(bounds[0])) or not np.all(xyz<np.array(bounds[1])):raise ValueError('Source outside loaded scene bounds')
        base=int(Path(path).stem.rsplit('_',1)[1])
        rir=_render_ring(sim,sensor,xyz.astype(np.float32),base,.05)
        if rir.ndim!=2 or rir.shape[0]!=6 or not np.isfinite(rir).all():raise ValueError('Bad corrected RIR')
        target.parent.mkdir(parents=True,exist_ok=True);_atomic_npy(target,rir)
        atomic_json(target.with_suffix('.json'),dict(sha256=sha(target),reference=r,
            plan_sha256=sha(a.output/'plan.json'),loaded_bounds=bounds,
            coordinate_frame='explicit stage up +Y, front -Z; verified post-load bounds'))
    sys.stdout.flush();os._exit(0)


def render(a):
    queries,digest=load_manifest(a.root,'val',read(a.config))
    refs={r['path']:r for q in queries for r in q['candidates']}
    stages={};tasks=[]
    for scene in sorted({q['scene_id'] for q in queries}):
        mesh=a.root/f'RIR/Geometry/{scene}/floorplan.glb'
        meta=read(mesh.with_suffix('.json'))
        if sha(mesh)!=meta['glb_sha256']:raise ValueError('Canonical extrusion mesh changed')
        stage=a.output/'stages'/f'{scene}.stage_config.json'
        immutable_json(stage,dict(render_asset=str(mesh),up=[0,1,0],front=[0,0,-1],units_to_meters=1.))
        stages[scene]=dict(config_sha256=sha(stage),mesh_sha256=sha(mesh),expected_bounds=meta['mesh']['bounds_xyz_m'])
        paths=sorted(p for p in refs if Path(p).parts[3]==scene)
        for start in range(0,len(paths),12):tasks.append(dict(scene=scene,paths=paths[start:start+12]))
    immutable_json(a.output/'plan.json',dict(refs=refs,tasks=tasks,stages=stages,
        manifest_sha256=digest,test_accessed=False,change='stage loading frame only; original GLB bytes unchanged'))
    def task(i):
        cmd=[sys.executable,'-u','-m','acoustic_branch.extrusion_frame_fix','--mode','child','--root',str(a.root),
            '--config',str(a.config),'--output',str(a.output),'--renderer',str(a.renderer),'--helpers',str(a.helpers),'--task',str(i)]
        log=a.output/'logs'/f'{i:04d}.log';log.parent.mkdir(exist_ok=True)
        with log.open('a') as f:
            for attempt in range(3):
                try:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=240);return i
                except (subprocess.TimeoutExpired,subprocess.CalledProcessError):
                    if attempt==2:raise
    with ThreadPoolExecutor(max_workers=38) as pool:
        for n,_ in enumerate(pool.map(task,range(len(tasks))),1):
            atomic_json(a.output/'render_status.json',dict(status='running',completed_tasks=n,total_tasks=len(tasks),updated_unix=time.time()))
    atomic_json(a.output/'render_status.json',dict(status='complete',physical_arrays=len(refs),test_accessed=False))


def evaluate(a):
    from . import planar_probe as p
    from .stage_comparison import feature_pair
    import importlib.util
    config=read(a.config);queries,digest=load_manifest(a.root,'val',config)
    plan=read(a.output/'plan.json')
    if plan['manifest_sha256']!=digest or read(a.output/'render_status.json')['status']!='complete':raise ValueError('Fixed bank incomplete')
    spec=importlib.util.spec_from_file_location('grid_fixed',a.source);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    legacy=m.GridScoreConfig(sample_rate_hz=48000,direct_guard_samples=96,usable_samples=6144,nfft=256,hop=64,
        bands=((0,500),(500,1500),(1500,4000)),normalise='sum',distance='symmetric',shape_feature=True,feature='stft_band')
    observed={q['observed']['path']:q['observed'] for q in queries}
    jobs=[('observed',path) for path in observed]+[('fixed',path) for path in plan['refs']]
    def load(job):
        kind,path=job;target=a.root/path if kind=='observed' else corrected_path(a.output,path)
        meta=read(target.with_suffix('.json'))
        if sha(target)!=meta['rir_sha256' if kind=='observed' else 'sha256']:raise ValueError('RIR SHA mismatch')
        if kind=='fixed' and meta['plan_sha256']!=sha(a.output/'plan.json'):raise ValueError('Wrong frame bank')
        return job,p.feature(np.load(target),config['features'],m,legacy)
    with ThreadPoolExecutor(max_workers=16) as pool:cache=dict(pool.map(load,jobs))
    methods={key:[] for key in ('STFT_L1','coherence_L2','common_arrival_L1')}
    for scene in sorted({q['scene_id'] for q in queries}):
        sq=[q for q in queries if q['scene_id']==scene]
        candidates=[q['candidates'][0] for q in sq]
        bank=[p.rolled(cache['fixed',r['path']],r['roll']) for r in candidates]
        obs=[p.rolled(cache['observed',q['observed']['path']],q['observed']['roll']) for q in sq]
        for method in methods:
            d=p.distances(obs,bank,method)
            for i,q in enumerate(sq):
                prediction=candidates[int(d[i].argmin())]
                candidate_features=[p.rolled(cache['fixed',r['path']],r['roll']) for r in q['candidates']]
                ds=p.distances([obs[i]],candidate_features,method)[0];win=ds==ds.min()
                methods[method].append(dict(scene=scene,sample_index=q['sample_index'],xyz=q['observed']['xyz'],
                    gt_yaw=q['observed']['yaw'],image_path=q['image_path'],sampled_credit=float(win[0]/win.sum()),
                    xy_error_m=float(np.linalg.norm((np.array(q['observed']['xyz'])-prediction['xyz'])[[0,2]])),
                    yaw_error_deg=abs((q['observed']['yaw']-prediction['yaw']+180)%360-180),
                    predicted_xyz=prediction['xyz'],predicted_yaw=prediction['yaw']))
    prior=read(a.root/'AcousticContrastive/diagnostics/planar_common_v2/results.json')
    result=dict(test_accessed=False,manifest_sha256=digest,plan_sha256=sha(a.output/'plan.json'),
        scope='FP world-frame ingestion bug fixed, unchanged mesh/map/candidate/observed RIR',methods={})
    for label,old in [('extruded_FP','extruded_RLR'),('native_stage','native_RLR')]:
        result['methods'][label]={method:dict(summary=summarize(prior['methods'][old][method]['records']),
            per_scene={s:summarize([r for r in prior['methods'][old][method]['records'] if r['scene']==s]) for s in plan['stages']},
            records=prior['methods'][old][method]['records']) for method in methods}
    result['methods']['corrected_extruded_FP']={method:dict(summary=summarize(rows),
        per_scene={s:summarize([r for r in rows if r['scene']==s]) for s in plan['stages']},records=rows) for method,rows in methods.items()}
    atomic_json(a.output/'results.json',result);atomic_json(a.output/'status.json',dict(status='complete',test_accessed=False))
    print({k:{m:v['summary'] for m,v in ms.items()} for k,ms in result['methods'].items()},flush=True)


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

"""CPU-only repeatability / native-stage / furniture control, val scenes only."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np

from .contracts import atomic_json, sha
from .data import read, load_manifest


def child(root, output, scene, condition, renderer, helpers):
    sys.path[:0]=[str(helpers),str(renderer)]
    from hssd_positive_pairs import make_geometry_simulator, furnished_registry, proxy_quads, write_glb
    from render_hssd_paired_rirs import _add_audio_sensor
    from render_hssd_exp4_ring_rirs import _render_ring, _atomic_npy
    from hssd_cpu_rir_contract import require_cpu_device
    require_cpu_device(-1)
    plan=read(output/'plan.json')
    queries=plan['scenes'][scene]
    source=root/'source/hssd-hab'
    registry=None
    if condition=='fp_inward':
        from PIL import Image
        meta=read(root/f'2DFP/{scene}/floorplan_metadata.json')
        height=read(root/f'RIR/Geometry/{scene}/floorplan.json')
        free=np.asarray(Image.open(root/f'2DFP/{scene}/map.png').convert('L'))==255
        quads=proxy_quads(free,meta['map_spacing_m'],meta['world_bounds_m']['x_min'],
            meta['world_bounds_m']['z_min'],height['floor_y_m'],height['ceiling_y_m'])
        geometry=output/'inward_geometry'/scene/'floorplan.glb'
        write_glb(geometry,((points,(-np.array(normal)).tolist()) for points,normal in quads))
    elif condition=='fp_repeat':
        geometry=root/f'RIR/Geometry/{scene}/floorplan.glb'
    elif condition in ('native_stage','native_stage_candidates'):
        geometry=source/f'stages/{scene}.glb'
    elif condition=='furnished_repeat':
        geometry=source/f'scenes/{scene}.scene_instance.json'
        registry,expected=furnished_registry(root,source,scene)
    else:raise ValueError(condition)
    sim=make_geometry_simulator(geometry,registry)
    if condition=='furnished_repeat' and sim.get_rigid_object_manager().get_num_objects()!=expected:
        raise ValueError('Object count mismatch')
    settings=SimpleNamespace(gpu_device_id=-1,sample_rate=48000,indirect_rays=20000,ray_depth=50,
        source_rays=512,diffraction=True,max_diffraction_order=10,acoustic_threads=2)
    sensor=_add_audio_sensor(sim,settings)
    entries=[(q,r,i) for q in queries for i,r in enumerate(q['candidates'])] if condition=='native_stage_candidates' else [(q,q['observed'],None) for q in queries]
    for q,ref,index in entries:
        stem=f'{q["sample_index"]:05d}'+('' if index is None else f'_{index}')
        path=output/'rirs'/condition/scene/f'{stem}.npy'
        if path.exists() and path.with_suffix('.json').exists():
            if sha(path)!=read(path.with_suffix('.json'))['sha256']:raise ValueError('Control integrity mismatch')
            continue
        base=int(Path(ref['path']).stem.rsplit('_',1)[1])
        value=_render_ring(sim,sensor,np.asarray(ref['xyz'],np.float32),base,.05)
        if value.ndim!=2 or value.shape[0]!=6 or not np.isfinite(value).all():raise ValueError('Bad RIR')
        path.parent.mkdir(parents=True,exist_ok=True)
        _atomic_npy(path,value)
        atomic_json(path.with_suffix('.json'),dict(sha256=sha(path),geometry_sha256=sha(geometry),
            condition=condition,query=q,reference=ref,base=base,geometry_path=str(geometry)))
    atomic_json(output/f'{scene}_{condition}_complete.json',dict(status='complete',records=len(entries)))
    # Native shutdown has a known thread-join hang. Every output has been
    # synchronously committed and checksummed; do not wait on simulator teardown.
    sys.stdout.flush()
    os._exit(0)


def run(a):
    output=a.output
    output.mkdir(parents=True,exist_ok=True)
    if not (output/'plan.json').exists():
        config=read(a.config)
        queries,digest=load_manifest(a.root,'val',config)
        scenes={}
        for scene in sorted({q['scene_id'] for q in queries}):
            unique={}
            for q in queries:
                if q['scene_id']==scene:unique.setdefault(tuple(q['observed']['xyz']),q)
            values=list(unique.values())
            scenes[scene]=[values[int(i)] for i in np.linspace(0,len(values)-1,8)]
        atomic_json(output/'plan.json',dict(scenes=scenes,manifest_sha256=digest,test_accessed=False,
            selection='8 evenly indexed unique positions per val scene, not selected by error'))
    plan=read(output/'plan.json')
    tasks=[(s,c) for s in plan['scenes'] for c in ('fp_repeat','native_stage','furnished_repeat','native_stage_candidates','fp_inward')]
    def task(t):
        s,c=t
        if (output/f'{s}_{c}_complete.json').exists():
            return dict(scene=s,condition=c,status='reused_complete')
        command=[sys.executable,'-u','-m','acoustic_branch.geometry_control','--child',
            '--root',str(a.root),'--config',str(a.config),'--output',str(output),
            '--renderer',str(a.renderer),'--helpers',str(a.helpers),'--scene',s,'--condition',c]
        log=output/f'{s}_{c}.log'
        with log.open('a') as f:
            for attempt in range(3):
                try:
                    subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,timeout=240,check=True)
                    return dict(scene=s,condition=c,status='complete')
                except (subprocess.TimeoutExpired,subprocess.CalledProcessError):
                    if attempt==2:raise
    results=[]
    with ThreadPoolExecutor(max_workers=9) as pool:
        for result in pool.map(task,tasks):
            results.append(result)
            atomic_json(output/'status.json',dict(status='running',tasks=results,updated_unix=time.time()))
    atomic_json(output/'status.json',dict(status='complete',tasks=results,test_accessed=False,updated_unix=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','config','output','renderer','helpers'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--child',action='store_true');p.add_argument('--scene');p.add_argument('--condition')
    a=p.parse_args()
    if a.child:child(a.root,a.output,a.scene,a.condition,a.renderer,a.helpers)
    else:
        try:run(a)
        except Exception as e:
            atomic_json(a.output/'status.json',dict(status='failed',error=repr(e)))
            raise

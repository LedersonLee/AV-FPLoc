"""Small native-renderer asset-ingestion controls; never rewrites shared FP."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image

from .contracts import atomic_json,sha
from .data import read


def write_mesh(path,quads,inward=False,both=False,obj=False):
    positions=[];normals=[];indices=[]
    for points,normal in quads:
        q=np.asarray(points,float);n=np.asarray(normal,float)*(-1 if inward else 1)
        if np.dot(np.cross(q[1]-q[0],q[2]-q[0]),n)<0:q=q[[0,3,2,1]]
        orientations=[(q,n),(q[[0,3,2,1]],-n)] if both else [(q,n)]
        for points,normal in orientations:
            i=len(positions);positions.extend(points);normals.extend([normal]*4)
            indices.extend([[i,i+1,i+2],[i,i+2,i+3]])
    v=np.asarray(positions,dtype='<f4');n=np.asarray(normals,dtype='<f4');f=np.asarray(indices,dtype='<u4')
    path.parent.mkdir(parents=True,exist_ok=True)
    if obj:
        text='\n'.join(['o acoustic_control']+['v '+' '.join(map(str,x)) for x in v]+['vn '+' '.join(map(str,x)) for x in n]+['f '+' '.join(f'{j+1}//{j+1}' for j in row) for row in f])+'\n'
        path.write_text(text);return
    vb,nb,fb=v.tobytes(),n.tobytes(),f.tobytes();binary=vb+nb+fb
    doc=dict(asset=dict(version='2.0',generator='explicit-normal acoustic ingestion control'),scene=0,
        scenes=[dict(nodes=[0])],nodes=[dict(mesh=0)],
        meshes=[dict(primitives=[dict(attributes=dict(POSITION=0,NORMAL=1),indices=2,material=0)])],
        materials=[dict(name='acoustic_control',doubleSided=True,pbrMetallicRoughness=dict(baseColorFactor=[.7,.7,.7,1],metallicFactor=0,roughnessFactor=1))],
        buffers=[dict(byteLength=len(binary))],bufferViews=[dict(buffer=0,byteOffset=0,byteLength=len(vb),target=34962),
            dict(buffer=0,byteOffset=len(vb),byteLength=len(nb),target=34962),
            dict(buffer=0,byteOffset=len(vb)+len(nb),byteLength=len(fb),target=34963)],
        accessors=[dict(bufferView=0,componentType=5126,count=len(v),type='VEC3',min=v.min(0).tolist(),max=v.max(0).tolist()),
            dict(bufferView=1,componentType=5126,count=len(n),type='VEC3'),dict(bufferView=2,componentType=5125,count=f.size,type='SCALAR')])
    js=json.dumps(doc,separators=(',',':')).encode();js+=b' '*((-len(js))%4)
    binary+=b'\0'*((-len(binary))%4)
    path.write_bytes(struct.pack('<III',0x46546c67,2,28+len(js)+len(binary))+struct.pack('<II',len(js),0x4e4f534a)+js+struct.pack('<II',len(binary),0x004e4942)+binary)


def child(a):
    sys.path[:0]=[str(a.helpers),str(a.renderer)]
    from hssd_positive_pairs import make_geometry_simulator,proxy_quads
    from render_hssd_paired_rirs import _add_audio_sensor
    from render_hssd_exp4_ring_rirs import _render_ring,_atomic_npy
    from hssd_cpu_rir_contract import require_cpu_device
    require_cpu_device(-1)
    q=read(a.output/'plan.json')['scenes'][a.scene][0]
    height=read(a.root/f'RIR/Geometry/{a.scene}/floorplan.json')
    if a.kind.startswith('box'):
        from .planar_paths import segments_from_mask
        free=np.ones((1,1),bool)
        x,z=np.array(q['observed']['xyz'])[[0,2]]
        quads=list(proxy_quads(free,4.,x-2+2,z-2+2,height['floor_y_m'],height['ceiling_y_m']))
    else:
        meta=read(a.root/f'2DFP/{a.scene}/floorplan_metadata.json')
        free=np.asarray(Image.open(a.root/f'2DFP/{a.scene}/map.png').convert('L'))==255
        quads=list(proxy_quads(free,meta['map_spacing_m'],meta['world_bounds_m']['x_min'],meta['world_bounds_m']['z_min'],height['floor_y_m'],height['ceiling_y_m']))
    obj=a.kind.endswith('obj')
    if a.kind=='explicit_stage':
        path=a.output/'geometries'/a.scene/'explicit.stage_config.json'
        atomic_json(path,dict(render_asset=str(a.root/f'RIR/Geometry/{a.scene}/floorplan.glb'),
            up=[0,1,0],front=[0,0,-1],units_to_meters=1.0))
    else:
        path=a.output/'geometries'/a.scene/(a.kind+('.obj' if obj else '.glb'))
        write_mesh(path,quads,inward='inward' in a.kind or a.kind.startswith('box'),both='both' in a.kind,obj=obj)
    sim=make_geometry_simulator(path,None)
    bb=sim.get_active_scene_graph().get_root_node().cumulative_bb
    bounds=dict(min=np.array(bb.min).tolist(),max=np.array(bb.max).tolist())
    if a.kind=='explicit_stage':
        np.testing.assert_allclose([bounds['min'],bounds['max']],height['mesh']['bounds_xyz_m'],atol=1e-4,rtol=0)
    settings=SimpleNamespace(gpu_device_id=-1,sample_rate=48000,indirect_rays=20000,ray_depth=50,
        source_rays=512,diffraction=True,max_diffraction_order=10,acoustic_threads=2)
    sensor=_add_audio_sensor(sim,settings)
    base=int(Path(q['observed']['path']).stem.rsplit('_',1)[1])
    rir=_render_ring(sim,sensor,np.asarray(q['observed']['xyz'],np.float32),base,.05)
    target=a.output/'rirs'/a.scene/f'{a.kind}.npy';target.parent.mkdir(parents=True,exist_ok=True);_atomic_npy(target,rir)
    atomic_json(target.with_suffix('.json'),dict(sha256=sha(target),geometry_sha256=sha(path),
        source=q['observed']['xyz'],bounds=bounds,reference=q,kind=a.kind))
    sys.stdout.flush();os._exit(0)


def run(a):
    plan=read(a.root/'AcousticContrastive/diagnostics/learning_failure_v1/geometry_control/plan.json')
    atomic_json(a.output/'plan.json',plan)
    tasks=[(s,k) for s in plan['scenes'] for k in ('normal_outward','normal_inward','both_normals','inward_obj','box_glb','box_obj','explicit_stage')]
    def task(t):
        scene,kind=t
        target=a.output/'rirs'/scene/f'{kind}.npy'
        if target.exists() and target.with_suffix('.json').exists():
            if sha(target)!=read(target.with_suffix('.json'))['sha256']:raise ValueError('Prior control integrity failure')
            return dict(scene=scene,kind=kind,status='reused_complete')
        cmd=[sys.executable,'-u','-m','acoustic_branch.mesh_ingestion_probe','--child','--root',str(a.root),
            '--output',str(a.output),'--renderer',str(a.renderer),'--helpers',str(a.helpers),'--scene',scene,'--kind',kind]
        log=a.output/f'{scene}_{kind}.log'
        with log.open('a') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=180)
        return dict(scene=scene,kind=kind,status='complete')
    results=[]
    with ThreadPoolExecutor(max_workers=18) as pool:
        for value in pool.map(task,tasks):
            results.append(value);atomic_json(a.output/'status.json',dict(status='running',tasks=results))
    atomic_json(a.output/'status.json',dict(status='complete',tasks=results,test_accessed=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','output','renderer','helpers'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--child',action='store_true');p.add_argument('--scene');p.add_argument('--kind')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    try:child(a) if a.child else run(a)
    except Exception as e:
        atomic_json(a.output/'status.json',dict(status='failed',error=repr(e)));raise

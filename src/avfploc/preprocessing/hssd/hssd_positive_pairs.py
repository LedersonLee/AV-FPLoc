#!/usr/bin/env python3
"""Prepare and render split-preserving, CPU-only HSSD acoustic positives.

FP geometry is extruded from the approved 1-cm raster, never the native stage.
Camera centres come from capture metadata, never from a new navmesh sample.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import time
import traceback
import uuid

import numpy as np
from PIL import Image

PROTOCOL = dict(sample_rate_hz=48000, indirect_rays=20000, ray_depth=50,
                source_rays=512, diffraction=True, max_diffraction_order=10,
                ring_radius_m=.05, materials=False, transmission=False)
RING = np.array([np.pi, 4*np.pi/3, 5*np.pi/3, 2*np.pi, np.pi/3, 2*np.pi/3])
VERSION = 'positive_pairs_v1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n'); tmp.replace(path)


def immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'Existing immutable contract differs: {path}')
    else:
        atomic_json(path, value)


def recipe(yaw):
    acoustic = (-float(yaw)) % 360
    base = acoustic % 60
    if not np.isclose(base, round(base)):
        raise ValueError('Only integer acoustic headings supported')
    return int(round(base)), -int(round((acoustic-base)/60))


def runs(vector):
    edges = np.diff(np.r_[False, np.asarray(vector, bool), False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def rectangles(mask):
    """Lossless row-run merging: no downsampling/closing or component selection."""
    active = {}
    for row in range(len(mask)+1):
        current = set(runs(mask[row])) if row < len(mask) else set()
        for segment in list(active):
            if segment not in current:
                yield active.pop(segment), row, segment[0], segment[1]
        for segment in current:
            active.setdefault(segment, row)


def proxy_quads(free, spacing, xmin, zmin, floor, ceiling):
    """Raster coordinates refer to pixel CENTRES; boundaries are half a cell away."""
    x = lambda col: xmin + (col-.5)*spacing
    z = lambda row: zmin + (row-.5)*spacing
    for r0, r1, c0, c1 in rectangles(free):
        yield [[x(c0),floor,z(r0)],[x(c1),floor,z(r0)],
               [x(c1),floor,z(r1)],[x(c0),floor,z(r1)]], [0,-1,0]
        yield [[x(c0),ceiling,z(r0)],[x(c1),ceiling,z(r0)],
               [x(c1),ceiling,z(r1)],[x(c0),ceiling,z(r1)]], [0,1,0]
    padded = np.pad(free, 1)
    top = free & ~padded[:-2,1:-1]
    bottom = free & ~padded[2:,1:-1]
    left = free & ~padded[1:-1,:-2]
    right = free & ~padded[1:-1,2:]
    for boundary, edge, normal in ((top,0,[0,0,-1]),(bottom,1,[0,0,1])):
        for row in np.flatnonzero(boundary.any(axis=1)):
            for c0,c1 in runs(boundary[row]):
                zz = z(row+edge)
                yield [[x(c0),floor,zz],[x(c1),floor,zz],
                       [x(c1),ceiling,zz],[x(c0),ceiling,zz]], normal
    for boundary, edge, normal in ((left,0,[-1,0,0]),(right,1,[1,0,0])):
        for col in np.flatnonzero(boundary.any(axis=0)):
            for r0,r1 in runs(boundary[:,col]):
                xx = x(col+edge)
                yield [[xx,floor,z(r0)],[xx,floor,z(r1)],
                       [xx,ceiling,z(r1)],[xx,ceiling,z(r0)]], normal


def write_glb(path, quads):
    vertices=[]; faces=[]
    for points, normal in quads:
        q = np.asarray(points, np.float64)
        if np.dot(np.cross(q[1]-q[0], q[2]-q[0]), normal) < 0:
            q = q[[0,3,2,1]]
        i=len(vertices); vertices.extend(q.tolist())
        faces.extend([[i,i+1,i+2],[i,i+2,i+3]])
    v=np.asarray(vertices,dtype='<f4'); f=np.asarray(faces,dtype='<u4')
    if not len(v): raise ValueError('Empty proxy')
    vb=v.tobytes(); fb=f.tobytes(); binary=vb+fb
    doc=dict(asset=dict(version='2.0',generator='HSSD lossless canonical-map extrusion'),
        scene=0, scenes=[dict(nodes=[0])], nodes=[dict(mesh=0)],
        meshes=[dict(primitives=[dict(attributes=dict(POSITION=0),indices=1,material=0)])],
        materials=[dict(doubleSided=True,pbrMetallicRoughness=dict(baseColorFactor=[.7,.7,.7,1],metallicFactor=0,roughnessFactor=1))],
        buffers=[dict(byteLength=len(binary))],
        bufferViews=[dict(buffer=0,byteOffset=0,byteLength=len(vb),target=34962),
                     dict(buffer=0,byteOffset=len(vb),byteLength=len(fb),target=34963)],
        accessors=[dict(bufferView=0,componentType=5126,count=len(v),type='VEC3',min=v.min(0).tolist(),max=v.max(0).tolist()),
                   dict(bufferView=1,componentType=5125,count=f.size,type='SCALAR')])
    js=json.dumps(doc,separators=(',',':')).encode(); js += b' '*((-len(js))%4)
    binary += b'\0'*((-len(binary))%4)
    blob=struct.pack('<III',0x46546c67,2,12+8+len(js)+8+len(binary))
    blob+=struct.pack('<II',len(js),0x4e4f534a)+js+struct.pack('<II',len(binary),0x004e4942)+binary
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_bytes()!=blob: raise ValueError(f'Existing proxy differs: {path}')
    else:
        temp=path.with_suffix('.'+uuid.uuid4().hex+'.tmp');temp.write_bytes(blob);temp.replace(path)
    return dict(vertices=len(v),triangles=len(f),bounds_xyz_m=[v.min(0).tolist(),v.max(0).tolist()])


def inside_map(free, meta, xyz):
    s=meta['map_spacing_m']; b=meta['world_bounds_m']
    col=int(np.floor((xyz[0]-b['x_min'])/s+.5))
    row=int(np.floor((xyz[2]-b['z_min'])/s+.5))
    return 0<=row<free.shape[0] and 0<=col<free.shape[1] and bool(free[row,col])


def furnished_registry(root, source, scene):
    """Load the SAME assets, but not the entire dataset's object-template library."""
    scene_path=source/'scenes'/f'{scene}.scene_instance.json'
    instance=json.loads(scene_path.read_text())
    handles=sorted({str(o['template_name']) for o in instance['object_instances']})
    paths=[]
    for handle in handles:
        name=Path(handle).name
        candidates=[source/'objects'/name[0]/f'{name}.object_config.json',
                    source/'objects/openings'/f'{name}.object_config.json',
                    source/'objects'/f'{name}.object_config.json',
                    source/'objects/decomposed'/name.split('_part_')[0]/f'{name}.object_config.json']
        found=next((p for p in candidates if p.is_file()),None)
        if found is None: raise FileNotFoundError(f'Missing furnished template: {handle}')
        paths.append(found)
    directory=root/'RIR/Geometry'/scene
    rel=lambda p:os.path.relpath(p,directory)
    registry=json.loads((source/'hssd-hab.scene_dataset_config.json').read_text())
    registry['stages']={'paths':{'.json':[rel(source/'stages'/f'{scene}.stage_config.json')]}}
    registry['objects']={'paths':{'.json':[rel(p) for p in paths]}}
    registry['scene_instances']['paths']={'.json':[rel(scene_path)]}
    registry['semantic_scene_descriptor_instances']={k:rel(source/v) for k,v in registry.get('semantic_scene_descriptor_instances',{}).items()}
    path=directory/'furnished.scene_dataset_config.json';immutable_json(path,registry)
    return path,len(instance['object_instances'])


def make_geometry_simulator(geometry, registry):
    """Keep a software depth context, but do not read/decode RGB textures for audio."""
    import habitat_sim
    backend=habitat_sim.SimulatorConfiguration()
    backend.scene_id=str(geometry)
    if registry is not None: backend.scene_dataset_config_file=str(registry)
    backend.load_semantic_mesh=False;backend.enable_physics=False
    backend.gpu_device_id=-1;backend.requires_textures=False
    agent=habitat_sim.agent.AgentConfiguration()
    dummy=habitat_sim.CameraSensorSpec();dummy.uuid='dummy_depth'
    dummy.resolution=[16,16];dummy.sensor_type=habitat_sim.SensorType.DEPTH
    agent.sensor_specifications=[dummy]
    return habitat_sim.Simulator(habitat_sim.Configuration(backend,[agent]))


def make_extruded_simulator(geometry):
    """Never let Habitat infer the coordinate frame of a custom extrusion GLB.

    The native generic-GLB stage defaults can rotate (X,Y,Z)->(X,Z,-Y).
    File bounds alone do not validate the acoustic scene: verify AFTER loading.
    """
    geometry=Path(geometry)
    contract=json.loads(geometry.with_suffix('.json').read_text())
    if sha(geometry)!=contract['glb_sha256']:raise ValueError('Extrusion mesh changed')
    stage=geometry.with_name(geometry.stem+'.y_up.stage_config.json')
    immutable_json(stage,dict(render_asset=geometry.name,up=[0,1,0],front=[0,0,-1],units_to_meters=1.0))
    sim=make_geometry_simulator(stage,None)
    box=sim.get_active_scene_graph().get_root_node().cumulative_bb
    bounds=[np.asarray(box.min).tolist(),np.asarray(box.max).tolist()]
    np.testing.assert_allclose(bounds,contract['mesh']['bounds_xyz_m'],atol=1e-4,rtol=0)
    return sim,dict(verified=True,up=[0,1,0],front=[0,0,-1],stage_config_sha256=sha(stage),loaded_bounds=bounds)


def height_contract(regions, policy='uniform'):
    heights={(round(float(r['floor_height']),4),round(float(r['extrusion_height']),4)) for r in regions}
    if len(heights)==1:
        floor,height=next(iter(heights))
        if height<=1.55: raise ValueError('Ceiling below sensor')
        return floor,height,None
    if policy!='area_weighted_mean': raise ValueError(f'Nonuniform height requires review: {heights}')
    floors={v[0] for v in heights}
    if len(floors)!=1: raise ValueError(f'Different floor levels cannot be averaged: {floors}')
    records=[];valid=[]
    for region in regions:
        name=str(region.get('name',''));h=float(region['extrusion_height'])
        polygon=np.asarray(region['poly_loop'],float)[:,[0,2]]
        area=abs(np.dot(polygon[:,0],np.roll(polygon[:,1],1))-np.dot(polygon[:,1],np.roll(polygon[:,0],1)))/2
        excluded=any(word in name.lower() for word in ('terrace','balcony','outdoor','patio')) or h<=1.55 or not np.isfinite(h) or area<=0
        records.append(dict(name=name,height_m=h,area_m2=float(area),included=not excluded))
        if not excluded: valid.append((h,area))
    if not valid: raise ValueError('No physically valid indoor ceiling heights')
    height=float(np.average([v[0] for v in valid],weights=[v[1] for v in valid]))
    return next(iter(floors)),height,dict(policy='indoor_region_area_weighted_mean',height_m=height,regions=records,
        authorization='User approved approximate mean scene height; invalid heights below sensor and outdoor regions excluded')


def prepare_scene(root, source, split, entry, split_hash, height_policy='uniform'):
    scene=entry['scene_id']; rgb=root/'UnLoc/RGB'/scene
    pilot_path=rgb/'pilot_manifest.json'; pilot=json.loads(pilot_path.read_text())
    captures={int(r['sample_index']):r for r in pilot['records'] if r['accepted']}
    replacement_path=root/'UnLoc/furnished_depth_qa_v2_nohit_only/replacements'/scene/'replacement_manifest.json'
    replacements=json.loads(replacement_path.read_text())
    if replacements['split']!=split or replacements['scene_id']!=scene: raise ValueError('Replacement split mismatch')
    for r in replacements['records']:
        if r.get('accepted'):
            if int(r['sample_index']) in captures: raise ValueError('Duplicate capture ID')
            captures[int(r['sample_index'])]=r
    rows_path=root/entry['sample_metadata_path']
    rows=[json.loads(line) for line in rows_path.read_text().splitlines() if line]
    if len(rows)!=entry['accepted_frames']: raise ValueError('Adapter count mismatch')
    map_path=root/'2DFP'/scene/'map.png'; meta_path=map_path.with_name('floorplan_metadata.json')
    meta=json.loads(meta_path.read_text()); free=np.asarray(Image.open(map_path).convert('L'))==255
    if meta['raster_view']!='+Y top-down: +X right, +Z down': raise ValueError('Unexpected map axes')
    if free[0].any() or free[-1].any() or free[:,0].any() or free[:,-1].any(): raise ValueError('Map exterior not closed')
    semantic_path=source/'semantics/scenes'/f'{scene}.semantic_config.json'
    semantics=json.loads(semantic_path.read_text())
    # Semantic exporters contain micrometre-scale floating-point floor noise.
    # Collapse only sub-0.1-mm differences, never actual room-height differences.
    floor,height,aggregation=height_contract(semantics['region_annotations'],height_policy)
    ceiling=floor+height
    geometry=root/'RIR/Geometry'/scene/'floorplan.glb'
    mesh=write_glb(geometry,proxy_quads(free,meta['map_spacing_m'],meta['world_bounds_m']['x_min'],meta['world_bounds_m']['z_min'],floor,ceiling))
    geom_contract=dict(scene_id=scene,geometry='canonical_map_extrusion',map_path=str(map_path.relative_to(root)),
        map_sha256=sha(map_path),metadata_sha256=sha(meta_path),floor_y_m=floor,ceiling_y_m=ceiling,
        height_source='semantic region floor_height + extrusion_height; known-height assumption, not a ceiling estimator',
        semantic_sha256=sha(semantic_path),axes='native HSSD Y-up; X right/Z down on raster; no recenter or axis reflection',
        resolution_m=meta['map_spacing_m'],resampling='none; exact free-cell union with merged coplanar rectangles',
        openings='preserve canonical map; internal openings free; exterior closed',glb_sha256=sha(geometry),mesh=mesh)
    if aggregation is not None:
        geom_contract['height_aggregation']=aggregation
        geom_contract['height_source']='User-approved approximate scene mean from valid indoor semantic-region heights'
    immutable_json(geometry.with_suffix('.json'),geom_contract)
    centres={};queries=[];excluded=[]
    for row in rows:
        if row['scene_id']!=scene: raise ValueError('Cross-scene row')
        capture=captures[int(row['sample_index'])]; xyz=list(map(float,capture['pose_xyz_m']))
        yaw=float(row['pose_xzyaw'][2]);base,roll=recipe(yaw)
        np.testing.assert_allclose([xyz[0],xyz[2],capture['logical_yaw_deg']],row['pose_xzyaw'],rtol=0,atol=1e-8)
        if not np.isclose(xyz[1],floor+1.5,rtol=0,atol=1e-4): raise ValueError('Capture floor/height mismatch; never snap it')
        offsets=np.column_stack([.05*np.cos(RING+np.deg2rad(base)),np.zeros(6),-.05*np.sin(RING+np.deg2rad(base))])
        if not all(inside_map(free,meta,p) for p in np.vstack([xyz,np.asarray(xyz)+offsets])):
            excluded.append(dict(sample_index=row['sample_index'],reason='array_intersects_canonical_FP_wall',centre_xyz_m=xyz,base_acoustic_yaw_deg=base))
            continue
        key=tuple(xyz); p=centres.setdefault(key,dict(centre_xyz_m=xyz,base_headings=[]))
        if base not in p['base_headings']: p['base_headings'].append(base)
        queries.append(dict(sample_index=row['sample_index'],image_path=row['image_path'],centre_xyz_m=xyz,
                            world_yaw_deg=yaw,base_acoustic_yaw_deg=base,channel_roll=roll))
    positions=[]
    for i,key in enumerate(sorted(centres)):
        p=centres[key];p['pose_id']=i;p['base_headings'].sort();positions.append(p)
    ids={tuple(p['centre_xyz_m']):p['pose_id'] for p in positions}
    for q in queries:
        q['pose_id']=ids[tuple(q['centre_xyz_m'])]
        stem=f'records/p{q["pose_id"]:05d}/base_acoustic_yaw_{q["base_acoustic_yaw_deg"]:03d}.npy'
        q['fp_rir_path']=f'RIR/FP/{split}/{scene}/{stem}'
        q['observed_rir_path']=f'RIR/Observed/{split}/{scene}/{stem}'
    scene_path=source/'scenes'/f'{scene}.scene_instance.json'
    schedule=dict(version=VERSION,scene_id=scene,split=split,split_sha256=split_hash,
        adapter_sha256=sha(rows_path),capture_sha256=sha(pilot_path),replacement_sha256=sha(replacement_path),
        geometry_path=str(geometry.relative_to(root)),geometry_sha256=sha(geometry),height_contract=geom_contract,
        scene_path=str(scene_path.relative_to(source)),scene_sha256=sha(scene_path),
        positions=positions,queries=queries,excluded=excluded,query_count=len(queries),
        position_count=len(positions),physical_records_per_role=sum(len(p['base_headings']) for p in positions),
        acoustics=PROTOCOL,microphone_angles_rad=RING.tolist(),
        heading_contract='world yaw 0=+X, positive toward+Z; acoustic a=(-worldYaw)%360; base=a%60; roll=-(a-base)/60',
        centre_contract='Exact captured RGB world XYZ; source at array centre; no navmesh snapping or added height')
    path=root/'RIR/manifests'/VERSION/'schedules'/split/f'{scene}.json';immutable_json(path,schedule)
    return dict(scene_id=scene,split=split,schedule_path=str(path.relative_to(root)),query_count=len(queries),
                excluded_queries=len(excluded),positions=len(positions),physical_records_per_role=schedule['physical_records_per_role'])


def prepare(args):
    root=args.root; source=args.source; split_path=root/'manifests/scene_splits_v1.json'
    splits=json.loads(split_path.read_text());split_hash=sha(split_path);all_ids=[]
    for key in ('train','val','test'): all_ids.extend(splits[key])
    if len(all_ids)!=len(set(all_ids)): raise ValueError('Scene split overlap')
    todo=[]
    for split in args.splits:
        manifest=json.loads((root/'UnLoc/frame_adapter_v4_furnished_qa/manifests'/f'{split}_scenes.json').read_text())
        if manifest['split']!=split or {e['scene_id'] for e in manifest['scenes']}!=set(splits[split]): raise ValueError('Split manifest mismatch')
        todo.extend((split,e) for e in manifest['scenes'])
    good=[];errors=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(prepare_scene,root,source,split,e,split_hash,args.height_policy):(split,e['scene_id']) for split,e in todo}
        for future in as_completed(futures):
            split,scene=futures[future]
            try: good.append(future.result());print(f'[prepared] {scene}',flush=True)
            except Exception as exc: errors.append(dict(split=split,scene_id=scene,error=str(exc)));print(f'[blocked] {scene}: {exc}',flush=True)
    good.sort(key=lambda r:(r['split'],r['scene_id']))
    index=dict(version=VERSION,split_sha256=split_hash,scenes=good,blocked_scenes=errors,
        counts=dict(scenes=len(good),queries=sum(r['query_count'] for r in good),positions=sum(r['positions'] for r in good),
                    physical_records_both_roles=2*sum(r['physical_records_per_role'] for r in good),excluded_queries=sum(r['excluded_queries'] for r in good)))
    index_path=root/'RIR/manifests'/VERSION/'index.json'
    if index_path.exists():
        previous=json.loads(index_path.read_text())
        if previous['split_sha256']!=split_hash: raise ValueError('Split contract changed')
        current={(r['split'],r['scene_id']):r for r in good}
        for r in previous['scenes']:
            if current.get((r['split'],r['scene_id']))!=r: raise ValueError('Existing scene schedule changed')
    atomic_json(index_path,index)
    print(json.dumps(index['counts']),flush=True)


def render_task(task):
    root,source,renderer_dir,entry,role,pose_ids,threads=task
    root=Path(root);source=Path(source);sys.path.insert(0,renderer_dir)
    from types import SimpleNamespace
    from hssd_cpu_rir_contract import require_cpu_device
    from render_hssd_paired_rirs import _add_audio_sensor
    from render_hssd_exp4_ring_rirs import _render_ring,_atomic_npy
    require_cpu_device(-1)
    schedule_path=root/entry['schedule_path']; schedule=json.loads(schedule_path.read_text());fingerprint=sha(schedule_path)
    if sha(root/'manifests/scene_splits_v1.json')!=schedule['split_sha256']: raise ValueError('Split hash changed')
    scene=schedule['scene_id'];split=schedule['split']
    geometry=root/schedule['geometry_path'] if role=='FP' else source/schedule['scene_path']
    expected=schedule['geometry_sha256'] if role=='FP' else schedule['scene_sha256']
    if sha(geometry)!=expected: raise ValueError('Geometry changed')
    settings=SimpleNamespace(gpu_device_id=-1,sample_rate=48000,indirect_rays=20000,ray_depth=50,
        source_rays=512,diffraction=True,max_diffraction_order=10,acoustic_threads=threads)
    # No navmesh build: exact captured positions must never be moved or rejected by resampling.
    sim=None;loaded_frame=None;written=0;skipped=0;start=time.monotonic()
    try:
        for p in schedule['positions']:
            if p['pose_id'] not in pose_ids: continue
            for base in p['base_headings']:
                # Positive defaults are unchanged. FP-only negative schedules use
                # a distinct namespace and cannot overwrite captured positives.
                directory=schedule.get('record_directory','records')
                prefix=schedule.get('pose_prefix','p')
                if directory not in ('records','negative_records') or prefix not in ('p','n'):
                    raise ValueError('Invalid record namespace')
                if directory=='negative_records' and role!='FP': raise ValueError('Negative renders must be FP-only')
                path=root/'RIR'/role/split/scene/directory/f'{prefix}{p["pose_id"]:05d}'/f'base_acoustic_yaw_{base:03d}.npy'
                meta=path.with_suffix('.json')
                if path.exists() and meta.exists():
                    saved=json.loads(meta.read_text())
                    if saved['schedule_sha256']!=fingerprint or saved['rir_sha256']!=sha(path): raise ValueError(f'Stale/corrupt RIR: {path}')
                    if role=='FP' and not (saved.get('loaded_frame') or {}).get('verified'):
                        raise ValueError(f'Legacy FP frame not verified; preserve this data and use a versioned corrected output: {path}')
                    skipped+=1;continue
                if sim is None:
                    registry,expected_objects=furnished_registry(root,source,scene) if role=='Observed' else (None,0)
                    if role=='FP':sim,loaded_frame=make_extruded_simulator(geometry)
                    else:sim=make_geometry_simulator(geometry,registry)
                    if role=='Observed' and sim.get_rigid_object_manager().get_num_objects()!=expected_objects:
                        raise ValueError('Furnished scene lost objects during registry loading')
                    sensor=_add_audio_sensor(sim,settings)
                if role=='FP':
                    centre=np.asarray(p['centre_xyz_m'])
                    offsets=np.column_stack([.05*np.cos(RING+np.deg2rad(base)),np.zeros(6),-.05*np.sin(RING+np.deg2rad(base))])
                    points=np.vstack([centre,centre+offsets])
                    bounds=np.asarray(loaded_frame['loaded_bounds'])
                    if not np.all(points>bounds[0]) or not np.all(points<bounds[1]):
                        raise ValueError('Source/microphone outside post-load extrusion bounds')
                rir=_render_ring(sim,sensor,np.asarray(p['centre_xyz_m'],np.float32),base,.05)
                if rir.ndim!=2 or rir.shape[0]!=6 or not np.isfinite(rir).all() or not np.all(np.any(rir!=0,axis=1)): raise ValueError('Invalid six-channel RIR')
                _atomic_npy(path,rir)
                atomic_json(meta,dict(version=schedule.get('version',VERSION),role=role,geometry_kind='canonical_map_extrusion' if role=='FP' else 'furnished_native_scene',
                    scene_id=scene,split=split,pose_id=p['pose_id'],centre_xyz_m=p['centre_xyz_m'],source_xyz_m=p['centre_xyz_m'],
                    base_acoustic_yaw_deg=base,microphone_angles_rad=(RING+np.deg2rad(base)).tolist(),
                    rir_shape=list(rir.shape),rir_sha256=sha(path),schedule_sha256=fingerprint,geometry_sha256=expected,
                    acoustics=PROTOCOL,cpu_only=True,gpu_device_id=-1,acoustic_threads=threads,loaded_frame=loaded_frame))
                written+=1
                print(f'[RIR] {role}/{split}/{scene} p{p["pose_id"]:05d} b{base:03d}',flush=True)
    finally:
        if sim is not None: sim.close()
    return dict(scene_id=scene,split=split,role=role,pose_ids=pose_ids,written=written,skipped=skipped,elapsed_seconds=time.monotonic()-start)


def _legacy_render(args):
    import fcntl
    for name in ('CUDA_VISIBLE_DEVICES','NVIDIA_VISIBLE_DEVICES'):
        if os.environ.get(name) not in ('-1','void','none',''): raise ValueError(f'GPU visibility not disabled: {name}')
    contract_root=args.root/'RIR/manifests'/VERSION
    lock=(contract_root/'render.lock').open('a')
    try: fcntl.flock(lock,fcntl.LOCK_EX if args.wait_lock else fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: raise RuntimeError('Another positive renderer owns this dataset; refusing overlapping writes')
    index=json.loads((contract_root/'index.json').read_text())
    entries=[e for e in index['scenes'] if e['split'] in args.splits and (not args.scene or e['scene_id']==args.scene)
             and (not args.only_scenes or e['scene_id'] in args.only_scenes)]
    tasks=[]
    for e in entries:
        schedule=json.loads((args.root/e['schedule_path']).read_text())
        ids=[p['pose_id'] for p in schedule['positions']]
        if args.limit: ids=ids[:args.limit]
        for role in ('FP','Observed'):
            for offset in range(0,len(ids),args.chunk):
                tasks.append((str(args.root),str(args.source),str(args.renderer_dir),e,role,ids[offset:offset+args.chunk],args.acoustic_threads))
    run_root=args.root/'RIR/manifests'/VERSION/'runs'/args.run_name
    run_root.mkdir(parents=True,exist_ok=True)
    results=[];errors=[];start=time.monotonic()
    atomic_json(run_root/'status.json',dict(status='running',tasks=len(tasks),workers=args.workers,acoustic_threads=args.acoustic_threads,cpu_only=True))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(render_task,t):t for t in tasks}
        for f in as_completed(futures):
            t=futures[f]
            try: result=f.result();results.append(result);atomic_json(run_root/f'task_{tasks.index(t):05d}.json',result)
            except Exception as exc:
                failure=dict(scene_id=t[3]['scene_id'],role=t[4],pose_ids=t[5],error=str(exc),traceback=traceback.format_exc())
                errors.append(failure);atomic_json(run_root/f'error_{tasks.index(t):05d}.json',failure)
                print('[ERROR] '+json.dumps(failure),flush=True)
            atomic_json(run_root/'status.json',dict(status='running',tasks=len(tasks),completed_tasks=len(results),failed_tasks=len(errors),
                records_written=sum(r['written'] for r in results),records_verified_existing=sum(r['skipped'] for r in results),
                elapsed_seconds=time.monotonic()-start,workers=args.workers,acoustic_threads=args.acoustic_threads,cpu_only=True))
    status=dict(status='complete' if not errors else 'failed',tasks=len(tasks),completed_tasks=len(results),failed_tasks=len(errors),
                records_written=sum(r['written'] for r in results),records_verified_existing=sum(r['skipped'] for r in results),
                elapsed_seconds=time.monotonic()-start,limited_smoke=bool(args.limit),splits=args.splits,cpu_only=True)
    atomic_json(run_root/'status.json',status)
    if errors: raise RuntimeError(f'{len(errors)} render tasks failed; inspect {run_root}')


def render(args):
    from .hssd_rir_supervisor import positive_render
    return positive_render(args)


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','render'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--splits',nargs='+',choices=['train','val','test'],default=['train'])
    p.add_argument('--workers',type=int,default=8);p.add_argument('--renderer-dir',type=Path)
    p.add_argument('--acoustic-threads',type=int,default=2);p.add_argument('--chunk',type=int,default=12)
    p.add_argument('--scene');p.add_argument('--limit',type=int);p.add_argument('--run-name',default='node2_train')
    p.add_argument('--only-scenes',nargs='+');p.add_argument('--wait-lock',action='store_true')
    p.add_argument('--height-policy',choices=['uniform','area_weighted_mean'],default='uniform')
    a=p.parse_args()
    if a.workers<1 or a.acoustic_threads<1 or a.chunk<1: p.error('Counts must be positive')
    if a.phase=='render' and a.renderer_dir is None: p.error('--renderer-dir required')
    (prepare if a.phase=='prepare' else render)(a)


if __name__=='__main__': main()

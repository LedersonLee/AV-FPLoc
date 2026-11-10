"""Fresh native furnished/stage-only6-mic pairs at fixed development poses.

No canonical extrusion, resampling, navmesh snapping, or Test observations.
Only the new output directory is writable. Renderer helpers are hashed inputs.
"""
import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np
from .prepare_targets import read,sha


def preserve_openings(original,registry_path,directory):
    """Keep explicitly opening-class templates; remove other object instances.

    This is an auditable folder-based control, not a claim that every remaining
    removed object is movable furniture (fixed fixtures may also be removed).
    """
    registry=read(registry_path)
    opening_handles={Path(p).name.replace('.object_config.json','') for p in registry['objects']['paths']['.json'] if '/openings/' in p}
    altered=dict(original);altered['object_instances']=[o for o in original['object_instances'] if Path(o['template_name']).name in opening_handles]
    directory.mkdir(exist_ok=False,parents=True)
    instance=directory/'opening_preserved.scene_instance.json';instance.write_text(json.dumps(altered,indent=2))
    for section in ['stages','objects']:
        for suffix,paths in registry[section]['paths'].items():
            registry[section]['paths'][suffix]=[str((registry_path.parent/p).resolve()) for p in paths]
    registry['scene_instances']['paths']={'.json':[str(instance)]}
    registry['semantic_scene_descriptor_instances']={k:str((registry_path.parent/v).resolve()) for k,v in registry.get('semantic_scene_descriptor_instances',{}).items()}
    reg=directory/'dataset.scene_dataset_config.json';reg.write_text(json.dumps(registry,indent=2))
    return instance,reg,len(altered['object_instances'])


def main():
    p=argparse.ArgumentParser()
    for k in ['root','targets','out']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--code-sha',required=True);p.add_argument('--scene-limit',type=int)
    p.add_argument('--queries',type=int,default=8);p.add_argument('--threads',type=int,default=2)
    p.add_argument('--part',choices=['val','official_test'],default='val')
    p.add_argument('--opening-preserved-only',action='store_true')
    a=p.parse_args();a.out.mkdir(exist_ok=False,parents=True)
    source=a.root/'source/hssd-hab';helper=a.root/'RIR/tools/positive_pairs_v1'
    sys.path.insert(0,str(helper))
    from hssd_positive_pairs import make_geometry_simulator,furnished_registry
    from render_hssd_paired_rirs import _add_audio_sensor
    from render_hssd_exp4_ring_rirs import _render_ring
    from hssd_cpu_rir_contract import require_cpu_device
    require_cpu_device(-1)
    protocol=read(a.targets/'protocol.json');families=set();scenes=[]
    if a.part=='official_test':assert protocol.get('method_freeze_sha256') and a.scene_limit is None
    for s in protocol['scenes'][a.part]:
        family=s.split('_')[0]
        if family not in families:scenes.append(s);families.add(family)
    if a.scene_limit:scenes=scenes[:a.scene_limit]
    settings=SimpleNamespace(gpu_device_id=-1,sample_rate=48000,indirect_rays=20000,ray_depth=50,
        source_rays=512,diffraction=True,max_diffraction_order=10,acoustic_threads=a.threads)
    config=dict(code_sha=a.code_sha,scenes=scenes,part=a.part,queries_per_scene=a.queries,selection='one variant per source family in frozen order; uniform query indices before pair results',
        targets_protocol_sha256=sha(a.targets/'protocol.json'),method_freeze_sha256=protocol.get('method_freeze_sha256'),
        geometry='same native stage, with all original object instances or with zero object instances',acoustics=vars(settings),
        source_receiver='exact captured centre,6 omni microphones at radius0.05m;48kHz; original stored base heading',
        default_materials=True,transmission=False,cpu_only=True,
        repeat='fresh simulator furnished repeat of first selected query in each scene; detects numerical/render variability',
        helper_hashes={f:sha(helper/f) for f in ['hssd_positive_pairs.py','render_hssd_paired_rirs.py','render_hssd_exp4_ring_rirs.py','render_hssd_ring_symmetry_smoke.py','hssd_cpu_rir_contract.py']})
    if a.opening_preserved_only:config.update(geometry='native stage plus all original objects whose template config is in objects/openings; other object instances removed, including possible fixed fixtures',repeat='none; paired against existing fresh furnished and stage-only runs')
    (a.out/'protocol.json').write_text(json.dumps(config,indent=2));records=[]
    for scene in scenes:
        refs=read(a.targets/f'{scene}_queries.json')
        indices=np.linspace(0,len(refs)-1,min(a.queries,len(refs))).round().astype(int)
        scene_path=source/f'scenes/{scene}.scene_instance.json';stage_path=source/f'stages/{scene}.glb'
        original=read(scene_path);assert original['stage_instance']=={'template_name':'stages/'+scene}
        registry=a.root/f'RIR/Geometry/{scene}/furnished.scene_dataset_config.json'
        if not registry.exists():registry,_=furnished_registry(a.out,source,scene)
        variants=['opening_preserved'] if a.opening_preserved_only else ['furnished','stage_only','furnished_repeat']
        for variant in variants:
            geometry=stage_path if variant=='stage_only' else scene_path
            actual_registry=None if variant=='stage_only' else registry
            expected=0 if variant=='stage_only' else len(original['object_instances'])
            if variant=='opening_preserved':geometry,actual_registry,expected=preserve_openings(original,registry,a.out/scene/'configuration')
            start=time.perf_counter();sim=None
            try:
                sim=make_geometry_simulator(geometry,actual_registry)
                objects=sim.get_rigid_object_manager().get_num_objects()
                assert objects==expected,(scene,variant,objects,expected)
                sim.seed(42)  # Habitat RNG; not asserted to control SoundSpaces internal RNG.
                sensor=_add_audio_sensor(sim,settings)
                for i in indices[:1] if variant=='furnished_repeat' else indices:
                    q=refs[int(i)];centre=np.asarray(q['centre_xyz_m'],np.float32)
                    rir=_render_ring(sim,sensor,centre,q['base_acoustic_yaw_deg'],.05)
                    assert rir.ndim==2 and rir.shape[0]==6 and np.isfinite(rir).all() and np.all(np.any(rir!=0,axis=1))
                    path=a.out/scene/variant/f'q{q["query_id"]}.npy';path.parent.mkdir(parents=True,exist_ok=True)
                    assert not path.exists();np.save(path,rir)
                    record=dict(scene=scene,variant=variant,query_id=q['query_id'],query_index=int(i),
                        centre_xyz_m=q['centre_xyz_m'],source_xyz_m=q['centre_xyz_m'],world_yaw_deg=q['world_yaw_deg'],
                        base_acoustic_yaw_deg=q['base_acoustic_yaw_deg'],original_rir_path=q['rir_path'],
                        file=str(path.relative_to(a.out)),sha256=sha(path),shape=list(rir.shape),loaded_objects=objects,
                        stage_sha256=sha(stage_path),scene_config_sha256=sha(scene_path),render_geometry_sha256=sha(geometry),registry_sha256=sha(actual_registry) if actual_registry else None,
                        elapsed_since_sim_start_s=time.perf_counter()-start)
                    records.append(record);(a.out/'manifest.json').write_text(json.dumps(records,indent=2))
                    print(json.dumps(record),flush=True)
            finally:
                if sim is not None:sim.close()
    (a.out/'COMPLETE').write_text(a.code_sha+'\n')


if __name__=='__main__':main()

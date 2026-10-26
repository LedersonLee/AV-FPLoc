"""Freeze pilot scene membership and GT-independent metric candidate banks."""
import argparse
import copy
from pathlib import Path

import numpy as np
from PIL import Image

from acoustic_branch.contracts import immutable_json, sha
from acoustic_branch.data import read
from acoustic_branch.test_rirs import grid_cells

REL='ReliabilityFusion/pilots/v1'
ACOUSTIC='AcousticContrastive/pilots/scene_subset_v1/corrected_yup_v2'


def prepare(root,vision_root=None):
    root=Path(root)
    vision_root=Path(vision_root) if vision_root else root
    audit=read(root/'ReliabilityFusion/preflight/desdf_range_v1.json')
    if audit['status']!='passed':raise ValueError('Range preflight not passed')
    membership=read(root/'manifests/scene_splits_v1.json')
    split_hash=sha(root/'manifests/scene_splits_v1.json')
    assert split_hash==audit['split_sha256']
    old=read(root/'AcousticContrastive/pilots/scene_subset_v1/scene_splits.json')['scenes']
    def cells(scene):
        folder=root/'2DFP'/scene
        return grid_cells(np.asarray(Image.open(folder/'map.png').convert('L'))==255,
                          read(folder/'floorplan_metadata.json'))
    ordered=sorted(old['train'],key=lambda s:(len(cells(s)[1]),s))
    # Geometric size coverage, not test performance, labels, or model confidence.
    selected={'train':[ordered[i] for i in np.rint(np.linspace(0,len(ordered)-1,6)).astype(int)],
              'val':sorted(old['val']),'test':sorted(old['test'])}
    assert len({s for scenes in selected.values() for s in scenes})==sum(map(len,selected.values()))
    for split,scenes in selected.items():assert set(scenes)<=set(membership[split])
    config=read(root/ACOUSTIC/'config.json')
    entries=[]
    for split in ('train','val'):
        for scene in selected[split]:
            source=root/f'RIR/manifests/positive_pairs_v1/schedules/{split}/{scene}.json'
            positive=read(source)
            assert positive['split']==split and positive['scene_id']==scene and positive['split_sha256']==split_hash
            geometry=root/positive['geometry_path']
            assert sha(geometry)==positive['geometry_sha256']
            immutable_json(geometry.with_name('floorplan.y_up.stage_config.json'),
                dict(render_asset='floorplan.glb',up=[0,1,0],front=[0,0,-1],units_to_meters=1.))
            rc,xy=cells(scene)
            from acoustic_branch.pilot_evaluate import bank_windows
            payload=np.load(root/'UnLoc/structural_targets'/scene/'desdf.npy',allow_pickle=True).item()
            assert payload['max_range_m']==10 and payload['desdf_spacing_m']==.1
            bank_windows(payload,xy)  # Fail before simulation if source grids/masks disagree.
            schedule=copy.deepcopy(positive)
            height=positive['height_contract']['floor_y_m']+1.5
            schedule.update(version='reliability_fusion_bank_v1',queries=[],query_count=0,excluded=[],
                grid_spacing_m=.2,grid_rc=rc.tolist(),record_directory='records',pose_prefix='p',
                positions=[dict(pose_id=2000000+i,centre_xyz_m=[float(x),height,float(z)],base_headings=list(range(0,60,10)))
                    for i,(x,z) in enumerate(xy)],position_count=len(xy),physical_records_per_role=len(xy)*6,
                namespace='p2000000+ fixed grid; not query GT; no GT injection')
            path=root/REL/'candidate_schedules'/split/f'{scene}.json'
            immutable_json(path,schedule)
            entries.append(dict(scene_id=scene,split=split,cells=len(xy),candidate_xz=xy.tolist(),
                queries=len(positive['queries']),fp_arrays=len(xy)*6,
                observed_arrays=positive['physical_records_per_role'],schedule_path=str(path.relative_to(root)),
                schedule_sha256=sha(path),observation_schedule_path=str(source.relative_to(root)),
                observation_schedule_sha256=sha(source)))
    bank=dict(version='reliability_fusion_bank_v1',split_sha256=split_hash,scenes=entries,
        grid_spacing_m=.2,yaw_step_deg=10,fp_arrays=sum(e['fp_arrays'] for e in entries),
        observed_arrays=sum(e['observed_arrays'] for e in entries),selection=selected,
        selection_rule='Six acoustic-train scenes at evenly spaced size-order quantiles; original3val/test; no performance selection',
        test_accessed_for_training=False)
    immutable_json(root/REL/'bank_index.json',bank)
    immutable_json(root/REL/'config.json',dict(version='reliability_fusion_v1',seed=42,features=config['features'],
        acoustic_model=config['model'],acoustic_checkpoint=ACOUSTIC+'/runs/acoustic_fresh_v1/best.pt',
        acoustic_sha256=sha(root/ACOUSTIC/'runs/acoustic_fresh_v1/best.pt'),
        visual_checkpoint='UnLoc/evaluation_checkpoints/epoch020/model.ckpt',
        visual_sha256=sha(vision_root/'UnLoc/evaluation_checkpoints/epoch020/model.ckpt'),
        split_sha256=split_hash,bank_sha256=sha(root/REL/'bank_index.json'),
        mode='one_query_weight',hidden=[32,16],epochs=100,patience=15,batch_size=32,lr=.001,weight_decay=.001,
        soft_target_xy_sigma_m=.2,soft_target_yaw_sigma_deg=10.,score_temperature=1.,top_k=50,
        primary_selection='val scene-macro R@1m; ties joint recall then NLL',
        score_source='frozen in-sample train branches; not OOF',
        test_notice='Existing3 test scenes exploratory; no test-based fitting/selection; global split unchanged'))
    # Freeze existing test INPUT bytes, without fitting or inspecting outcomes.
    # Test score construction remains gated behind selected training checkpoints.
    config=read(root/REL/'config.json')
    frozen=Path('/ssd1/AV-FPLoc/exp4_acoustic_xy_observability/analysis/hssd_epoch020_evaluation_20261018')
    paths={}
    eval_relative=ACOUSTIC+'/evaluation'
    for name in ('bank_index.json','ready.json','raw_verification.json'):
        rel=eval_relative+'/'+name;paths['root:'+rel]=sha(root/rel)
    for scene in selected['test']:
        for role in ('candidate','observed'):
            rel=eval_relative+'/'+scene+'_'+role+'_embeddings_'+config['acoustic_sha256'][:12]+'.npy'
            paths['root:'+rel]=sha(root/rel)
        for rel in ('depth/'+scene+'.npz','rows/'+scene+'.json'):
            paths['vision_eval:'+rel]=sha(frozen/rel)
    rel='AcousticContrastive/pilots/scene_subset_v1/evaluation/report/corrected-test/records.json'
    paths['vision:'+rel]=sha(vision_root/rel)
    immutable_json(root/REL/'test_input_contract.json',dict(files=paths,test_fitting=False,
        scope='Input hashes only; no test model/weight/threshold selection'))
    return bank


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--vision-root',type=Path)
    args=parser.parse_args();bank=prepare(args.root,args.vision_root)
    print({k:v for k,v in bank.items() if k!='scenes'})

"""Read-only audit of frozen HSSD UnLoc range contracts before fusion design.

Checks actual arrays, active targets, replica hashes and cached test targets.
GT is used only for integrity checks; no model or fusion parameter is fitted.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json

import numpy as np

from .contracts import atomic_json, sha


def read(path):
    return json.loads(Path(path).read_text())


def audit(root, replica, vision_eval, output):
    membership=read(root/'manifests/scene_splits_v1.json')
    owners={scene:split for split,scenes in membership.items() for scene in scenes}
    if len(owners)!=sum(map(len,membership.values())):
        raise ValueError('Overlapping scene splits')
    adapter=root/'UnLoc/frame_adapter_v4_furnished_qa'
    summaries=read(adapter/'summary.json')
    contract=read(root/'UnLoc/runs/hssd_unloc_v4_full_20261017/training_contract.json')
    assert summaries['structural_target_max_range_m']==10
    assert contract['adapter_root'].endswith('/UnLoc/frame_adapter_v4_furnished_qa')
    assert contract['floorplan_root'].endswith('/UnLoc/structural_targets')
    # Identical 40 pinhole feature centers to export_hssd_unloc_frames.py.
    bearings=np.arctan(((np.arange(40)+.5)*16-.5-319.5)/240.)
    cosine=np.cos(bearings)
    entries=[]
    for split in ('train','val','test'):
        manifest=read(adapter/'manifests'/f'{split}_scenes.json')
        assert {e['scene_id'] for e in manifest['scenes']}==set(membership[split])
        entries.extend((split,e) for e in manifest['scenes'])
    def work(item):
        split,entry=item
        scene=entry['scene_id']
        desdf_path=root/'UnLoc/structural_targets'/scene/'desdf.npy'
        payload=np.load(desdf_path,allow_pickle=True).item()
        field=payload['desdf']
        assert payload['max_range_m']==10 and payload['desdf_spacing_m']==.1
        assert payload['map_spacing_m']==.01 and field.shape[2]==36
        assert np.isfinite(field).all() and field.min()>=0 and field.max()<=10+1e-5
        target_path=root/entry['depth40_path']
        rows_path=root/entry['sample_metadata_path']
        target=np.load(target_path)
        rows=[json.loads(line) for line in rows_path.read_text().splitlines() if line]
        assert target.shape==(len(rows),40) and len(rows)==entry['accepted_frames']
        assert np.isfinite(target).all() and target.min()>0
        target=target[[r['target_index'] for r in rows]]
        pose=np.array([r['pose_xzyaw'] for r in rows])
        col=np.rint((pose[:,0]-payload['origin_x_m'])/.01).astype(int)//10
        row=np.rint((pose[:,1]-payload['origin_z_min_m'])/.01).astype(int)//10
        bins=np.rint((pose[:,2,None]+np.rad2deg(bearings)[None])*36/360).astype(int)%36
        expected=field[row[:,None],col[:,None],bins]*cosine[None]
        difference=np.abs(target-expected)
        hashes={}
        for path in (desdf_path,target_path):
            relative=str(path.relative_to(root))
            hashes[relative]=sha(path)
            assert hashes[relative]==sha(replica/relative),f'Replica differs: {relative}'
        cached=None
        if split=='test':
            cached_path=vision_eval/'depth'/f'{scene}.npz'
            cached_rows=read(vision_eval/'rows'/f'{scene}.json')
            with np.load(cached_path) as cached_data:
                assert len(cached_rows)==len(rows)
                source={r['sample_index']:r for r in rows}
                for cached_row in cached_rows:
                    active=source[cached_row['sample_index']]
                    assert cached_row['target_index']==active['target_index']
                    assert np.allclose(cached_row['pose_xzyaw'],active['pose_xzyaw'],atol=1e-7,rtol=0)
                    assert cached_row['depth40_path'].endswith(entry['depth40_path'])
                predictions=cached_data['location']
                cached=dict(queries=len(predictions),prediction_min=float(predictions.min()),
                    prediction_max=float(predictions.max()),active_row_target_references_match=True,
                    cached_target_arrays_stored='target' in cached_data.files)
        return dict(scene_id=scene,split=split,queries=len(rows),metadata_radial_cap_m=payload['max_range_m'],
            desdf_spacing_m=payload['desdf_spacing_m'],actual_desdf_max_m=float(field.max()),
            max_reconstructed_depth_error_m=float(difference.max()),
            mismatched_rays_gt_2cm=int((difference>.02).sum()),
            target_radialized_max_m=float((target/cosine[None]).max()),
            replica_hashes_match=True,source_hashes=hashes,cached_test=cached)
    with ThreadPoolExecutor(max_workers=4) as pool:
        records=list(pool.map(work,entries))
    source_root=Path(__file__).resolve().parents[1]
    scripts=source_root/'experiments/exp4_acoustic_xy_observability/scripts'
    sources=[scripts/n for n in ('hssd_furnished_2dfp_rays.py','export_hssd_unloc_frames.py',
        'export_hssd_unloc_furnished_adapter.py','train_hssd_unloc.py','evaluate_hssd_unloc_test.py')]
    sources += [source_root/'acoustic_branch/pilot_evaluate.py',source_root/'third_party/UnLoc/modules/depth_net.py']
    result=dict(status='passed' if sum(r['mismatched_rays_gt_2cm'] for r in records)==0 else 'failed',
        stage='design_integrity_audit_only',training_performed=False,test_fitting=False,
        cap_semantics='10m Euclidean ray range before conversion to camera-forward depth; encoder prediction has no hard10m clamp.',
        split_counts={k:len(v) for k,v in membership.items()},queries=sum(r['queries'] for r in records),
        checked_desdf_copies=2*len(records),source_hashes={str(p):sha(p) for p in sources},
        split_sha256=sha(root/'manifests/scene_splits_v1.json'),
        mismatch_rays=sum(r['mismatched_rays_gt_2cm'] for r in records),
        max_reconstructed_depth_error_m=max(r['max_reconstructed_depth_error_m'] for r in records),
        records=records)
    atomic_json(output,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('records','source_hashes')},indent=2),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('root','replica','vision-eval','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    audit(args.root,args.replica,args.vision_eval,args.output)

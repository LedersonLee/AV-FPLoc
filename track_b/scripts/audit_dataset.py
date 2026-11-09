"""Read-only development audit. Never load test observations or reference RIRs."""
import argparse, csv, hashlib, json, time
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from track_b.data.geometry import candidate_bank, canonical_rir, camera_forward, mic_offsets


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text())


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args(); root=args.root; out=args.out; out.mkdir(parents=True,exist_ok=False)
    split=read(root/'manifests/scene_splits_v1.json'); assert sum(map(len,split.values()))==len(set(sum(split.values(),[])))
    start=time.time(); rows=[]; qc=[]; sampled=[]
    # Test IDs and asset identity metadata only; no test poses, maps or waveforms.
    groups={s:s.split('_')[0] for v in split.values() for s in v}
    test_groups={groups[s] for s in split['test']}
    for part in ('train','val'):
      for scene in split[part]:
        sp=root/f'RIR/manifests/positive_pairs_v1/schedules/{part}/{scene}.json'
        if not sp.exists(): qc.append(dict(scene_id=scene,reason='missing_schedule')); continue
        sch=read(sp); meta=read(root/f'2DFP/{scene}/floorplan_metadata.json')
        free=np.array(Image.open(root/f'2DFP/{scene}/map.png'))==255
        origin=[meta['world_bounds_m'][k] for k in ('x_min','z_min')]; spacing=meta['map_spacing_m']
        bank,valid=candidate_bank(free,spacing,origin)
        xy=bank[::36,:2]; tree=cKDTree(xy)
        pilot=read(root/f'UnLoc/RGB/{scene}/pilot_manifest.json')
        captures={r['sample_index']:r for r in pilot['records'] if r['accepted']}
        repl=root/f'UnLoc/furnished_depth_qa_v2_nohit_only/replacements/{scene}/replacement_manifest.json'
        if repl.exists(): captures.update({r['sample_index']:r for r in read(repl)['records'] if r.get('accepted')})
        stat=dict(scene_id=scene,split=part,queries=0,available=0,bank_candidates=len(bank),free_area_m2=float(free.sum()*spacing**2),
                  out_of_footprint=0,missing_rgb=0,missing_rir=0,alignment_failures=0,excluded_test_family=groups[scene] in test_groups,
                  map_sha256=sha(root/f'2DFP/{scene}/map.png'),schedule_sha256=sha(sp),quantization_distances=[])
        for q in sch['queries']:
            rgb=root/q['image_path']; rir=root/q['observed_rir_path']; xyz=np.array(q['centre_xyz_m']); yaw=q['world_yaw_deg']
            assert q['observed_rir_path'].startswith('RIR/Observed/')
            cap=captures.get(q['sample_index']); aligned=cap is not None
            if cap is not None:
                aligned=bool(np.allclose(cap['pose_xyz_m'],xyz,atol=1e-6) and np.isclose(cap['logical_yaw_deg'],yaw))
                aligned &= bool(np.allclose(camera_forward(cap['habitat_yaw_deg']),[np.cos(np.deg2rad(yaw)),np.sin(np.deg2rad(yaw))],atol=1e-6))
            pix=np.rint((xyz[[0,2]]-origin)/spacing).astype(int)
            inside=bool(0<=pix[0]<free.shape[1] and 0<=pix[1]<free.shape[0] and free[pix[1],pix[0]])
            dist=float(tree.query(xyz[[0,2]])[0]); dyaw=abs((yaw+5)%10-5)
            exists_rgb=rgb.is_file(); exists_rir=rir.is_file()
            flags=[]
            if not exists_rgb: flags.append('missing_rgb')
            if not exists_rir: flags.append('missing_rir')
            if not aligned: flags.append('alignment_failure')
            if not inside: flags.append('outside_footprint')
            if stat['excluded_test_family']: flags.append('test_source_family')
            stat['queries']+=1; stat['available']+=int(exists_rgb and exists_rir);stat['missing_rgb']+=not exists_rgb;stat['missing_rir']+=not exists_rir
            stat['alignment_failures']+=not aligned;stat['out_of_footprint']+=not inside;stat['quantization_distances'].append(dist)
            rows.append(dict(scene_id=scene,building_group_id=groups[scene],split=part,query_id=q['sample_index'],rgb_path=q['image_path'],rir_path=q['observed_rir_path'],
                fp_path=f'2DFP/{scene}/map.png',world_to_fp=json.dumps(dict(origin_xz=origin,spacing=spacing)),pose_world=json.dumps([*xyz.tolist(),yaw]),
                sensor_calib=json.dumps(dict(camera=pilot['camera_contract'],acoustics=sch['acoustics'],base=q['base_acoustic_yaw_deg'],roll=q['channel_roll'])),
                sim_version='historical renderer; exact binary not recorded in schedule',asset_hash=sch['scene_sha256'],validity_flags=';'.join(flags),furniture_state='furnished',
                nearest_grid_m=dist,nearest_yaw_deg=dyaw))
        ds=stat.pop('quantization_distances');stat['quantization_p50_p95_max_m']=np.quantile(ds,[.5,.95,1]).tolist()
        stat['grid_joint_upper_bound']=float(np.mean(np.array(ds)<=1));stat['grid_fine_upper_bound']=float(np.mean(np.array(ds)<=.1))
        # One observed waveform per scene; this is a sample QC, not full waveform integrity.
        q=next((q for q in sch['queries'] if (root/q['observed_rir_path']).exists()),None)
        if q and not stat['excluded_test_family']:
            path=root/q['observed_rir_path']; r=np.load(path); side=read(path.with_suffix('.json'))
            canonical_rir(r,q['world_yaw_deg'],q['base_acoustic_yaw_deg'])
            sampled.append(dict(scene_id=scene,shape=list(r.shape),finite=bool(np.isfinite(r).all()),sha_match=sha(path)==side['rir_sha256'],
                centre_match=bool(np.allclose(side['centre_xyz_m'],q['centre_xyz_m'])),direct_peak_first32=np.argmax(abs(r[:,:32]),axis=1).tolist(),bytes=path.stat().st_size))
        qc.append(stat);print(json.dumps(stat),flush=True)
    with (out/'manifest.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    (out/'dataset_qc.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in qc))
    (out/'waveform_sample_qc.json').write_text(json.dumps(sampled,indent=2))
    eligible={p:sorted({r['scene_id'] for r in rows if r['split']==p and not r['validity_flags']}) for p in ('train','val')}
    # Preserve official train/val, remove cross-split source families conservatively.
    vg={groups[s] for s in eligible['val']};eligible['train']=[s for s in eligible['train'] if groups[s] not in vg]
    frozen=dict(official=split,development=eligible,building_groups=groups,test_observations_accessed=False,
                source_family_rule='scene ID prefix before underscore; conservative proxy, near-duplicate geometry audit pending',split_source_sha256=sha(root/'manifests/scene_splits_v1.json'))
    (out/'split_manifest.json').write_text(json.dumps(frozen,indent=2))
    summary=dict(scenes=len(qc),queries=len(rows),available=sum(r.get('available',0) for r in qc),sampled_waveforms=len(sampled),
        missing_rgb=sum(r.get('missing_rgb',0) for r in qc),missing_rir=sum(r.get('missing_rir',0) for r in qc),alignment_failures=sum(r.get('alignment_failures',0) for r in qc),
        out_of_footprint=sum(r.get('out_of_footprint',0) for r in qc),development_scenes={k:len(v) for k,v in eligible.items()},elapsed_s=time.time()-start,
        waveform_qc_pass=all(r['finite'] and r['sha_match'] and r['centre_match'] for r in sampled),
        limitations=['Black mask combines wall/exterior/unknown; no physical-wall truth','Source-family prefix is conservative, geometry near-duplicates unresolved','One waveform per scene inspected','No pristine independent test evaluation'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)

if __name__=='__main__': main()

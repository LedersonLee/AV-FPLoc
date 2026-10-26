"""Frozen 12/3/3 fusion scope; missing dense TRAIN banks, never TEST work."""
import copy
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import io
import os
from pathlib import Path
import time
import traceback

import numpy as np
from numpy.lib.format import open_memmap
from PIL import Image

from acoustic_branch.contracts import atomic_json,immutable_json,sha,PROTOCOL
from acoustic_branch.data import read
from acoustic_branch.features import extract as ec_extract
from acoustic_branch.envelope_cache import extract as env_extract,DEFINITION
from acoustic_branch.test_rirs import grid_cells
from acoustic_branch.render_adapter import supervisor_tools
from .revision_data import REL as OLD,MODEL

REL='ReliabilityFusion/pilots/subset12_logit_v1'


def prepare(root):
    root=Path(root);out=root/REL;old=read(root/OLD/'config.json')
    subset_path=root/'AcousticContrastive/pilots/scene_subset_v1/scene_splits.json';subset=read(subset_path)
    membership=read(root/'manifests/scene_splits_v1.json');selected=subset['scenes']
    assert {k:len(v) for k,v in selected.items()}==dict(train=12,val=3,test=3)
    assert subset['parent_split_sha256']==sha(root/'manifests/scene_splits_v1.json')==old['split_sha256']
    assert len(set(s for v in selected.values() for s in v))==18
    for split,ids in selected.items():assert set(ids)<=set(membership[split])
    assert all(set(old['selection'][s])<=set(selected[s]) for s in selected)
    acoustic=read(root/MODEL/'configs/all.json')
    assert acoustic['scene_subset_sha256']==sha(subset_path) and acoustic['subset_counts']==dict(train=12,val=3,test=3)
    original=read(root/'ReliabilityFusion/pilots/v1/bank_index.json');old_entries={e['scene_id']:e for e in original['scenes']}
    entries=[];new=[]
    for split in ('train','val'):
        for sid in sorted(selected[split]):
            if sid in old_entries:
                e=dict(old_entries[sid],reuse_frozen_scores=True);assert e['split']==split
            else:
                assert split=='train'
                source=root/f'RIR/manifests/positive_pairs_v1/schedules/{split}/{sid}.json';positive=read(source)
                assert positive['scene_id']==sid and positive['split']==split and positive['split_sha256']==old['split_sha256']
                geometry=root/positive['geometry_path'];assert sha(geometry)==positive['geometry_sha256']
                immutable_json(geometry.with_name('floorplan.y_up.stage_config.json'),dict(render_asset='floorplan.glb',up=[0,1,0],front=[0,0,-1],units_to_meters=1.))
                geom=root/'2DFP'/sid;meta=read(geom/'floorplan_metadata.json');rc,xy=grid_cells(np.asarray(Image.open(geom/'map.png').convert('L'))==255,meta)
                schedule=copy.deepcopy(positive);height=positive['height_contract']['floor_y_m']+1.5
                schedule.update(version='subset12_dense_train_v1',queries=[],query_count=0,excluded=[],grid_spacing_m=.2,grid_rc=rc.tolist(),
                    positions=[dict(pose_id=2000000+i,centre_xyz_m=[float(x),height,float(z)],base_headings=list(range(0,60,10))) for i,(x,z) in enumerate(xy)],
                    record_directory='records',pose_prefix='p',position_count=len(xy),physical_records_per_role=len(xy)*6,
                    namespace='TRAIN dense0.2m fixed map-origin grid, p2000000+; independent of GT; not captured-query poses')
                path=out/'candidate_schedules/train'/f'{sid}.json';immutable_json(path,schedule)
                e=dict(scene_id=sid,split=split,cells=len(xy),candidate_xz=xy.tolist(),queries=len(positive['queries']),fp_arrays=len(xy)*6,
                    observed_arrays=positive['physical_records_per_role'],schedule_path=str(path.relative_to(root)),schedule_sha256=sha(path),
                    observation_schedule_path=str(source.relative_to(root)),observation_schedule_sha256=sha(source),reuse_frozen_scores=False)
                new.append(sid)
            entries.append(e)
    assigned=[[],[]];loads=[0,0]
    for e in sorted([e for e in entries if not e['reuse_frozen_scores']],key=lambda e:-e['cells']):
        worker=int(np.argmin(loads));assigned[worker].append(e['scene_id']);loads[worker]+=e['cells']
    index=dict(selection=selected,scenes=entries,missing_train_scenes=new,feature_assignment=assigned,
        new_fp_arrays=sum(e['fp_arrays'] for e in entries if not e['reuse_frozen_scores']),test_arrays_accessed=False)
    immutable_json(out/'bank_index.json',index)
    config=dict(version='subset12_logit_v1',selection=selected,subset_sha256=sha(subset_path),split_sha256=old['split_sha256'],
        old_config_sha256=sha(root/OLD/'config.json'),bank_sha256=sha(out/'bank_index.json'),features=acoustic['features'],envelope=DEFINITION,
        acoustic_checkpoint=old['acoustic_checkpoint'],acoustic_sha256=old['acoustic_sha256'],
        visual_checkpoint='UnLoc/evaluation_checkpoints/epoch020/model.ckpt',visual_sha256=read(root/'ReliabilityFusion/pilots/v1/config.json')['visual_sha256'],
        feature_columns=old['feature_columns'],weight_grid=old['weight_grid'],hidden=[32,16],initial_weight=.1,
        seeds=[42,43],runs=['success_set','logit_regression'],epochs=40,early_stop=False,lr=.001,weight_decay=0.,batch_size=16,
        optimizer='AdamW; shared matched40epoch budget, clip1, no scheduler; both methods no decay; scene-balanced eligible queries',
        target='same nearest successful0:.05:1 grid; logit target is midpoint of exact MAP-winner interval',
        empty_targets='no fabricated regression target; zero supervised loss, retain every query in train/val metrics',
        model_selection='Val scene-macro R1; ties Joint; then own-arm lower validation loss; include epoch0',
        counts=dict(train_scenes=12,train_queries=sum(e['queries'] for e in entries if e['split']=='train'),val_scenes=3,val_queries=sum(e['queries'] for e in entries if e['split']=='val')),
        candidate_grid_m=.2,yaw_step_deg=10,range_cap_m=10,desdf_spacing_m=.1,
        no_test_access=True,no_encoder_updates=True,score_source='frozen in-sample branch predictions, not out-of-fold',
        historical_notice='6scene fusion results preserved, NOT treated as train-matched controls')
    assert config['counts']==dict(train_scenes=12,train_queries=6387,val_scenes=3,val_queries=1704)
    immutable_json(out/'config.json',config);return index


def tasks(root,index,part):
    root=Path(root);all_tasks=[]
    for e in index['scenes']:
        if e['reuse_frozen_scores']:continue
        schedule=read(root/e['schedule_path']);assert sha(root/e['schedule_path'])==e['schedule_sha256']
        ids=[p['pose_id'] for p in schedule['positions']]
        for first in range(0,len(ids),12):
            all_tasks.append((str(root),str(root/'source/hssd-hab'),str(root/'RIR/tools/hard_negatives_epoch020_v1'),e,'FP',ids[first:first+12],2))
    return [(i,t) for i,t in enumerate(all_tasks) if i%2==part]


def extract_scene(root,entry,config,workers):
    result={}
    for role,key in (('FP','schedule_path'),('Observed','observation_schedule_path')):
        schedule=read(root/entry[key]);fingerprint=sha(root/entry[key]);paths=[]
        for p in schedule['positions']:
            for base in p['base_headings']:
                paths.append((root/f'RIR/{role}/train/{entry["scene_id"]}/records/p{p["pose_id"]:05d}/base_acoustic_yaw_{base:03d}.npy',p['centre_xyz_m'],base))
        folder=root/REL/'features/train'/entry['scene_id']/role;folder.mkdir(parents=True,exist_ok=True);rp=folder/'receipt.json'
        if rp.exists():
            r=read(rp);assert r['schedule_sha256']==fingerprint and r['config_sha256']==sha(root/REL/'config.json')
            for n,d in r['files'].items():assert sha(folder/n)==d
            result[role]=dict(path=str(rp.relative_to(root)),sha256=sha(rp));continue
        def work(item):
            path,xyz,base=item;meta=read(path.with_suffix('.json'))
            assert (meta['role'],meta['split'],meta['scene_id'])==(role,'train',entry['scene_id'])
            assert meta['schedule_sha256']==fingerprint and meta['acoustics']==PROTOCOL
            np.testing.assert_allclose(meta['centre_xyz_m'],xyz,atol=1e-7,rtol=0);assert meta['base_acoustic_yaw_deg']==base
            if role=='FP':
                frame=meta.get('loaded_frame') or {};assert frame.get('verified') and frame['up']==[0,1,0] and frame['front']==[0,0,-1]
                assert meta['geometry_sha256']==schedule['geometry_sha256'];np.testing.assert_allclose(frame['loaded_bounds'],schedule['height_contract']['mesh']['bounds_xyz_m'],atol=1e-4,rtol=0)
            payload=path.read_bytes();assert hashlib.sha256(payload).hexdigest()==meta['rir_sha256']
            array=np.load(io.BytesIO(payload),allow_pickle=False);e,c=ec_extract(array,config['features']);env=env_extract(array)
            return [e,c,env],meta['rir_sha256']
        arrays=None;digests=[];names=('energy','coherence','envelope')
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for first in range(0,len(paths),256):
                for i,(values,digest) in enumerate(pool.map(work,paths[first:first+256]),start=first):
                    if arrays is None:arrays=[open_memmap(folder/(n+'.tmp.npy'),mode='w+',dtype='float16',shape=(len(paths),*v.shape)) for n,v in zip(names,values)]
                    for arr,value in zip(arrays,values):arr[i]=value
                    digests.append(digest)
                atomic_json(folder/'progress.json',dict(complete=min(first+256,len(paths)),total=len(paths),updated_unix=time.time()))
        files={}
        for arr,name in zip(arrays,names):
            arr.flush();arr._mmap.close();(folder/(name+'.tmp.npy')).replace(folder/(name+'.npy'));files[name+'.npy']=sha(folder/(name+'.npy'))
        immutable_json(rp,dict(paths=[str(p.relative_to(root)) for p,_,_ in paths],raw_sha256=digests,files=files,
            features=config['features'],envelope=DEFINITION,schedule_sha256=fingerprint,config_sha256=sha(root/REL/'config.json'),read_once_for_three_streams=True))
        result[role]=dict(path=str(rp.relative_to(root)),sha256=sha(rp))
    return result


def cpu_run(root,part,workers=38,feature_workers=48):
    root=Path(root);out=root/REL;own=out/'preprocessing'/f'part{part}';own.mkdir(parents=True,exist_ok=True)
    lock=(own/'worker.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert os.environ.get('NVIDIA_VISIBLE_DEVICES')=='void' and os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
    index=read(out/'bank_index.json');config=read(out/'config.json');assert sha(out/'bank_index.json')==config['bank_sha256']
    def status(stage,**extra):atomic_json(own/'status.json',dict(status='running',stage=stage,part=part,updated_unix=time.time(),**extra))
    try:
        chosen=tasks(root,index,part);expected=sum(len(t[5])*6 for _,t in chosen);status('render_dense_train',expected_records=expected)
        supervisor_tools().supervise_tasks([t for _,t in chosen],own/'render',workers,expected_records=expected,
            task_ids=[i for i,_ in chosen],task_lock_dir=out/'preprocessing/task_locks')
        atomic_json(own/'render_done.json',dict(status='complete',bank_sha256=config['bank_sha256'],expected_records=expected,part=part))
        while not all((out/'preprocessing'/f'part{i}/render_done.json').exists() for i in (0,1)):
            status('wait_other_render_partition');time.sleep(15)
        features={}
        for sid in index['feature_assignment'][part]:
            status('verify_extract_three_streams',scene_id=sid)
            entry=next(e for e in index['scenes'] if e['scene_id']==sid)
            features[sid]=extract_scene(root,entry,config,feature_workers)
        immutable_json(own/'features_ready.json',dict(status='complete',features=features,bank_sha256=config['bank_sha256'],part=part,test_accessed=False))
        atomic_json(own/'status.json',dict(status='complete',part=part,updated_unix=time.time(),test_accessed=False))
    except Exception:
        atomic_json(own/'status.json',dict(status='failed',error=traceback.format_exc(),part=part,updated_unix=time.time()));raise


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','cpu']);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--part',type=int,choices=[0,1]);p.add_argument('--workers',type=int,default=38);p.add_argument('--feature-workers',type=int,default=48);a=p.parse_args()
    if a.mode=='prepare':print(prepare(a.root)['new_fp_arrays'])
    else:cpu_run(a.root,a.part,a.workers,a.feature_workers)

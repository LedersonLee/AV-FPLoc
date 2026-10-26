"""CPU-only corrected FP bank -> verified STFT/coherency features on node2."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np
from numpy.lib.format import open_memmap

from acoustic_branch.contracts import atomic_json, immutable_json, sha, PROTOCOL, recipe
from acoustic_branch.data import read
from acoustic_branch.features import extract
from acoustic_branch.render_adapter import supervisor_tools
from .prepare import REL,ACOUSTIC


def raw_path(root,entry,pose,base,role='FP'):
    return root/f'RIR/{role}/{entry["split"]}/{entry["scene_id"]}/records/p{pose:05d}/base_acoustic_yaw_{base:03d}.npy'


def reuse(root,index):
    """Reuse completed byte-identical corrected arrays at exactly matching poses."""
    plan=read(root/ACOUSTIC/'plan.json')
    selected={e['scene_id'] for e in index['scenes']}
    lookup={}
    for path,ref in plan['refs'].items():
        if Path(path).parts[3] in selected:
            lookup[(Path(path).parts[3],*np.round(ref['xyz'],7),recipe(ref['yaw'])[0])]=path
    work=[]
    for entry in index['scenes']:
        schedule=read(root/entry['schedule_path'])
        for p in schedule['positions']:
            for base in p['base_headings']:
                key=(entry['scene_id'],*np.round(p['centre_xyz_m'],7),base)
                if key not in lookup:continue
                work.append((entry,schedule,p,base,root/lookup[key].replace('RIR/FP_corrected_yup_v2/','RIR/FP/')))
    def reuse_one(item):
        entry,schedule,p,base,source=item;dest=raw_path(root,entry,p['pose_id'],base)
        if dest.exists() and dest.with_suffix('.json').exists():return 1
        if not source.exists():return 0
        metadata=read(source.with_suffix('.json'));frame=metadata.get('loaded_frame') or {}
        if not frame.get('verified') or frame.get('up')!=[0,1,0]:return 0
        if metadata['geometry_sha256']!=schedule['geometry_sha256'] or metadata['acoustics']!=PROTOCOL:
            raise ValueError('Reuse geometry/protocol mismatch')
        if metadata['rir_sha256']!=sha(source):raise ValueError('Reuse checksum mismatch')
        np.testing.assert_allclose(metadata['centre_xyz_m'],p['centre_xyz_m'],atol=1e-7,rtol=0)
        dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists() and sha(dest)!=metadata['rir_sha256']:
            # Only our new namespace, never an existing user/source recording.
            partial=dest.with_suffix('.uncommitted.npy')
            if partial.exists():raise ValueError('Repeated partial reuse requires inspection')
            dest.rename(partial)
        if not dest.exists():
            try:os.link(source,dest)
            except OSError:
                temporary=dest.with_suffix('.reuse.tmp.npy');shutil.copy2(source,temporary)
                if sha(temporary)!=metadata['rir_sha256']:raise ValueError('Reuse copy checksum mismatch')
                temporary.replace(dest)
        atomic_json(dest.with_suffix('.json'),dict(metadata,version='reliability_fusion_bank_v1',
            schedule_sha256=entry['schedule_sha256'],pose_id=p['pose_id'],centre_xyz_m=p['centre_xyz_m'],
            source_xyz_m=p['centre_xyz_m'],reused_from=str(source.relative_to(root))))
        return 1
    with ThreadPoolExecutor(max_workers=32) as pool:
        reused=sum(pool.map(reuse_one,work))
    return reused


def extract_scene(root,entry,config,workers):
    result={}
    for role,key in [('FP','schedule_path'),('Observed','observation_schedule_path')]:
        schedule=read(root/entry[key]);expected=sha(root/entry[key])
        paths=[raw_path(root,entry,p['pose_id'],base,role) for p in schedule['positions'] for base in p['base_headings']]
        folder=root/REL/'features'/entry['split']/entry['scene_id']/role
        folder.mkdir(parents=True,exist_ok=True)
        receipt_path=folder/'receipt.json'
        if receipt_path.exists():
            receipt=read(receipt_path)
            assert receipt['paths']==[str(p.relative_to(root)) for p in paths] and receipt['features']==config['features']
            for name,digest in receipt['files'].items():assert sha(folder/name)==digest
            result[role]=dict(path=str(receipt_path.relative_to(root)),sha256=sha(receipt_path));continue
        def work(path):
            meta=read(path.with_suffix('.json'))
            if meta['role']!=role or meta['split']!=entry['split'] or meta['scene_id']!=entry['scene_id'] or meta['acoustics']!=PROTOCOL:
                raise ValueError('Raw role/split/protocol mismatch')
            if meta['schedule_sha256']!=expected or sha(path)!=meta['rir_sha256']:
                raise ValueError('Raw schedule/checksum mismatch')
            if role=='FP':
                frame=meta.get('loaded_frame') or {}
                if not frame.get('verified') or frame.get('up')!=[0,1,0] or frame.get('front')!=[0,0,-1]:
                    raise ValueError('Uncorrected FP frame')
                np.testing.assert_allclose(frame['loaded_bounds'],schedule['height_contract']['mesh']['bounds_xyz_m'],atol=1e-4,rtol=0)
            e,c=extract(np.load(path,allow_pickle=False),config['features'])
            return e,c,meta['rir_sha256']
        arrays=None;digests=[]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            # Bound completed futures/arrays rather than retaining the entire bank.
            for first in range(0,len(paths),256):
                for i,(e,c,digest) in enumerate(pool.map(work,paths[first:first+256]),start=first):
                    if arrays is None:
                        arrays=[open_memmap(folder/'energy.tmp.npy',mode='w+',dtype='float16',shape=(len(paths),*e.shape)),
                                open_memmap(folder/'coherence.tmp.npy',mode='w+',dtype='float16',shape=(len(paths),*c.shape))]
                    arrays[0][i]=e;arrays[1][i]=c;digests.append(digest)
                atomic_json(root/REL/'feature_progress.json',dict(scene_id=entry['scene_id'],role=role,
                    complete=min(first+256,len(paths)),total=len(paths),updated_unix=time.time()))
        files={}
        for array,name in zip(arrays,('energy','coherence')):
            array.flush();array._mmap.close();(folder/(name+'.tmp.npy')).replace(folder/(name+'.npy'));files[name+'.npy']=sha(folder/(name+'.npy'))
        immutable_json(receipt_path,dict(paths=[str(p.relative_to(root)) for p in paths],raw_sha256=digests,
            files=files,features=config['features'],frame_version='corrected_yup_v2',schedule_sha256=expected))
        result[role]=dict(path=str(receipt_path.relative_to(root)),sha256=sha(receipt_path))
    return result


def run(args):
    root=args.root;out=root/REL
    lock=(out/'preprocess.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='-1' or os.environ.get('NVIDIA_VISIBLE_DEVICES')!='void':
        raise ValueError('Node2 GPU must remain inaccessible')
    index=read(out/'bank_index.json');config=read(out/'config.json')
    assert sha(out/'bank_index.json')==config['bank_sha256']
    assert sha(root/'manifests/scene_splits_v1.json')==config['split_sha256']
    def status(stage,**extra):atomic_json(out/'preprocess_status.json',dict(status='running',stage=stage,updated_unix=time.time(),**extra))
    try:
        status('reuse_verified_completed_arrays')
        reused=reuse(root,index);status('render_corrected_dense_banks',reused_arrays=reused)
        tasks=[]
        for entry in index['scenes']:
            schedule=read(root/entry['schedule_path']);ids=[p['pose_id'] for p in schedule['positions']]
            for first in range(0,len(ids),12):
                tasks.append((str(root),str(root/'source/hssd-hab'),str(args.renderer),entry,'FP',ids[first:first+12],2))
        supervisor_tools().supervise_tasks(tasks,out/'render',args.workers,
            expected_records=index['fp_arrays'],task_lock_dir=out/'task_locks')
        status('verify_and_extract_features',reused_arrays=reused)
        features={}
        for entry in index['scenes']:
            features[entry['scene_id']]=extract_scene(root,entry,config,args.feature_workers)
        atomic_json(out/'ready.json',dict(status='complete',bank_sha256=sha(out/'bank_index.json'),
            config_sha256=sha(out/'config.json'),features=features,verified_fp_arrays=index['fp_arrays'],
            test_accessed=False,reused_arrays=reused,updated_unix=time.time()))
        atomic_json(out/'preprocess_status.json',dict(status='complete',stage='features_ready',reused_arrays=reused,updated_unix=time.time()))
    except Exception:
        atomic_json(out/'preprocess_status.json',dict(status='failed',error=traceback.format_exc(),updated_unix=time.time()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--renderer',type=Path,required=True);p.add_argument('--workers',type=int,default=38)
    p.add_argument('--feature-workers',type=int,default=64);run(p.parse_args())

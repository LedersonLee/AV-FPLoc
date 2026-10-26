"""Add checked envelope features to the frozen fusion bank; never render RIRs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import io
import os
from pathlib import Path
import time
import traceback

import numpy as np

from acoustic_branch.contracts import atomic_json, immutable_json, sha
from acoustic_branch.data import read
from acoustic_branch.envelope_cache import extract, DEFINITION
from .prepare import REL as OLD

REL = 'ReliabilityFusion/pilots/ecn14_v1'
MODEL = 'AcousticContrastive/pilots/scene_subset_v1/input_ablation_v1'


def prepare(root):
    root = Path(root); out = root/REL
    index = read(root/OLD/'bank_index.json')
    membership = read(root/'manifests/scene_splits_v1.json')
    all_ids = [s for v in index['selection'].values() for s in v]
    if len(set(all_ids)) != len(all_ids): raise ValueError('Overlapping scenes')
    for split, scenes in index['selection'].items():
        if not set(scenes) <= set(membership[split]): raise ValueError('Parent split mismatch')
    config = dict(version='ecn14_v1', seed=42, old_bank_sha256=sha(root/OLD/'bank_index.json'),
        selection=index['selection'], split_sha256=sha(root/'manifests/scene_splits_v1.json'),
        acoustic_checkpoint=MODEL+'/runs/all/best.pt', acoustic_sha256=sha(root/MODEL/'runs/all/best.pt'),
        acoustic_config_sha256=sha(root/MODEL/'configs/all.json'),
        old_ready_sha256=sha(root/OLD/'ready.json'),
        old_scores_sha256=sha(root/OLD/'scores_ready.json'), envelope=DEFINITION,
        score_temperature_grid=[.25,.5,1.], weight_grid=np.linspace(0,1,21).tolist(),
        feature_columns=[*range(10),13,14,15,16], hidden=[32,16], epochs=40, patience=8,
        lr=.001, weight_decay=.001, batch_size=16, xy_sigma_m=.2, yaw_sigma_deg=10.,
        selection_rule='val scene-macro R1, then joint R1/30, then lower loss',
        fixed_weight_selection='train scene-macro R1, then joint, then closest to .5',
        margin_m=1., margin_yaw_deg=30., score_margin=.1,
        runs=['adaptive_nll_T1_half','adaptive_nll_T1','adaptive_nll_T05',
              'adaptive_nll_T025','adaptive_margin_xy','adaptive_margin_joint','constant_nll_T1'],
        test_notice='Previously inspected exploratory subset; not a blind new test. No test fitting.',
        score_source='frozen in-sample train branch predictions, not out-of-fold',
        yaw_step_deg=10, desdf_spacing_m=.1, range_cap_m=10, candidate_grid_m=.2)
    immutable_json(out/'config.json', config)
    return config


def preprocess(root, workers=48):
    root=Path(root); out=root/REL; config=read(out/'config.json')
    if os.environ.get('NVIDIA_VISIBLE_DEVICES')!='void': raise ValueError('CPU-only worker required')
    lock=(out/'envelope.lock').open('a+'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if sha(root/OLD/'ready.json')!=config['old_ready_sha256']: raise ValueError('Old ready changed')
    ready=read(root/OLD/'ready.json'); index=read(root/OLD/'bank_index.json')
    if sha(root/OLD/'bank_index.json')!=config['old_bank_sha256']: raise ValueError('Bank changed')
    entries=[]; complete=0; total=sum(len(read(root/v['path'])['paths']) for r in ready['features'].values() for v in r.values())
    start=time.time()
    def status(**extra):
        atomic_json(out/'preprocess_status.json',dict(status='running',complete=complete,total=total,
            seconds=time.time()-start,workers=workers,updated_unix=time.time(),test_accessed=False,**extra))
    try:
        for entry in index['scenes']:
            split,scene=entry['split'],entry['scene_id']
            if split not in ('train','val') or scene not in config['selection'][split]: raise ValueError('TEST forbidden')
            for role in ('FP','Observed'):
                ref=ready['features'][scene][role]; receipt=root/ref['path']
                if sha(receipt)!=ref['sha256']: raise ValueError('Feature receipt changed')
                source=read(receipt); paths=source['paths']; digests=source['raw_sha256']; chunks=[]
                def work(i):
                    path=root/paths[i]; meta=read(path.with_suffix('.json'))
                    if (meta['split'],meta['scene_id'],meta['role'])!=(split,scene,role): raise ValueError('Raw identity mismatch')
                    payload=path.read_bytes()
                    if hashlib.sha256(payload).hexdigest()!=digests[i] or meta['rir_sha256']!=digests[i]: raise ValueError('Raw checksum changed')
                    return extract(np.load(io.BytesIO(payload),allow_pickle=False)).astype(np.float16)
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    for first in range(0,len(paths),256):
                        end=min(first+256,len(paths)); folder=out/'envelope'/split/scene/role/f'{first//256:05d}'
                        folder.mkdir(parents=True,exist_ok=True); rp=folder/'receipt.json'
                        contract=dict(source_receipt_sha256=ref['sha256'],paths=paths[first:end],raw_sha256=digests[first:end],
                            definition=DEFINITION,config_sha256=sha(out/'config.json'))
                        if rp.exists():
                            row=read(rp)
                            if row['contract']!=contract or sha(folder/'envelope.npy')!=row['sha256']: raise ValueError('Envelope shard changed')
                        else:
                            values=list(pool.map(work,range(first,end)))
                            np.save(folder/'envelope.tmp.npy',np.stack(values),allow_pickle=False)
                            (folder/'envelope.tmp.npy').replace(folder/'envelope.npy')
                            immutable_json(rp,dict(contract=contract,sha256=sha(folder/'envelope.npy')))
                        chunks.append(dict(first=first,count=end-first,path=str((folder/'envelope.npy').relative_to(root)),
                            receipt_sha256=sha(rp))); complete+=end-first
                        status(scene_id=scene,split=split,role=role)
                entries.append(dict(scene_id=scene,split=split,role=role,chunks=chunks,records=len(paths)))
        immutable_json(out/'envelope_ready.json',dict(status='complete',config_sha256=sha(out/'config.json'),
            entries=entries,arrays=complete,test_accessed=False))
        atomic_json(out/'preprocess_status.json',dict(status='complete',complete=complete,total=total,
            seconds=time.time()-start,test_accessed=False,updated_unix=time.time()))
    except Exception:
        atomic_json(out/'preprocess_status.json',dict(status='failed',complete=complete,total=total,
            error=traceback.format_exc(),updated_unix=time.time())); raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--workers',type=int,default=48);a=p.parse_args();preprocess(a.root,a.workers)

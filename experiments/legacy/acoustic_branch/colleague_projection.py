"""HSSD adaptation of the colleague's yaw-zero linear invariant projection.

Exact reference feature and ranking loss; original12/3/3 scene split. The
acoustic model is XY-only. Query microphones remain in their actual order:
reconstructing a stored physical capture is not GT-yaw canonicalization.
"""
import argparse
import io
from concurrent.futures import ThreadPoolExecutor
import fcntl
from functools import lru_cache
import importlib.util
from pathlib import Path
import sys
import time

import numpy as np

from .contracts import PROTOCOL, atomic_json, immutable_json, sha
from .data import read

REL = 'AcousticContrastive/colleague_projection_v1'
SUBSET = 'AcousticContrastive/pilots/scene_subset_v1/scene_splits.json'
CORRECTED = 'AcousticContrastive/pilots/scene_subset_v1/corrected_yup_v2'


@lru_cache(None)
def reference(path):
    spec=importlib.util.spec_from_file_location('hssd_colleague_grid_reference',path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module


def describe(rir, path):
    ref=reference(str(path))
    config=ref.GridScoreConfig(feature='stft_band',nfft=256,hop=64,sample_rate_hz=48000,
        direct_guard_samples=96,usable_samples=6144)
    result=ref.featurise(ref.band_energy(rir,config,48000),config).astype(np.float32)
    if result.shape!=(18,97) or not np.isfinite(result).all():
        raise ValueError('Reference feature must be finite18x97')
    return result.reshape(-1)


def membership(root):
    full=read(root/'manifests/scene_splits_v1.json');subset=read(root/SUBSET)
    if subset['parent_split_sha256']!=sha(root/'manifests/scene_splits_v1.json'):
        raise ValueError('Original split changed')
    owners=[s for scenes in full.values() for s in scenes]
    if len(set(owners))!=len(owners):raise ValueError('Global split overlap')
    selected=subset['scenes']
    for split in ('train','val','test'):
        if not set(selected[split])<=set(full[split]):raise ValueError('Subset leakage')
    if {k:len(v) for k,v in selected.items()}!={'train':12,'val':3,'test':3}:
        raise ValueError('Expected unchanged12/3/3 subset')
    return selected


def load_raw(root,path,schedule,role,roll=0,single_read=False):
    p=root/path;m=read(p.with_suffix('.json'))
    if (m['role']!=role or m['split']!=schedule['split'] or m['scene_id']!=schedule['scene_id']
        or m['schedule_sha256']!=schedule['_sha256'] or m['acoustics']!=PROTOCOL):
        raise ValueError('Raw RIR protocol/split/schedule mismatch: '+str(p))
    if single_read:
        import hashlib
        payload=p.read_bytes()
        digest=hashlib.sha256(payload).hexdigest()
    else:digest=sha(p)
    if digest!=m['rir_sha256']:raise ValueError('Raw checksum mismatch: '+str(p))
    if role=='FP':
        frame=m.get('loaded_frame') or {}
        if (not frame.get('verified') or frame.get('up')!=[0,1,0] or frame.get('front')!=[0,0,-1]
            or m['geometry_sha256']!=schedule['geometry_sha256']):
            raise ValueError('Legacy/uncorrected extrusion forbidden')
        np.testing.assert_allclose(frame['loaded_bounds'],schedule['height_contract']['mesh']['bounds_xyz_m'],atol=1e-4,rtol=0)
    a=np.load(io.BytesIO(payload) if single_read else p,allow_pickle=False)
    if a.ndim!=2 or a.shape[0]!=6 or not np.isfinite(a).all() or not np.all(np.any(a!=0,axis=1)):
        raise ValueError('Invalid physical RIR')
    return np.roll(a,int(roll),axis=0),m['rir_sha256']


def schedule_at(root,path,split,scene):
    s=read(root/path)
    if s['split']!=split or s['scene_id']!=scene or s['split_sha256']!=sha(root/'manifests/scene_splits_v1.json'):
        raise ValueError('Schedule split changed')
    if sha(root/s['geometry_path'])!=s['geometry_sha256']:raise ValueError('Geometry changed')
    s['_sha256']=sha(root/path)
    return s


def cache_train_val(root,ref_path,workers=8):
    out=root/REL;out.mkdir(parents=True,exist_ok=True)
    selected=membership(root)
    contract=dict(version='colleague_projection_hssd_v1',split_sha256=sha(root/'manifests/scene_splits_v1.json'),
        subset_sha256=sha(root/SUBSET),reference_sha256=sha(ref_path),
        input_shape=[18,97],dimension=128,temperature=.05,lr=.001,epochs=40,per_room=400,
        seed=0,optimizer='Adam, no scheduler, no explicit weight decay',
        selection='val paired-pose in-room micro recall with distance<1m; earliest tie',
        acoustic_heading='FP world0 only; no query rotation compensation; no acoustic yaw prediction',
        adaptation='HSSD actual available yaw0 paired physical positions, at most400 per scene; original scene membership',
        test_accessed_for_fitting=False)
    immutable_json(out/'config.json',contract)
    for split in ('train','val'):
        for scene in selected[split]:
            path=f'RIR/manifests/positive_pairs_v1/schedules/{split}/{scene}.json'
            schedule=schedule_at(root,path,split,scene)
            positions=[p for p in schedule['positions'] if 0 in p['base_headings']]
            ids=np.linspace(0,len(positions)-1,min(400,len(positions))).astype(int)
            positions=[positions[i] for i in ids]
            folder=out/'features'/split/scene;folder.mkdir(parents=True,exist_ok=True)
            if (folder/'receipt.json').exists():
                saved=read(folder/'receipt.json')
                if saved['config_sha256']!=sha(out/'config.json') or saved['schedule_sha256']!=schedule['_sha256']:
                    raise ValueError('Stale projection feature cache')
                if sha(folder/'features.npz')!=saved['features_sha256']:raise ValueError('Projection cache checksum mismatch')
                continue
            def work(p):
                features=[];records=[]
                for role in ('Observed','FP'):
                    raw=f'RIR/{role}/{split}/{scene}/records/p{p["pose_id"]:05d}/base_acoustic_yaw_000.npy'
                    a,digest=load_raw(root,raw,schedule,role)
                    features.append(describe(a,ref_path));records.append(dict(path=raw,sha256=digest))
                return features,records
            with ThreadPoolExecutor(max_workers=workers) as pool:values=list(pool.map(work,positions))
            features=np.array([v[0] for v in values])
            xy=np.array([[p['centre_xyz_m'][0],p['centre_xyz_m'][2]] for p in positions])
            tmp=folder/'features.tmp.npz'
            np.savez(tmp,observed=features[:,0],fp=features[:,1],xy=xy,pose_ids=[p['pose_id'] for p in positions])
            tmp.replace(folder/'features.npz')
            immutable_json(folder/'receipt.json',dict(config_sha256=sha(out/'config.json'),
                schedule_sha256=schedule['_sha256'],features_sha256=sha(folder/'features.npz'),
                positions=len(positions),raw_records=[r for v in values for r in v[1]]))
            print(f'[cached] {split}/{scene}: {len(positions)} paired yaw0 positions',flush=True)
    atomic_json(out/'features_ready.json',dict(status='complete',test_accessed=False,
        config_sha256=sha(out/'config.json'),updated_unix=time.time()))


def train(root,wandb_mode='online'):
    import torch
    out=root/REL;config=read(out/'config.json');selected=membership(root)
    assert read(out/'features_ready.json')['config_sha256']==sha(out/'config.json')
    run=out/'runs/linear_yaw0';run.mkdir(parents=True,exist_ok=True)
    if (run/'complete.json').exists():
        saved=read(run/'complete.json')
        if saved['best_sha256']!=sha(run/'best.pt'):raise ValueError('Selected checkpoint changed')
        return
    np.random.seed(0);torch.manual_seed(0);torch.set_num_threads(4)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    data={};receipts={}
    for split in ('train','val'):
        data[split]={}
        for scene in selected[split]:
            folder=out/'features'/split/scene;r=read(folder/'receipt.json')
            assert r['features_sha256']==sha(folder/'features.npz')
            receipts[f'{split}/{scene}']=sha(folder/'receipt.json')
            with np.load(folder/'features.npz') as z:
                data[split][scene]=dict(f=torch.tensor(z['observed'],device=device),
                    c=torch.tensor(z['fp'],device=device),xy=z['xy'].copy())
    immutable_json(run/'training_contract.json',dict(config_sha256=sha(out/'config.json'),features=receipts,
        scene_membership=selected,test_used_for_fitting=False))
    W=torch.nn.Parameter(.1*torch.eye(1746,128,device=device)+.01*torch.randn(1746,128,device=device))
    optimizer=torch.optim.Adam([W],lr=.001)
    tracker=None
    if wandb_mode!='disabled':
        import wandb
        tracker=wandb.init(project='hssd-acoustic-contrastive',name='colleague_projection_hssd12_yaw0',
            config=config,dir=str(run),mode=wandb_mode)
        atomic_json(run/'wandb.json',dict(id=tracker.id,url=tracker.url))
    def validation(matrix):
        hits=total=0;per_scene={}
        with torch.inference_mode():
            for scene,d in data['val'].items():
                f,c=d['f'],d['c']
                if matrix is not None:f,c=f@matrix,c@matrix
                j=torch.cdist(f,c,p=1).argmin(dim=1).cpu().numpy()
                errors=np.linalg.norm(d['xy'][j]-d['xy'],axis=1)
                count=int((errors<1).sum());hits+=count;total+=len(errors)
                per_scene[scene]=dict(positions=len(errors),recall_1m=float((errors<1).mean()))
        return hits/total,per_scene
    identity,identity_scenes=validation(None)
    history=[];best=-1.;rooms=list(data['train'])
    def save(path,epoch):
        tmp=path.with_suffix('.tmp.pt')
        torch.save(dict(W=W.detach().cpu(),optimizer=optimizer.state_dict(),epoch=epoch,
            config=config,torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state()),tmp)
        tmp.replace(path)
    for epoch in range(1,41):
        np.random.shuffle(rooms);losses=[]
        for scene in rooms:
            d=data['train'][scene];f,c=d['f']@W,d['c']@W
            logits=-torch.cdist(f,c,p=1)/.05
            loss=torch.nn.functional.cross_entropy(logits,torch.arange(len(f),device=device))
            if not torch.isfinite(loss):raise ValueError('Nonfinite projection loss')
            optimizer.zero_grad();loss.backward();optimizer.step();losses.append(float(loss.detach()))
        recall,per_scene=validation(W.detach())
        row=dict(epoch=epoch,train_loss=float(np.mean(losses)),val_recall_1m=recall)
        history.append(row);save(run/'last.pt',epoch)
        if recall>best:
            best=recall;save(run/'best.pt',epoch)
            atomic_json(run/'best_metrics.json',dict(epoch=epoch,val_recall_1m=recall,per_scene=per_scene))
        atomic_json(run/'history.json',history)
        atomic_json(run/'status.json',dict(status='running',epoch=epoch,best_val_recall_1m=best,test_accessed=False))
        if tracker:tracker.log(row,step=epoch)
        print(row,flush=True)
    best_metrics=read(run/'best_metrics.json')
    atomic_json(run/'complete.json',dict(status='complete',epochs=40,stopped_early=False,
        best_epoch=best_metrics['epoch'],best_val_recall_1m=best,identity_val_recall_1m=identity,
        identity_val_per_scene=identity_scenes,last_val_recall_1m=history[-1]['val_recall_1m'],
        best_sha256=sha(run/'best.pt'),config_sha256=sha(out/'config.json'),test_accessed=False))
    if tracker:tracker.finish()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['cache','train'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--reference',type=Path)
    p.add_argument('--workers',type=int,default=8);p.add_argument('--wandb-mode',default='online')
    a=p.parse_args()
    if a.action=='cache':
        if not a.reference:p.error('cache needs --reference')
        cache_train_val(a.root,a.reference,a.workers)
    else:train(a.root,a.wandb_mode)


if __name__=='__main__':main()

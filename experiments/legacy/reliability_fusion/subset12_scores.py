"""Train12/Val3 frozen score caches and checked logit regression labels."""
from pathlib import Path
import shutil
import time

import numpy as np
from numpy.lib.format import open_memmap
import torch

from acoustic_branch.contracts import atomic_json,sha
from acoustic_branch.data import read
from acoustic_branch.evaluation import standardized
from acoustic_branch.input_ablation_dense import embeddings
from acoustic_branch.pilot_evaluate import angular_rays,bank_windows
from acoustic_branch.train import AcousticEncoder
from .core import features
from .overfit_probe import winner_interval
from .success_set import good_weights
from .revision_data import REL as OLD
from .subset12_data import REL


def checked(folder,expected=None):
    receipt=read(folder/'receipt.json')
    if expected is not None:assert sha(folder/'receipt.json')==expected
    for name,digest in receipt['files'].items():assert sha(folder/name)==digest,'Cache integrity: '+str(folder/name)
    return receipt


def load_scenes(root,ssd,cache,config,status):
    out=root/REL;index=read(out/'bank_index.json');old_refs={e['scene_id']:e for e in read(root/OLD/'scores_ready.json')['entries']}
    worker_ready=[read(out/'preprocessing'/f'part{i}/features_ready.json') for i in (0,1)]
    raw_features={sid:roles for r in worker_ready for sid,roles in r['features'].items()}
    assert all(r['bank_sha256']==config['bank_sha256'] and not r['test_accessed'] for r in worker_ready)
    assert sha(root/config['acoustic_checkpoint'])==config['acoustic_sha256']
    checkpoint=torch.load(root/config['acoustic_checkpoint'],map_location='cpu',weights_only=False)
    assert checkpoint['config']['subset_counts']==dict(train=12,val=3,test=3)
    model=AcousticEncoder(**checkpoint['config']['model']).cuda().eval();model.load_state_dict(checkpoint['model']);scenes=[];refs=[]
    for entry in index['scenes']:
        sid,split=entry['scene_id'],entry['split'];assert split in ('train','val') and sid in config['selection'][split]
        if entry['reuse_frozen_scores']:
            ref=old_refs[sid];folder=ssd/OLD/'cache'/ref['cache_directory'];checked(folder,ref['receipt_sha256']);base='old'
        else:
            folder=cache/split/sid;folder.mkdir(parents=True,exist_ok=True);base='new'
            if (folder/'receipt.json').exists():
                receipt=checked(folder);assert receipt['config_sha256']==sha(out/'config.json')
            else:
                roles={};envelopes={}
                for role in ('FP','Observed'):
                    reference=raw_features[sid][role];rp=root/reference['path'];assert sha(rp)==reference['sha256'];r=read(rp)
                    assert r['config_sha256']==sha(out/'config.json') and r['features']==config['features'] and r['envelope']==config['envelope']
                    for name,digest in r['files'].items():assert sha(rp.parent/name)==digest
                    roles[role]=(np.load(rp.parent/'energy.npy',mmap_mode='r'),np.load(rp.parent/'coherence.npy',mmap_mode='r'),r)
                    envelopes[role]=np.load(rp.parent/'envelope.npy',mmap_mode='r')
                queries=read(root/entry['observation_schedule_path'])['queries'];status('encode_new_train_bank',scene_id=sid,queries=len(queries),cells=entry['cells'])
                candidate,observed=embeddings(model,entry,roles['FP'],roles['Observed'],envelopes,queries,256)
                vision=ssd/'UnLoc/train_likelihood/epoch020';assert read(vision/'protocol.json')['checkpoint_sha256']==config['visual_sha256']
                depth_path=vision/'depth'/f'{sid}.npz';assert sha(depth_path)==read(vision/'scenes'/f'{sid}.json')['depth_sha256']
                with np.load(depth_path) as b:
                    lut={int(v):i for i,v in enumerate(b['sample_indices'])};ids=[lut[q['sample_index']] for q in queries]
                    depth,scale=b['location'][ids].copy(),b['scale'][ids].copy()
                xy=np.asarray(entry['candidate_xz']);payload=np.load(ssd/'UnLoc/structural_targets'/sid/'desdf.npy',allow_pickle=True).item()
                assert payload['max_range_m']==10 and payload['desdf_spacing_m']==.1
                windows=bank_windows(payload,xy);scores=open_memmap(folder/'scores.tmp.npy',mode='w+',dtype=np.float64,shape=(len(queries),2,len(xy),36))
                cg=torch.tensor(candidate.reshape(-1,128),dtype=torch.float64,device='cuda');vectors=[];truth=[]
                for first in range(0,len(queries),8):
                    with torch.no_grad():audio=(torch.tensor(observed[first:first+8],dtype=torch.float64,device='cuda')@cg.T).cpu().numpy().reshape(-1,len(xy),36)
                    if first==0:np.testing.assert_allclose(audio[0],np.einsum('cyd,d->cy',candidate.astype(np.float64),observed[0].astype(np.float64)),atol=1e-10,rtol=0)
                    for j,a in enumerate(audio):
                        i=first+j;rays,scales=angular_rays(depth[i],scale[i]);v=standardized(-(abs(windows-rays)/scales).sum(-1)/40)
                        scores[i,0]=v;scores[i,1]=standardized(a);vectors.append(features(v,a,xy,np.ones(11))[config['feature_columns']])
                        q=queries[i];truth.append([q['centre_xyz_m'][0],q['centre_xyz_m'][2],q['world_yaw_deg']])
                    if first%64==0:status('score_new_train_bank',scene_id=sid,complete=min(first+8,len(queries)),total=len(queries))
                del cg,candidate,observed,windows;scores.flush();scores._mmap.close();(folder/'scores.tmp.npy').replace(folder/'scores.npy')
                for name,value in [('features',vectors),('truth',truth),('xy',xy)]:np.save(folder/(name+'.npy'),np.asarray(value))
                atomic_json(folder/'queries.json',queries);names=['scores.npy','features.npy','truth.npy','xy.npy','queries.json']
                atomic_json(folder/'receipt.json',dict(config_sha256=sha(out/'config.json'),files={n:sha(folder/n) for n in names},
                    acoustic_sha256=config['acoustic_sha256'],visual_sha256=config['visual_sha256'],scene_id=sid,split=split,queries=len(queries),test_accessed=False))
        arrays={n:np.load(folder/(n+'.npy'),mmap_mode='r') for n in ('scores','features','truth','xy')}
        assert arrays['features'].shape==(entry['queries'],14) and len(arrays['xy'])==entry['cells']
        scenes.append(dict(scene_id=sid,split=split,folder=folder,arrays=arrays,entry=entry))
        refs.append(dict(scene_id=sid,split=split,base=base,folder=str(folder),receipt_sha256=sha(folder/'receipt.json'),queries=entry['queries']))
    del model;torch.cuda.empty_cache()
    assert sum(len(s['arrays']['truth']) for s in scenes if s['split']=='train')==6387
    assert sum(len(s['arrays']['truth']) for s in scenes if s['split']=='val')==1704
    atomic_json(out/'scores_ready.json',dict(status='complete',entries=refs,config_sha256=sha(out/'config.json'),test_accessed=False));return scenes


def attach_targets(root,cache,scenes,config,status):
    out=root/REL;grid=np.array(config['weight_grid']);stats={};digests={}
    for s in scenes:
        sid=s['scene_id'];path=cache/'targets'/s['split']/(sid+'.npz');path.parent.mkdir(parents=True,exist_ok=True);rp=path.with_suffix('.json')
        if path.exists() and rp.exists():
            receipt=read(rp);assert receipt['score_receipt_sha256']==sha(s['folder']/'receipt.json') and receipt['sha256']==sha(path) and receipt['config_sha256']==sha(out/'config.json')
            with np.load(path) as b:
                for k in b.files:s[k]=b[k].copy()
        else:
            a=s['arrays'];curve_path=s['folder']/'weight_curve.npz';poses=[]
            if curve_path.exists():
                with np.load(curve_path) as b:poses=b['poses'].copy()
            else:
                wg=torch.tensor(grid,dtype=torch.float64,device='cuda')[:,None,None]
                for i,(v,ac) in enumerate(a['scores']):
                    with torch.no_grad():ids=((1-wg)*torch.tensor(v,device='cuda')+wg*torch.tensor(ac,device='cuda')).flatten(1).argmax(1).cpu().numpy()
                    poses.append(np.column_stack([a['xy'][ids//36],ids%36*10]))
                    if i%100==0:status('train_val_grid_targets',scene_id=sid,query=i,total=len(a['truth']))
                poses=np.asarray(poses)
            good=good_weights(poses,a['truth']);target=np.full(len(good),.1,np.float64);valid=np.zeros(len(good),bool);bounds=np.full((len(good),2),np.nan)
            for i,g in enumerate(good):
                if not g.any():continue
                v,ac=a['scores'][i]
                for idx in np.flatnonzero(g)[np.argsort(abs(grid[g]-.1),kind='stable')]:
                    winner,lo,hi=winner_interval(v,ac,float(grid[idx]));centre=(lo+hi)/2
                    if hi-lo<=1e-12 or not 0<centre<1:continue
                    win=int(np.argmax((1-centre)*v+centre*ac));p=np.r_[a['xy'][win//36],win%36*10];truth=a['truth'][i]
                    if np.linalg.norm(p[:2]-truth[:2])<=1 and abs((p[2]-truth[2]+180)%360-180)<=30:
                        target[i]=centre;bounds[i]=[lo,hi];valid[i]=True;break
            temporary=path.with_suffix('.tmp.npz');np.savez_compressed(temporary,good=good,valid=valid,interior_target=target,target_bounds=bounds,grid_poses=poses);temporary.replace(path)
            atomic_json(rp,dict(sha256=sha(path),score_receipt_sha256=sha(s['folder']/'receipt.json'),config_sha256=sha(out/'config.json'),test_accessed=False))
            s.update(good=good,valid=valid,interior_target=target,target_bounds=bounds,grid_poses=poses)
        stats[sid]=dict(split=s['split'],queries=len(s['valid']),grid_success_possible=int(s['good'].any(1).sum()),supervised_queries=int(s['valid'].sum()),
            empty_grid=int((~s['good'].any(1)).sum()),no_strict_interior=int((s['good'].any(1)&~s['valid']).sum()))
        digests[sid]=sha(path)
    atomic_json(out/'targets_ready.json',dict(status='complete',scenes=stats,sha256=digests,test_accessed=False));return stats

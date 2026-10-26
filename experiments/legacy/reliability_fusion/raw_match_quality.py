"""Recover absolute acoustic-match quality discarded by query z-scoring.

Re-encode four GT-free probe poses per query, not the whole dense bank. Their
raw cosines determine the affine mean/std removed by standardization; two
extra probes check the reconstruction. No new RIR, encoder training or Test.
"""
import argparse
from pathlib import Path
import time
import traceback

import numpy as np

from acoustic_branch.contracts import atomic_json,immutable_json,sha,recipe
from .input_supervision_audit import read,resolve
from .case_multilabel import DATA

REL='ReliabilityFusion/pilots/raw_match_quality_v2'
SOURCE='ReliabilityFusion/pilots/subset12_logit_v1'


def reconstruct(z,raw):
    z,raw=np.asarray(z,dtype=np.float64),np.asarray(raw,dtype=np.float64)
    dz=z-z.mean(-1,keepdims=True);dr=raw-raw.mean(-1,keepdims=True)
    sigma=(dz*dr).sum(-1)/(dz*dz).sum(-1)
    mean=raw.mean(-1)-sigma*z.mean(-1)
    residual=np.max(abs(raw-(mean[...,None]+sigma[...,None]*z)),axis=-1)
    if not(np.isfinite(mean).all()and np.isfinite(sigma).all()):raise ValueError('Nonfinite affine fit')
    if np.any(sigma<=1e-6)or np.any(sigma>1)or np.any(residual>2e-5):
        raise ValueError(f'Raw/zscore probes disagree: std {sigma.min()}..{sigma.max()}, max residual {residual.max()}')
    return mean,sigma,residual


def encode(model,e,c,n,items,batch):
    import torch
    result=[]
    with torch.inference_mode():
        for first in range(0,len(items),batch):
            ee=[];cc=[];nn=[]
            for index,roll in items[first:first+batch]:
                ee.append(np.roll(np.array(e[index],dtype=np.float32),roll,axis=0))
                cc.append(np.roll(np.array(c[index],dtype=np.float32),roll,axis=1))
                nn.append(np.roll(np.array(n[index],dtype=np.float32),roll,axis=0))
            tensors=[torch.as_tensor(np.stack(v),device='cuda')for v in(ee,cc,nn)]
            result.append(model(*tensors).cpu().numpy())
    return np.concatenate(result)


def encode_fp_original_shapes(model,e,c,n,items,batch,cells):
    """Match dense encoder batch shapes without evaluating all bank poses.

    FP dense encoding used batches of256 cells per heading. cuDNN kernels can
    differ in last-batch shapes even for a GroupNorm model. Pad sparse probes
    to the original batch LENGTH (no BatchNorm/cross-sample operations).
    """
    groups={}
    for i,(physical,roll)in enumerate(items):
        cell=physical//6;length=min(batch,cells-(cell//batch)*batch)
        groups.setdefault(length,[]).append((i,(physical,roll)))
    result=np.empty((len(items),128),np.float32)
    for length,group in groups.items():
        for first in range(0,len(group),length):
            chunk=group[first:first+length];values=[item for _,item in chunk]
            padded=values+[values[-1]]*(length-len(values))
            encoded=encode(model,e,c,n,padded,length)
            for j,(index,_)in enumerate(chunk):result[index]=encoded[j]
    return result


def arrays(root,entry,new_roles):
    """Reuse exact checked energy/coherence/envelope inputs of frozen model."""
    sid=entry['scene_id']
    if not entry['reuse_frozen_scores']:
        result={}
        for role in('FP','Observed'):
            ref=new_roles[sid][role];path=root/ref['path'];assert sha(path)==ref['sha256'];r=read(path)
            for name,digest in r['files'].items():assert sha(path.parent/name)==digest
            result[role]=tuple(np.load(path.parent/(name+'.npy'),mmap_mode='r')for name in('energy','coherence','envelope'))+(r,)
        return result
    from acoustic_branch.pilot_evaluate import feature_arrays
    from .revision import envelope_arrays
    old=root/'ReliabilityFusion/pilots/v1';revision=root/'ReliabilityFusion/pilots/ecn14_v1'
    ready=read(old/'ready.json');env_ready=read(revision/'envelope_ready.json')
    env=envelope_arrays(root,revision,entry,env_ready);result={}
    for role in('FP','Observed'):
        e,c,r=feature_arrays(root,ready,sid,role);result[role]=(e,c,env[role],r)
    return result


def run(root,ssd,batch=256,limit=None):
    import torch
    from acoustic_branch.train import AcousticEncoder
    torch.set_num_threads(4);out=root/REL;out.mkdir(parents=True,exist_ok=True)
    parent=read(root/SOURCE/'config.json');ready=read(root/DATA/'ready.json')
    refs=sorted(ready['scenes'],key=lambda r:(r['split']!='train',r['scene_id']))
    assert len(refs)==15 and {r['split']for r in refs}=={'train','val'}
    checkpoint=root/parent['acoustic_checkpoint'];assert sha(checkpoint)==parent['acoustic_sha256']
    cfg=dict(version='v2',source_code_sha256=sha(__file__),model_sha256=parent['acoustic_sha256'],
        data_ready_sha256=sha(root/DATA/'ready.json'),features=['raw_best_cosine','raw_cosine_mean','raw_cosine_std'],
        source='Frozen exact same E+C+N encoder and verified feature caches. No new simulation or encoder training.',
        probes='Raw-score max/min under stored normalized acoustic score, plus deterministic 1/3 and2/3 flat bank indices; no GT.',
        affine='Least-squares raw=mean+std*z over4 probes; max abs residual<=2e-5, std>1e-6. Re-encoding FP probes cached per unique flat pose.',
        numeric_batch_contract='Match original FP dense batch lengths (256 except final cells block); pad sparse probes to original length. Observed query batch lengths unchanged. v1 failed residual2.84e-5; guard tolerance NOT relaxed. v1 code/data retained.',
        selection={k:parent['selection'][k]for k in('train','val')},test_accessed=False)
    immutable_json(out/'config.json',cfg)
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert state['config']['model']['features']=='all'
    model=AcousticEncoder(**state['config']['model']).cuda().eval();model.load_state_dict(state['model'])
    index={e['scene_id']:e for e in read(root/SOURCE/'bank_index.json')['scenes']}
    new_roles={sid:roles for k in(0,1)for sid,roles in read(root/SOURCE/'preprocessing'/f'part{k}/features_ready.json')['features'].items()}
    started=time.perf_counter();receipts=[]
    try:
        for ref in refs[:limit]if limit else refs:
            sid=ref['scene_id'];dest=out/'scenes'/sid;dest.mkdir(parents=True,exist_ok=True)
            if(dest/'receipt.json').exists():
                receipt=read(dest/'receipt.json');assert receipt['config_sha256']==sha(out/'config.json')and sha(dest/'features.npz')==receipt['features_sha256'];receipts.append(receipt);continue
            source=resolve(ref['source_ref'],ssd);assert sha(source/'receipt.json')==ref['source_ref']['receipt_sha256']
            r=read(source/'receipt.json');assert r.get('acoustic_sha256',r.get('acoustic_checkpoint_sha256'))==cfg['model_sha256']
            entry=index[sid];assert entry['split']==ref['split']
            scores=np.load(source/'scores.npy',mmap_mode='r');q=read(source/'queries.json');xy=np.load(source/'xy.npy')
            np.testing.assert_array_equal(xy,np.array(entry['candidate_xz']))
            probes=[];zs=[];max_z=[]
            for row in scores:
                a=row[1].ravel();ids=[int(a.argmax()),int(a.argmin()),len(a)//3,len(a)*2//3]
                probes.append(ids);zs.append(a[ids]);max_z.append(a[ids[0]])
            unique=sorted(set(np.array(probes).ravel().tolist()));fp_items=[]
            for idx in unique:
                cell,yaw=divmod(idx,36);base,roll=recipe(yaw*10);fp_items.append((cell*6+base//10,roll))
            atomic_json(out/'status.json',dict(status='running',stage='verify_feature_caches',scene_id=sid,complete=len(receipts),total=len(refs),test_accessed=False))
            role=arrays(root,entry,new_roles);e,c,n,fr=role['FP'];oe,oc,on,orr=role['Observed']
            assert len(e)==len(xy)*6 and len(e)==len(n)
            lut={p:i for i,p in enumerate(orr['paths'])};obs_items=[]
            for query in q:
                assert query['channel_roll']==recipe(query['world_yaw_deg'])[1]
                obs_items.append((lut[query['observed_rir_path']],query['channel_roll']))
            atomic_json(out/'status.json',dict(status='running',stage='GPU_probe_encoding',scene_id=sid,complete=len(receipts),total=len(refs),fp_probe_poses=len(unique),queries=len(q),test_accessed=False))
            fp=encode_fp_original_shapes(model,e,c,n,fp_items,batch,len(xy));obs=encode(model,oe,oc,on,obs_items,batch)
            np.testing.assert_allclose(np.linalg.norm(fp,axis=1),1,atol=1e-5);np.testing.assert_allclose(np.linalg.norm(obs,axis=1),1,atol=1e-5)
            ilut={idx:i for i,idx in enumerate(unique)}
            raw=np.sum(fp[np.array([[ilut[idx]for idx in pp]for pp in probes])].astype(np.float64)*obs[:,None,:].astype(np.float64),axis=-1)
            z=np.array(zs);mean,std,error=reconstruct(z,raw)
            features=np.column_stack((mean+std*np.array(max_z),mean,std));assert np.all(abs(features[:,:2])<=1.001)
            np.savez_compressed(dest/'features.npz',features=features,probe_ids=np.array(probes),probe_z=z,probe_raw=raw,
                affine_error=error,sample_indices=np.array([v['sample_index']for v in q]),observed_embeddings=obs)
            receipt=dict(scene_id=sid,split=ref['split'],queries=len(q),fp_probe_poses=len(unique),
                features_sha256=sha(dest/'features.npz'),source_score_receipt_sha256=sha(source/'receipt.json'),
                config_sha256=sha(out/'config.json'),max_affine_error=float(error.max()),test_accessed=False)
            atomic_json(dest/'receipt.json',receipt);receipts.append(receipt)
            atomic_json(out/'status.json',dict(status='running',stage='scene_complete',complete=len(receipts),total=len(refs),scene_id=sid,elapsed_s=time.perf_counter()-started,test_accessed=False))
            print('RAW QUALITY',sid,len(q),'max affine error',float(error.max()),'elapsed',round(time.perf_counter()-started,1),flush=True)
            del role,e,c,n,oe,oc,on,fp,obs,scores
        atomic_json(out/'ready.json',dict(scenes=receipts,complete=len(receipts)==len(refs),config_sha256=sha(out/'config.json'),test_accessed=False))
        atomic_json(out/'status.json',dict(status='complete'if len(receipts)==len(refs)else'pilot_complete',scenes=len(receipts),total=len(refs),elapsed_s=time.perf_counter()-started,test_accessed=False))
    except Exception:
        atomic_json(out/'status.json',dict(status='failed',error=traceback.format_exc(),scenes=len(receipts),test_accessed=False));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--ssd',type=Path,required=True);p.add_argument('--batch',type=int,default=256);p.add_argument('--limit',type=int);a=p.parse_args();run(a.root,a.ssd,a.batch,a.limit)

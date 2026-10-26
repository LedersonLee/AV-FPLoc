"""Build exact same-bank frozen scores; test is a separate post-selection step."""
import argparse
from pathlib import Path
import time

import numpy as np
from numpy.lib.format import open_memmap

from acoustic_branch.contracts import atomic_json,sha
from acoustic_branch.data import read
from acoustic_branch.evaluation import select,standardized
from acoustic_branch.pilot_evaluate import embeddings,angular_rays,bank_windows
from .prepare import REL,ACOUSTIC
from .core import features,target_statistics,FEATURE_NAMES


def build(root,vision_root,test=False):
    import torch
    from acoustic_branch.train import AcousticEncoder
    root,vision_root=Path(root),Path(vision_root);out=root/REL
    config=read(out/'config.json');index=read(out/'bank_index.json')
    if test:
        training=read(out/'runs/training_complete.json')
        assert training['status']=='complete' and training['best_sha256']==sha(out/'runs/adaptive/best.pt')
        bank_root=root/ACOUSTIC/'evaluation'
        banks=read(bank_root/'bank_index.json');ready=read(bank_root/'ready.json');entries=banks['scenes']
        assert {e['scene_id'] for e in entries}==set(index['selection']['test'])
        frozen=Path('/ssd1/AV-FPLoc/exp4_acoustic_xy_observability/analysis/hssd_epoch020_evaluation_20261018')
        locations=dict(root=root,vision=vision_root,vision_eval=frozen)
        for key,digest in read(out/'test_input_contract.json')['files'].items():
            scope,relative=key.split(':',1)
            assert sha(locations[scope]/relative)==digest,'Frozen test input changed: '+key
    else:
        ready=read(out/'ready.json');entries=index['scenes'];assert ready['bank_sha256']==config['bank_sha256']
        assert ready['config_sha256']==sha(out/'config.json') and not ready['test_accessed']
    assert ready['status']=='complete'
    checkpoint=root/config['acoustic_checkpoint'];assert sha(checkpoint)==config['acoustic_sha256']
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=AcousticEncoder(**config['acoustic_model']).cuda().eval();model.load_state_dict(state['model'])
    torch.set_num_threads(8)
    cached=[]
    for entry in entries:
        split=entry.get('split','test');scene=entry['scene_id']
        assert (split=='test')==test
        folder=out/'score_cache'/split/scene;folder.mkdir(parents=True,exist_ok=True)
        receipt_path=folder/'receipt.json'
        if receipt_path.exists():
            receipt=read(receipt_path);assert receipt['config_sha256']==sha(out/'config.json')
            for name,digest in receipt['files'].items():assert sha(folder/name)==digest
            cached.append(dict(scene_id=scene,split=split,directory=str(folder.relative_to(root)),receipt_sha256=sha(receipt_path)));continue
        begin=time.monotonic()
        if test:
            old=bank_root/(scene+'_candidate_embeddings_'+config['acoustic_sha256'][:12]+'.npy')
            observed_path=bank_root/(scene+'_observed_embeddings_'+config['acoustic_sha256'][:12]+'.npy')
            candidate=np.load(old,mmap_mode='r');observed=np.load(observed_path)
            queries=read(root/entry['observation_schedule_path'])['queries']
            frozen=Path('/ssd1/AV-FPLoc/exp4_acoustic_xy_observability/analysis/hssd_epoch020_evaluation_20261018')
            assert read(frozen/'protocol.json')['checkpoint_sha256']==config['visual_sha256']
            row_lut={r['sample_index']:i for i,r in enumerate(read(frozen/'rows'/f'{scene}.json'))}
            with np.load(frozen/'depth'/f'{scene}.npz') as d:
                ids=[row_lut[q['sample_index']] for q in queries];depth=d['location'][ids];scale=d['scale'][ids]
        else:
            candidate,observed,queries=embeddings(root,entry,ready,model,torch.device('cuda'),256)
            vision_file=out/'vision'/split/(scene+'.npz')
            vision_receipt=read(vision_file.with_suffix('.json'))
            assert vision_receipt['data_sha256']==sha(vision_file) and vision_receipt['checkpoint_sha256']==config['visual_sha256']
            with np.load(vision_file) as d:
                assert list(d['sample_indices'])==[q['sample_index'] for q in queries]
                depth,scale=d['location'].copy(),d['scale'].copy()
        xy=np.asarray(entry['candidate_xz'],np.float64)
        payload=np.load(vision_root/'UnLoc/structural_targets'/scene/'desdf.npy',allow_pickle=True).item()
        assert payload['max_range_m']==10 and payload['desdf_spacing_m']==.1
        windows=bank_windows(payload,xy)
        # Float64 preserves original baseline score/tie behavior. Torch training
        # converts batches to FP32; selection/report recomputes on float64.
        scores=open_memmap(folder/'scores.tmp.npy',mode='w+',dtype='float64',shape=(len(queries),2,len(xy),36))
        feat=[];expectations=[];truth=[];baselines=[];nearest=[]
        for i,q in enumerate(queries):
            rays,scales=angular_rays(depth[i],scale[i]);v=-(abs(windows-rays)/scales).sum(-1)/40
            a=np.einsum('cyd,d->cy',candidate,observed[i],optimize=True)
            vnorm,anorm=standardized(v),standardized(a)
            scores[i,0]=vnorm;scores[i,1]=anorm
            feat.append(features(v,a,xy,rays))
            gt=[q['centre_xyz_m'][0],q['centre_xyz_m'][2],q['world_yaw_deg']]
            truth.append(gt)
            stats,distance=target_statistics(vnorm,anorm,xy,gt,config['soft_target_xy_sigma_m'],config['soft_target_yaw_sigma_deg'])
            expectations.append(stats);nearest.append(distance)
            choices=select(v,a,config['top_k'])[0]
            baselines.append({method:[*xy[cell].tolist(),yaw*10] for method,(cell,yaw) in choices.items()})
        scores.flush();scores._mmap.close();(folder/'scores.tmp.npy').replace(folder/'scores.npy')
        for name,value in [('features',feat),('target_expectations',expectations),('truth',truth),('xy',xy)]:
            np.save(folder/(name+'.npy'),np.asarray(value))
        atomic_json(folder/'queries.json',queries);atomic_json(folder/'baselines.json',baselines)
        names=['scores.npy','features.npy','target_expectations.npy','truth.npy','xy.npy','queries.json','baselines.json']
        atomic_json(receipt_path,dict(config_sha256=sha(out/'config.json'),feature_names=FEATURE_NAMES,
            scene_id=scene,split=split,queries=len(queries),cells=len(xy),
            candidate_oracle_1m=float((np.asarray(nearest)<=1).mean()),
            max_candidate_quantization_error_m=max(nearest),files={n:sha(folder/n) for n in names},
            test_accessed=test,checkpoint_selection_completed_before_test=test,seconds=time.monotonic()-begin))
        cached.append(dict(scene_id=scene,split=split,directory=str(folder.relative_to(root)),receipt_sha256=sha(receipt_path)))
        atomic_json(out/'score_progress.json',dict(stage='test_scores' if test else 'train_val_scores',scene_id=scene,
            completed_scenes=len(cached),total_scenes=len(entries),updated_unix=time.time()))
        print('score_ready',split,scene,len(queries),len(xy),flush=True)
    atomic_json(out/('test_scores_ready.json' if test else 'scores_ready.json'),dict(status='complete',
        entries=cached,config_sha256=sha(out/'config.json'),feature_names=FEATURE_NAMES,test_accessed=test))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--vision-root',type=Path,required=True);p.add_argument('--test',action='store_true')
    a=p.parse_args();build(a.root,a.vision_root,a.test)

"""Audit frozen local checkpoints on val only, including captured-pose retrieval."""
import argparse
from pathlib import Path

import numpy as np
import torch

from .contracts import atomic_json, sha
from .data import read, load_manifest, check_cache
from .train import AcousticEncoder, PairDataset, loader, pass_epoch, retrieval
from .diagnose import tensors, scores


def run(root, config_path, output):
    config=read(config_path)
    torch.set_num_threads(4)
    queries,digest=load_manifest(root,'val',config)
    locations,_=check_cache(root,config,'val',digest,full=False)
    dataset=PairDataset(queries,locations)
    device=torch.device('cuda')
    result=dict(test_accessed=False,checkpoints={})
    for name in ('best','last'):
        path=root/f'AcousticContrastive/pilots/scene_subset_v1/runs/acoustic_v2/{name}.pt'
        checkpoint=torch.load(path,map_location='cpu',weights_only=False)
        model=AcousticEncoder(**config['model']).to(device)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        sampled=pass_epoch(model,loader(dataset,config,0,False),device,.07)
        captured,_=retrieval(model,dataset,config,device)
        records=[]
        with torch.no_grad():
            for start in range(0,len(queries),32):
                ids=list(range(start,min(start+32,len(queries))))
                o,c=scores(model,tensors(dataset,ids,device))
                s=torch.einsum('bd,bkd->bk',o,c).cpu().numpy()
                winners=s==s.max(-1,keepdims=True)
                credit=winners[:,0]/winners.sum(-1)
                for i,v in zip(ids,credit):
                    q=queries[i]
                    records.append(dict(scene=q['scene_id'],sample_index=q['sample_index'],
                        xyz=q['observed']['xyz'],top1_credit=float(v)))
        result['checkpoints'][name]=dict(epoch=checkpoint['epoch'],sha256=sha(path),
            sampled=sampled,captured_bank=captured,records=records)
        atomic_json(output/'checkpoint_results.json',result)
        print(name,sampled,captured,flush=True)
    atomic_json(output/'checkpoint_status.json',dict(status='complete',test_accessed=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    try:
        run(a.root,a.config,a.output)
    except Exception as e:
        atomic_json(a.output/'checkpoint_status.json',dict(status='failed',error=repr(e)))
        raise

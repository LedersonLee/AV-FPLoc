"""Capacity-only ablation. Frozen processed58 / logit MSE / Train12-Val3.

Data, normalization, targets, seeds, mini-batch order, optimizer and 40epoch
budget are shared. No Test paths are resolved. Original results are preserved.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path

import numpy as np
from acoustic_branch.contracts import atomic_json,immutable_json,sha
from .input_supervision_audit import read
from .response_input_localization import REL as DATA,fit_task

REL='ReliabilityFusion/pilots/capacity_probe_v1'
WIDTHS={'32_16':[32,16],'64_32':[64,32],'128_64':[128,64]}


def prepare(root):
    root=Path(root);out=root/REL;out.mkdir(parents=True,exist_ok=True)
    old=read(root/DATA/'config.json');ready=read(root/DATA/'ready.json')
    assert {r['split']for r in ready['scenes']}=={'train','val'}
    for split,n in [('train',12),('val',3)]:
        ids={r['scene_id']for r in ready['scenes']if r['split']==split}
        assert len(ids)==n and ids==set(old['selection'][split])
    assert not set(old['selection']['train'])&set(old['selection']['val'])
    for ref in ready['scenes']:
        folder=root/DATA/'scenes'/ref['scene_id']
        assert sha(folder/'data.npz')==ref['data_sha256'] and sha(folder/'queries.json')==ref['queries_sha256']
    config={k:old[k]for k in ('selection','seeds','epochs','lr','weight_decay','batch_size','initial_weight','clip_grad_norm','scheduler','early_stop','scene_balanced','target','model_selection','normalization','inputs','counts')}
    config.update(version='v1',input_arm='processed58',input_columns=list(range(58)),hidden_widths=WIDTHS,objectives=['logit_regression'],
        data_rel=DATA,data_ready_sha256=sha(root/DATA/'ready.json'),data_config_sha256=sha(root/DATA/'config.json'),
        objective='MSE of pre-sigmoid logit against frozen logit(interior_target); exactly the previous processed58 supervision',
        no_encoder_updates=True,test_accessed=False,notice='Capacity-only 40epoch matched-budget comparison; Val selects checkpoint including epoch0; not an independent Test or an exhaustive capacity/optimizer search.')
    immutable_json(out/'config.json',config)
    atomic_json(out/'ready.json',dict(source_sha256=sha(root/DATA/'ready.json'),source=DATA,test_accessed=False))
    print('READY',out,flush=True)


def train(root,ssd,workers):
    root,ssd=map(Path,(root,ssd));out=root/REL;cfg=read(out/'config.json')
    assert sha(root/DATA/'ready.json')==cfg['data_ready_sha256']and sha(root/DATA/'config.json')==cfg['data_config_sha256']
    tasks=[]
    for name,hidden in WIDTHS.items():
        for seed in cfg['seeds']:
            tasks.append((str(root),str(ssd),'processed58','logit_regression',seed,dict(output_rel=REL,data_rel=DATA,hidden=hidden,run_name=f'{name}_seed{seed}')))
    runs=[]
    with ProcessPoolExecutor(max_workers=workers)as pool:
        for f in as_completed([pool.submit(fit_task,t)for t in tasks]):
            r=f.result();runs.append(r);atomic_json(out/'status.json',dict(stage='training',done=len(runs),total=6,last=dict(hidden=r['hidden'],seed=r['seed']),test_accessed=False))
    # Same seed MUST see exactly the same shuffled scene/minibatch sequence.
    for seed in cfg['seeds']:
        rr=[r for r in runs if r['seed']==seed]
        assert len(rr)==3 and len({r['batch_order_sha256']for r in rr})==1
        fresh=next(r for r in rr if r['hidden']==[32,16])
        old=read(root/DATA/'runs'/f'processed58_logit_regression_seed{seed}'/'complete.json')
        for phase in ('best','last'):
            assert fresh['evaluated'][phase]['epoch']==old['evaluated'][phase]['epoch']
            for split in('train','val'):
                np.testing.assert_allclose(fresh['evaluated'][phase][split]['loss'],old['evaluated'][phase][split]['loss'],atol=0,rtol=0)
                assert fresh['evaluated'][phase][split]['micro']==old['evaluated'][phase][split]['micro']
                assert fresh['evaluated'][phase][split]['predictions']==old['evaluated'][phase][split]['predictions']
    atomic_json(out/'training_complete.json',dict(runs=6,small_model_reproduced=True,batch_order_matched=True,test_accessed=False))
    atomic_json(out/'status.json',dict(stage='trained',test_accessed=False));print('TRAINED: baseline and batch orders reproduced exactly',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','train']);p.add_argument('--root',type=Path,required=True);p.add_argument('--ssd',type=Path);p.add_argument('--workers',type=int,default=6);a=p.parse_args()
    if a.mode=='prepare':prepare(a.root)
    else:train(a.root,a.ssd,a.workers)

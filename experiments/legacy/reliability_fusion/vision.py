"""Cache frozen epoch20 vision predictions for train/val only."""
import argparse
import os
from pathlib import Path
import sys

import numpy as np

from acoustic_branch.contracts import atomic_json,sha
from acoustic_branch.data import read
from .prepare import REL


class RGBDataset:
    """Top-level/picklable so workers can SPAWN after CUDA initialization."""
    def __init__(self,rows):self.rows=rows
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        import cv2
        image=cv2.imread(self.rows[index]['image_path'],cv2.IMREAD_COLOR)
        if image is None or image.shape[:2]!=(480,640):raise ValueError('Invalid HSSD RGB')
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB).astype(np.float32)/255.
        image=(image-(.485,.456,.406))/(.229,.224,.225)
        return dict(ref_img=np.transpose(image,(2,0,1)).astype(np.float32),ref_mask=np.ones((480,640),np.uint8))


def run(root,vision_root,code_root):
    root,vision_root,code_root=map(Path,(root,vision_root,code_root))
    config=read(root/REL/'config.json');index=read(root/REL/'bank_index.json')
    checkpoint=vision_root/config['visual_checkpoint']
    assert sha(checkpoint)==config['visual_sha256']
    old=vision_root/'UnLoc/train_likelihood/epoch020'
    assert read(old/'protocol.json')['checkpoint_sha256']==config['visual_sha256']
    model=None
    helpers=code_root/'experiments/exp4_acoustic_xy_observability/scripts'
    sys.path[:0]=[str(helpers),str(code_root/'third_party/UnLoc')]
    from train_hssd_unloc import _build_runtime_classes,load_adapter_frame_rows,initialize_loader_worker
    by_scene={}
    for split in ('train','val'):
        for row in load_adapter_frame_rows(vision_root/'UnLoc/frame_adapter_v4_furnished_qa',split):
            by_scene.setdefault(row['scene_id'],[]).append(row)
    for entry in index['scenes']:
        split,scene=entry['split'],entry['scene_id']
        assert split in ('train','val')
        folder=root/REL/'vision'/split;folder.mkdir(parents=True,exist_ok=True)
        path=folder/(scene+'.npz');receipt=path.with_suffix('.json')
        source=read(root/entry['observation_schedule_path'])
        queries=source['queries'];lookup={r['sample_index']:r for r in by_scene[scene]}
        rows=[lookup[q['sample_index']] for q in queries]
        for q,r in zip(queries,rows):np.testing.assert_allclose(r['pose_xzyaw'],[q['centre_xyz_m'][0],q['centre_xyz_m'][2],q['world_yaw_deg']],atol=1e-7,rtol=0)
        if receipt.exists():
            saved=read(receipt)
            assert saved['checkpoint_sha256']==config['visual_sha256'] and saved['data_sha256']==sha(path)
            continue
        if split=='train':
            origin=old/'depth'/f'{scene}.npz'
            assert sha(origin)==read(old/'scenes'/f'{scene}.json')['depth_sha256']
            with np.load(origin) as values:
                lut={int(v):i for i,v in enumerate(values['sample_indices'])};ids=[lut[q['sample_index']] for q in queries]
                location,scale=values['location'][ids],values['scale'][ids]
            reused=True
        else:
            import torch
            from torch.utils.data import DataLoader
            if model is None:
                dataset_class,model_class=_build_runtime_classes()
                torch.set_num_threads(8)
                cwd=Path.cwd();os.chdir(code_root/'third_party/UnLoc')
                try:model=model_class.load_from_checkpoint(str(checkpoint),strict=False).cuda().eval()
                finally:os.chdir(cwd)
            loader=DataLoader(RGBDataset(rows),batch_size=4,shuffle=False,num_workers=8,
                worker_init_fn=initialize_loader_worker,pin_memory=True,multiprocessing_context='spawn')
            locations,scales=[],[]
            with torch.inference_mode():
                for batch in loader:
                    with torch.autocast(device_type='cuda',dtype=torch.float16):
                        mu,b=model.encoder(batch['ref_img'].cuda(non_blocking=True),batch['ref_mask'].cuda(non_blocking=True))[:2]
                    locations.append(mu.float().cpu().numpy());scales.append(b.float().cpu().numpy())
                    if sum(len(a) for a in locations)%64==0:
                        atomic_json(root/REL/'vision_progress.json',dict(scene_id=scene,complete=sum(len(a) for a in locations),total=len(rows)))
            location,scale=np.concatenate(locations),np.concatenate(scales);reused=False
        if location.shape!=(len(queries),40) or not np.isfinite(location).all() or not np.isfinite(scale).all() or not (scale>0).all():
            raise ValueError('Invalid frozen vision output')
        temporary=path.with_suffix('.tmp.npz');np.savez_compressed(temporary,location=location,scale=scale,
            sample_indices=[q['sample_index'] for q in queries]);temporary.replace(path)
        atomic_json(receipt,dict(scene_id=scene,split=split,checkpoint_sha256=config['visual_sha256'],
            observation_schedule_sha256=entry['observation_schedule_sha256'],data_sha256=sha(path),reused_train_cache=reused,
            precision='fp16 encoder; fp32 saved',test_accessed=False))
        print('vision_ready',split,scene,len(queries),'reused',reused,flush=True)
    atomic_json(root/REL/'vision_ready.json',dict(status='complete',test_accessed=False,
        checkpoint_sha256=config['visual_sha256']))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','vision-root','code-root'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.root,a.vision_root,a.code_root)

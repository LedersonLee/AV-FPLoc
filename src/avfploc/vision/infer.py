"""Portable RGB -> UnLoc depth -> full XY/yaw score pipeline."""
import os
import sys
from pathlib import Path
import numpy as np
from avfploc.geometry.rays import angular_rays


def score(depth, scale, ray_bank):
    depth, scale, bank = map(np.asarray, (depth, scale, ray_bank))
    if depth.ndim != 2 or depth.shape != scale.shape or bank.ndim != 2 or bank.shape[1] != 36:
        raise ValueError("Expected depth/scale[Q,40], ray_bank[XY,36]")
    if not all(np.isfinite(x).all() for x in (depth,scale,bank)) or (scale<=0).any():
        raise ValueError("Invalid depth/scale/rays")
    bearings=(np.arange(36)[:,None]+np.arange(-5,6))%36
    windows=bank[:,bearings]
    return np.stack([-np.sum(abs(windows-angular_rays(d,s)[0])/angular_rays(d,s)[1],axis=-1)/40 for d,s in zip(depth,scale)])


def rgb_depth(checkpoint, unloc_root, images, device="cuda", batch=4):
    """Trusted Lightning checkpoint; pinned patched UnLoc plus DA weights required."""
    import cv2
    import torch
    unloc_root=Path(unloc_root).resolve()
    checkpoint=Path(checkpoint).resolve()
    sys.path.insert(0,str(unloc_root))
    from .train import _build_runtime_classes
    _, cls = _build_runtime_classes()
    old=Path.cwd()
    try:
        os.chdir(unloc_root)
        model=cls.load_from_checkpoint(str(checkpoint), strict=True, map_location=device).to(device).eval()
    finally:
        os.chdir(old)
    depths,scales=[],[]
    with torch.inference_mode():
        for start in range(0,len(images),batch):
            values=[]
            for path in images[start:start+batch]:
                image=cv2.imread(str(path))
                if image is None or image.shape[:2]!=(480,640):
                    raise ValueError(f"Expected 480x640 RGB: {path}")
                image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB).astype(np.float32)/255
                values.append(np.transpose((image-(.485,.456,.406))/(.229,.224,.225),(2,0,1)))
            x=torch.as_tensor(np.array(values),dtype=torch.float32,device=device)
            mask=torch.ones((len(x),480,640),dtype=torch.uint8,device=device)
            with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=str(device).startswith('cuda')):
                mu,b=model.encoder(x,mask)[:2]
            depths.append(mu.float().cpu().numpy());scales.append(b.float().cpu().numpy())
    return np.concatenate(depths),np.concatenate(scales)

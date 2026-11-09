"""Common observed-pose metrics; never used by the inference scorer."""
import math
import torch
from track_b.scripts.train_structure import diverse_cpu


def pose_statistics(score,poses,gt,diverse=True):
    delta=poses-gt;dist=delta[:,:2].norm(dim=-1);yaw=torch.atan2(delta[:,2].sin(),delta[:,2].cos()).abs()
    ok=(dist<=1+1e-6)&(yaw<=math.pi/6+1e-6);idx=score.argmax();raw=score.topk(min(20,len(score))).indices
    near=(dist.square()/.35**2+yaw.square()/math.radians(15)**2).argmin();logp=score.log_softmax(-1)
    div=diverse_cpu(score.detach().cpu().numpy(),poses.detach().cpu().numpy()) if diverse else []
    return dict(joint=float(ok[idx]),r1=float(dist[idx]<=1+1e-6),fine=float(dist[idx]<=.1+1e-6),raw20=float(ok[raw].any()),
                diverse20=float(ok[div].any()) if div else None,nll=float(-logp[near]),entropy=float(-(logp.exp()*logp).sum()),
                oracle_joint=float(ok.any()),oracle_fine=float((dist<=.1+1e-6).any()))

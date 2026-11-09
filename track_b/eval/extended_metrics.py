"""Protocol precision, rank and reliability metrics, outside model inference."""
import math
import numpy as np
import torch
from track_b.eval.metrics import pose_statistics

@torch.no_grad()
def extended_pose_statistics(score,poses,gt,diverse=True):
    if score.ndim!=1 or len(score)==0 or poses.shape!=(len(score),3):raise ValueError('Expected nonempty full score vector and N,3 poses')
    out=pose_statistics(score,poses,gt,diverse)
    delta=poses-gt;dist=delta[:,:2].norm(dim=-1);yaw=torch.atan2(delta[:,2].sin(),delta[:,2].cos()).abs();best=score.argmax();order=score.topk(min(20,len(score))).indices
    ok=(dist<=1+1e-6)&(yaw<=math.pi/6+1e-6)
    nearest=(dist.square()/.35**2+yaw.square()/math.radians(15)**2).argmin()
    logp=score.log_softmax(-1);p=logp.exp();center=poses-poses[best];radius=center[:,:2].norm(dim=-1);angle=torch.atan2(center[:,2].sin(),center[:,2].cos()).abs()
    # The confidence region is centered at the prediction, never at GT.
    region=(radius<=1+1e-6)&(angle<=math.pi/6+1e-6)
    out.update(raw1=float(ok[best]),raw5=float(ok[order[:5]].any()),raw10=float(ok[order[:10]].any()),
        strict_05_15=float((dist[best]<=.5+1e-6)&(yaw[best]<=math.radians(15)+1e-6)),
        strict_02_10=float((dist[best]<=.2+1e-6)&(yaw[best]<=math.radians(10)+1e-6)),
        translation_error_m=float(dist[best]),yaw_error_deg=float(yaw[best]*180/math.pi),
        nearest_grid_translation_m=float(dist[nearest]),nearest_grid_yaw_deg=float(yaw[nearest]*180/math.pi),
        true_grid_rank=int((score>score[nearest]).sum())+1,true_grid_score_ties=int((score==score[nearest]).sum()),
        cell_confidence=float(p[best]),joint_region_confidence=float(p[region].sum()),
        normalized_entropy=out['entropy']/math.log(len(score)) if len(score)>1 else 0.,candidate_count=len(score))
    return out

def reliability_statistics(rows,bins=10):
    """Diagnostic reliability only; this routine fits no calibration parameters.

    Equal-width confidence bins use left-closed intervals, final interval includes1.
    Selective risk sorts confidence descending, preserving input order for ties.
    Aggregate within a declared cohort; bootstrap scenes externally for uncertainty.
    """
    if not rows or bins<1:raise ValueError('Nonempty observations and positive bins required')
    confidence=np.array([r['joint_region_confidence'] for r in rows]);correct=np.array([r['joint'] for r in rows])
    if not np.isfinite(confidence).all() or ((confidence<0)|(confidence>1+1e-6)).any():raise ValueError('Invalid confidence')
    indices=np.minimum((confidence*bins).astype(int),bins-1);table=[];ece=0.
    for b in range(bins):
        selected=indices==b;n=int(selected.sum())
        if n:
            c=float(confidence[selected].mean());a=float(correct[selected].mean());ece+=n/len(rows)*abs(c-a)
            table.append(dict(bin=b,count=n,confidence=c,accuracy=a))
        else:table.append(dict(bin=b,count=0,confidence=None,accuracy=None))
    order=np.argsort(-confidence,kind='stable');selective=[]
    for fraction in [.1,.25,.5,.75,1.]:
        n=max(1,math.ceil(fraction*len(rows)));selective.append(dict(requested_coverage=fraction,count=n,actual_coverage=n/len(rows),risk=float(1-correct[order[:n]].mean())))
    return dict(ece=float(ece),bins=table,selective_risk=selective,
        translation_median_m=float(np.median([r['translation_error_m'] for r in rows])),
        translation_p90_m=float(np.quantile([r['translation_error_m'] for r in rows],.9)),
        yaw_median_deg=float(np.median([r['yaw_error_deg'] for r in rows])),
        yaw_p90_deg=float(np.quantile([r['yaw_error_deg'] for r in rows],.9)))

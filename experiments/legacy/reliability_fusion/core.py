"""GT-free continuous features, score fusion and soft-pose supervision."""
import numpy as np
from scipy.special import softmax

from acoustic_branch.evaluation import standardized
from acoustic_branch.reliability_analysis import spatial_stats

FEATURE_NAMES=[*(f'{m}_{f}' for m in ('vision','acoustic') for f in
    ('separated_gap','spread_over_diagonal','peak_mass_1m','xy_entropy','yaw_entropy')),
    'visual_depth_mean','visual_depth_cv','visual_depth_range',
    'map_distance_over_diagonal','yaw_disagreement_fraction','xy_overlap','top50_overlap']


def features(vision,acoustic,xy,depth):
    """No GT, scene ID or absolute coordinate enters the learned gate."""
    xy=np.asarray(xy);depth=np.asarray(depth)
    diagonal=max(float(np.linalg.norm(np.ptp(xy,axis=0))),1.)
    values=[];stats=[]
    for score in (vision,acoustic):
        st,prob,top=spatial_stats(score,xy);stats.append((st,prob,top))
        values.extend([st['separated_gap'],st['top50_rms_m']/diagonal,
            st['peak_mass_1m'],st['entropy'],st['yaw_entropy']])
    values.extend([float(depth.mean()),float(depth.std()/max(abs(depth.mean()),1e-3)),float(np.ptp(depth))])
    vi,ai=[np.unravel_index(s.argmax(),s.shape) for s in (vision,acoustic)]
    values.extend([float(np.linalg.norm(xy[vi[0]]-xy[ai[0]])/diagonal),
        float(abs(((vi[1]-ai[1])*10+180)%360-180)/180),
        float(np.sqrt(stats[0][1]*stats[1][1]).sum()),
        len(set(stats[0][2])&set(stats[1][2]))/min(50,len(xy))])
    values=np.asarray(values,np.float32)
    if values.shape!=(len(FEATURE_NAMES),) or not np.isfinite(values).all():raise ValueError('Invalid gate feature vector')
    return values


def target_statistics(v,a,xy,truth,xy_sigma=.2,yaw_sigma=10.):
    """Sufficient statistics for exact soft-target CE; no dense label cache."""
    distance=np.linalg.norm(np.asarray(xy)-np.asarray(truth)[:2],axis=1)
    angle=abs((np.arange(v.shape[1])*10-truth[2]+180)%360-180)
    weights=softmax((-(distance[:,None]/xy_sigma)**2/2-(angle[None]/yaw_sigma)**2/2).ravel()).reshape(v.shape)
    return np.array([(weights*v).sum(),(weights*a).sum()],np.float64),float(distance.min())


def prediction(v,a,xy,weight):
    score=(1-float(weight))*v+float(weight)*a
    cell,yaw=np.unravel_index(score.argmax(),score.shape)
    return [*np.asarray(xy[cell]).tolist(),int(yaw)*10]


def metrics(poses,truth):
    poses,truth=np.asarray(poses),np.asarray(truth)
    distance=np.linalg.norm(poses[:,:2]-truth[:,:2],axis=1)
    angle=abs((poses[:,2]-truth[:,2]+180)%360-180)
    return dict(queries=len(poses),recall_01m=float((distance<=.1).mean()),recall_05m=float((distance<=.5).mean()),
        recall_1m=float((distance<=1).mean()),recall_1m_30deg=float(((distance<=1)&(angle<=30)).mean()),
        median_xy_m=float(np.median(distance)),median_yaw_deg=float(np.median(angle)))

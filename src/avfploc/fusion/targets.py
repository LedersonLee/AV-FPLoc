import numpy as np
from scipy.special import softmax
def target_statistics(v,a,xy,truth,xy_sigma=.2,yaw_sigma=10.):
    """Sufficient statistics for exact soft-target CE; no dense label cache."""
    distance=np.linalg.norm(np.asarray(xy)-np.asarray(truth)[:2],axis=1)
    angle=abs((np.arange(v.shape[1])*10-truth[2]+180)%360-180)
    weights=softmax((-(distance[:,None]/xy_sigma)**2/2-(angle[None]/yaw_sigma)**2/2).ravel()).reshape(v.shape)
    return np.array([(weights*v).sum(),(weights*a).sum()],np.float64),float(distance.min())

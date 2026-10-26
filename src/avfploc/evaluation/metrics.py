import numpy as np


def compare(predicted, truth, fixed=None):
    p,t=map(np.asarray,(predicted,truth))
    if p.shape!=t.shape or p.ndim!=2 or p.shape[1]!=3 or not len(p) or not np.isfinite(p).all() or not np.isfinite(t).all():
        raise ValueError("Expected finite matching nonempty [Q,3] poses")
    d=np.linalg.norm(p[:,:2]-t[:,:2],axis=1);a=abs((p[:,2]-t[:,2]+180)%360-180)
    out={f'recall_{r:g}m':float((d<=r).mean()) for r in (.1,.5,1)}
    out.update({f'yaw_within_{r}deg':float((a<=r).mean()) for r in (10,30,50)})
    out.update(queries=len(p),recall_1m_30deg=float(((d<=1)&(a<=30)).mean()),median_xy_m=float(np.median(d)),median_yaw_deg=float(np.median(a)))
    if fixed is not None:
        f=np.asarray(fixed)
        if f.shape!=p.shape or not np.isfinite(f).all():raise ValueError("Invalid reference poses")
        fd=np.linalg.norm(f[:,:2]-t[:,:2],axis=1);fa=abs((f[:,2]-t[:,2]+180)%360-180)
        for kind,before,after in [('xy',fd<=1,d<=1),('joint',(fd<=1)&(fa<=30),(d<=1)&(a<=30))]:
            out['repair_vs_fixed_'+kind]=int((~before&after).sum());out['harm_vs_fixed_'+kind]=int((before&~after).sum())
    return out

"""Thin metric-grid adapter of the frozen colleague NMS/evidence/gate code.

No GT enters proposal extraction or selection. Only held-out VAL chooses the
three scalars. Keep this protocol separate from our official fixed 1:1 fusion.
"""
from functools import lru_cache
import importlib.util
from pathlib import Path
import sys
import types

import numpy as np

GRID = [(w, s, tv) for w in (.5, 1., 2.) for s in (.02, .05, .1)
        for tv in (.005, .02, .05, .1, .2)]
PROTOCOL = dict(n_modes=10, nms_radius_m=1.5, local_radius_m=.5, quantile=.9,
                visual_evidence='centre', acoustic_evidence='quantile',
                rule='continuous', acoustic_transform='standard', tau_a=-2.,
                val_selection='pooled VAL distance<1m; earliest grid tie',
                output='Selected visual mode centre XY and its visual argmax yaw')


@lru_cache(None)
def reference(directory):
    directory=Path(directory)
    namespace='_hssd_colleague_mode_'+str(abs(hash(str(directory.resolve()))))
    for name in (namespace, namespace+'.likelihood'):
        module=types.ModuleType(name);module.__path__=[];sys.modules[name]=module
    def load(name,path):
        spec=importlib.util.spec_from_file_location(name,path)
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module
        spec.loader.exec_module(module);return module
    modes=load(namespace+'.modes',directory/'modes.py')
    fusion=load(namespace+'.likelihood.mode_fusion',directory/'mode_fusion.py')
    return modes,fusion


def evidence(vision,acoustic_xy,xy,spacing,refdir):
    """Unchanged raw visual log score; no full-bank z-score before margin gate."""
    modes,fusion=reference(str(refdir))
    vision=np.asarray(vision,dtype=np.float64);acoustic_xy=np.asarray(acoustic_xy,dtype=np.float64)
    xy=np.asarray(xy,dtype=np.float64)
    if vision.shape!=(len(xy),36) or acoustic_xy.shape!=(len(xy),):
        raise ValueError('Mismatched XY/yaw candidate support')
    if not (np.isfinite(vision).all() and np.isfinite(acoustic_xy).all()):
        raise ValueError('Nonfinite evidence')
    grid=np.rint((xy-xy.min(0))/spacing).astype(int)
    np.testing.assert_allclose(grid*spacing,xy-xy.min(0),atol=1e-6,rtol=0)
    config=modes.ModeConfig(nms_radius_m=1.5,n_modes=10,local_radius_m=.5,quantile=.9)
    # exp(logscore - max) has exactly the posterior's order; the normalization
    # constant cancels in log evidence standardization and the top-two gap.
    field=vision.max(1)
    selected=modes.extract_modes(field,grid[:,1],grid[:,0],spacing,config)
    discs=modes.local_discs(selected,grid[:,1],grid[:,0],spacing,.5)
    audio=modes.aggregate(acoustic_xy,selected,discs,config)['quantile']
    visual=field[selected]
    return dict(cells=selected,visual=visual,audio=audio,disc_sizes=np.array([len(d) for d in discs]),
                yaw=vision[selected].argmax(1),spacing=spacing)


def choose(row,scalars,refdir):
    _,fusion=reference(str(refdir));w,s,tv=map(float,scalars)
    cfg=fusion.ModeFusionConfig(vis_evidence='centre',ac_evidence='quantile',rule='continuous',
        weight=w,sigmoid_scale=s,tau_v=tv,tau_a=-2.,ac_transform='standard')
    k,active=fusion.choose(row['visual'],row['audio'],cfg)
    if len(row['cells'])<2:gamma=0.
    else:
        gv=1/(1+np.exp(np.clip((fusion.visual_ambiguity(row['visual'])-tv)/s,-700,700)))
        ga=1/(1+np.exp(np.clip(-(fusion.acoustic_discriminativeness(row['audio'])+2)/s,-700,700)))
        gamma=w*gv*ga
    score=fusion.standardise(row['visual'])+gamma*fusion.standardise(row['audio'])
    if int(score.argmax())!=k:raise ValueError('Wrapper/reference rule differs')
    return dict(index=int(k),cell=int(row['cells'][k]),yaw_bin=int(row['yaw'][k]),
                gamma=float(gamma),audio_active=bool(active),mode_scores=score)


def select_val(rows,truth,xy,refdir):
    """Only validation labels enter selection; no test argument or checkpoint fit."""
    curve=[];best=None
    for scalars in GRID:
        errors=[]
        for scene,values in rows.items():
            for i,row in enumerate(values):
                cell=choose(row,scalars,refdir)['cell']
                errors.append(float(np.linalg.norm(xy[scene][cell]-truth[scene][i,:2])))
        recall=float((np.array(errors)<1).mean())
        value=dict(scalars=list(scalars),val_recall_1m_strict=recall,queries=len(errors))
        curve.append(value)
        if best is None or recall>best['val_recall_1m_strict']:best=value
    return dict(selected=best,curve=curve,selection_split='val',test_accessed=False)

"""Temperature-controlled soft-pose CE and acceptable-peak margin objectives."""
import math
import torch
from torch.nn import functional as F

from .model import Gate


def objective(weight, vision, acoustic, expectations, temperature=1., inside=None, margin=.1):
    if not math.isfinite(temperature) or temperature<=0:raise ValueError('Positive finite temperature required')
    score=(1-weight[:,None])*vision+weight[:,None]*acoustic
    if inside is None:
        target=(1-weight)*expectations[:,0]+weight*expectations[:,1]
        return torch.logsumexp(score/temperature,dim=1)-target/temperature
    if not torch.all(inside.any(1)) or not torch.all((~inside).any(1)):
        raise ValueError('Both acceptable and unacceptable candidates required')
    positive=score.masked_fill(~inside,-torch.inf).amax(1)
    negative=score.masked_fill(inside,-torch.inf).amax(1)
    return F.softplus((negative-positive+margin)/temperature)*temperature


def initialize(model, weight):
    if not 0<weight<1:raise ValueError('Gate initialization must be interior')
    logit=math.log(weight/(1-weight))
    with torch.no_grad():
        if model.constant:model.logit.fill_(logit)
        else:model.net[-1].bias.fill_(logit)


def arm(name):
    temperature=.25 if name.endswith('T025') else .5 if name.endswith('T05') else 1.
    return dict(name=name,temperature=temperature,constant=name.startswith('constant'),
        half_initialization=name.endswith('_half'),
        margin_kind='joint' if name.endswith('margin_joint') else 'xy' if name.endswith('margin_xy') else None)

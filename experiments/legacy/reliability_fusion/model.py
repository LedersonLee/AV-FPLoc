"""Small query-level continuous gate; both localization branches stay frozen."""
import torch
from torch import nn


class Gate(nn.Module):
    def __init__(self,features,constant=False,hidden=(32,16)):
        super().__init__();self.constant=constant
        if len(hidden)!=2 or any(not isinstance(h,int) or isinstance(h,bool) or h<=0 for h in hidden):
            raise ValueError('Two positive integer hidden widths required')
        if constant:self.logit=nn.Parameter(torch.zeros(()))
        else:
            self.net=nn.Sequential(nn.Linear(features,hidden[0]),nn.ReLU(),nn.Linear(hidden[0],hidden[1]),nn.ReLU(),nn.Linear(hidden[1],1))
            nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)

    def forward(self,x):
        return torch.sigmoid(self.logit.expand(x.shape[0]) if self.constant else self.net(x).squeeze(-1))


def loss(weight,v,a,target_expectations):
    """Exact soft pose CE: logZ minus expected GT score, not a modality label."""
    score=(1-weight[:,None])*v+weight[:,None]*a
    expectation=(1-weight)*target_expectations[:,0]+weight*target_expectations[:,1]
    return (torch.logsumexp(score,dim=1)-expectation).mean()

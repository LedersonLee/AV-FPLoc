"""Declared adaptations, not a reproduction of unavailable original code."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class EchoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        layers=[]
        for cin,cout,k,s in [(6,32,8,4),(32,64,4,2),(64,64,3,1)]:
            layers.extend([nn.Conv2d(cin,cout,k,s),nn.BatchNorm2d(cout),nn.ReLU()])
        self.conv=nn.Sequential(*layers)
        self.projection=nn.Linear(64*2*4,512)
    def forward(self,x):
        return self.projection(F.adaptive_avg_pool2d(self.conv(x),(2,4)).flatten(1))


class SpatialEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=nn.Sequential(nn.Conv2d(18,32,5,2,2),nn.GELU(),nn.Conv2d(32,64,3,2,1),nn.GELU(),
                                nn.Conv2d(64,64,3,2,1),nn.GELU())
        # Preserve ordered frequency/time tokens, including early/late position.
        self.position=nn.Parameter(torch.randn(1,32,64)*.02)
        self.projection=nn.Linear(64,64)
    def forward(self,x):
        z=F.adaptive_avg_pool2d(self.conv(x),(4,8)).flatten(2).transpose(1,2)
        return self.projection(z)+self.position


class GeometryNet(nn.Module):
    def __init__(self,architecture='spatial',head='regression',directions=36,bins=80):
        super().__init__()
        self.architecture=architecture;self.head_kind=head;self.bins=bins
        self.encoder=EchoEncoder() if architecture=='echo' else SpatialEncoder()
        theta=torch.arange(directions)*2*math.pi/directions
        self.register_buffer('bearings',torch.stack([theta.cos(),theta.sin()],-1))
        out=1 if head=='regression' else bins
        if architecture=='echo':
            self.decoder=nn.Sequential(nn.Linear(512,256),nn.ReLU(),nn.Linear(256,directions*out))
        else:
            self.query=nn.Linear(2,64)
            self.attention=nn.MultiheadAttention(64,4,batch_first=True)
            self.decoder=nn.Sequential(nn.Linear(64,128),nn.GELU(),nn.Linear(128,out))
        self.register_buffer('centres',(torch.arange(bins)+.5)*20/bins)
    def forward(self,x):
        z=self.encoder(x)
        if self.architecture=='echo':raw=self.decoder(z).reshape(len(x),len(self.bearings),-1)
        else:
            q=self.query(self.bearings)[None].expand(len(x),-1,-1)
            v,_=self.attention(q,z,z,need_weights=False)
            raw=self.decoder(v+q)
        if self.head_kind=='regression':return 20*raw[...,0].sigmoid(),None
        return (raw.softmax(-1)*self.centres).sum(-1),raw


def masked_loss(mean,logits,target,mask):
    if not mask.any():raise ValueError('Batch has no observed structural targets')
    if logits is None:return torch.log1p((mean-target).abs())[mask].mean()
    bins=logits.shape[-1]
    # Two-bin interpolation preserves sub-bin targets and permits multiple modes.
    u=(target*bins/20-.5).clamp(0,bins-1)
    low=u.floor().long();high=(low+1).clamp_max(bins-1);w=u-low
    lp=logits.log_softmax(-1)
    loss=-(1-w)*lp.gather(-1,low[...,None])[...,0]-w*lp.gather(-1,high[...,None])[...,0]
    return loss[mask].mean()

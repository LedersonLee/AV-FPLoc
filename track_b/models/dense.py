"""Reference-free exact candidate scorer with independently trainable encoders.

Map tensor channels: free, unknown/blocked. A black pixel is not asserted to be
physical wall. Candidate patches are defined in the map's metric local frame.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class DenseScorer(nn.Module):
    def __init__(self, width=32, mode='joint'):
        super().__init__()
        if mode not in ('joint','visual','acoustic','concat','noninteraction'): raise ValueError(mode)
        self.mode=mode; self.width=width
        self.visual=nn.Sequential(nn.Conv2d(3,16,5,2,2),nn.GELU(),nn.Conv2d(16,width,3,2,1),nn.GELU())
        self.acoustic=nn.Sequential(nn.Conv1d(6,16,15,4,7),nn.GELU(),nn.Conv1d(16,width,9,4,4),nn.GELU())
        self.floorplan=nn.Sequential(nn.Conv2d(2,16,3,padding=1),nn.GELU(),nn.Conv2d(16,width,3,padding=1),nn.GELU())
        # Ordered 3x3 metric patch preserves heading without absolute world coordinates.
        self.fp_query=nn.Linear(width*9,width)
        self.v_attn=nn.MultiheadAttention(width,4,batch_first=True)
        self.a_attn=nn.MultiheadAttention(width,4,batch_first=True)
        self.v_head=nn.Sequential(nn.Linear(width*2,width),nn.GELU(),nn.Linear(width,1))
        self.a_head=nn.Sequential(nn.Linear(width*2,width),nn.GELU(),nn.Linear(width,1))
        self.gate=nn.Sequential(nn.Linear(width*3,width),nn.GELU(),nn.Linear(width,1),nn.Sigmoid())
        self.joint=nn.Sequential(nn.Linear(width*5,width*2),nn.GELU(),nn.Linear(width*2,1))
        self.register_buffer('stencil',torch.tensor([[x,y] for y in (-1.,0.,1.) for x in (-1.,0.,1.)]))

    def encode(self,rgb,rir,floorplan):
        # Preserve horizontal order in visual tokens and early/late time order in audio.
        v=F.adaptive_avg_pool2d(self.visual(rgb),(1,16)).squeeze(2).transpose(1,2)
        a=F.adaptive_avg_pool1d(self.acoustic(rir),32).transpose(1,2)
        # Deterministic positional embeddings, no query ID or scene embedding.
        def pos(t):
            p=torch.linspace(-1,1,t.shape[1],device=t.device)[:,None]
            f=torch.arange(1,t.shape[2]+1,device=t.device)[None,:]
            return t+torch.sin(p*f)*.1
        return pos(v),pos(a),self.floorplan(floorplan)

    def score_chunk(self,v,a,fp,poses,calibration):
        origin=calibration['origin']; spacing=calibration['spacing']
        frame_angle=torch.as_tensor(calibration.get('map_angle',0.),device=poses.device)
        fc,fs=frame_angle.cos(),frame_angle.sin()
        rotation=torch.stack([torch.stack([fc,-fs]),torch.stack([fs,fc])])
        local_xy=(poses[:,:2]-origin)@rotation
        angle=poses[:,2]-frame_angle; c,s=angle.cos(),angle.sin()
        x=self.stencil[:,0]; y=self.stencil[:,1]
        dx=c[:,None]*x-s[:,None]*y;dy=s[:,None]*x+c[:,None]*y
        coords=local_xy[:,None,:]+torch.stack([dx,dy],-1)
        pix=coords/spacing
        wh=torch.tensor([fp.shape[-1]-1,fp.shape[-2]-1],device=fp.device)
        grid=(pix/wh*2-1).unsqueeze(0).expand(fp.shape[0],-1,-1,-1)
        patch=F.grid_sample(fp,grid,align_corners=True,padding_mode='zeros').permute(0,2,3,1).flatten(2)
        q=self.fp_query(patch)
        if self.mode in ('concat','noninteraction'):
            vp=v.mean(1,keepdim=True).expand_as(q);ap=a.mean(1,keepdim=True).expand_as(q)
        else:
            vp=self.v_attn(q,v,v,need_weights=False)[0];ap=self.a_attn(q,a,a,need_weights=False)[0]
        sv=self.v_head(torch.cat([q,vp],-1)).squeeze(-1)
        sa=self.a_head(torch.cat([q,ap],-1)).squeeze(-1)
        if self.mode=='visual': return sv
        if self.mode=='acoustic': return sa
        if self.mode=='noninteraction': return sv+sa
        joint=self.joint(torch.cat([q,vp,ap,vp*ap,abs(vp-ap)],-1)).squeeze(-1)
        if self.mode=='concat': return joint
        gate=self.gate(torch.cat([q,vp,ap],-1)).squeeze(-1)
        return sv+gate*sa+joint

    def forward(self,rgb,rir,floorplan,calibration,candidate_grid,chunk_size=512):
        v,a,fp=self.encode(rgb,rir,floorplan)
        return torch.cat([self.score_chunk(v,a,fp,p,calibration) for p in candidate_grid.split(chunk_size)],dim=1)


def dense_loss(scores,poses,gt,sigma_xy=.35,sigma_yaw=math.radians(15)):
    dxy=(poses[None,:,:2]-gt[:,None,:2]).square().sum(-1)
    da=torch.atan2(torch.sin(poses[None,:,2]-gt[:,None,2]),torch.cos(poses[None,:,2]-gt[:,None,2]))
    target=torch.softmax(-dxy/(2*sigma_xy**2)-da.square()/(2*sigma_yaw**2),-1)
    return -(target*torch.log_softmax(scores,-1)).sum(-1).mean()


def topk(scores,poses,k=20,radius=None,yaw_radius=math.pi/6):
    order=torch.argsort(scores,descending=True,stable=True); chosen=[]
    for idx in order.tolist():
        if radius is not None and chosen:
            d=poses[chosen]-poses[idx]; da=torch.atan2(d[:,2].sin(),d[:,2].cos()).abs()
            if ((d[:,:2].norm(dim=-1)<=radius)&(da<=yaw_radius)).any(): continue
        chosen.append(idx)
        if len(chosen)==min(k,len(poses)): break
    return torch.tensor(chosen,device=scores.device)


def rasterize(scores,valid_xy,yaw_bins):
    """Scatter valid-only scores to the full masked SE(2) field."""
    if scores.shape[-1]!=int(valid_xy.sum())*yaw_bins:raise ValueError('Incomplete bank')
    field=scores.new_full((*scores.shape[:-1],*valid_xy.shape,yaw_bins),-torch.inf)
    field[...,valid_xy,:]=scores.reshape(*scores.shape[:-1],int(valid_xy.sum()),yaw_bins)
    return field

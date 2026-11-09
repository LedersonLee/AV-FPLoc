"""Equal-parameter full-field token interaction and active non-attention controls."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from track_b.models.structure import StructureMatcher,visible_rays

class Retrieval(nn.Module):
    def __init__(self,d=32):
        super().__init__();self.q=nn.Linear(d,d);self.k=nn.Linear(d,d,bias=False);self.v=nn.Linear(d,d);self.out=nn.Linear(d,d)
    def forward(self,q,tokens,attention):
        q=self.q(q);k=self.k(tokens);v=self.v(tokens)
        if attention:
            weight=torch.softmax(torch.einsum('nd,bld->bnl',q,k)/math.sqrt(q.shape[-1]),-1)
            x=weight@v
        else:
            # Every Q/K/V parameter remains active; no token attention or modality exchange.
            x=torch.tanh(q[None]+k.mean(1)[:,None])*v.mean(1)[:,None]
        return self.out(x)

class InteractionMatcher(StructureMatcher):
    def __init__(self,mode='attention'):
        super().__init__()
        if mode not in ('attention','concat','separable'):raise ValueError(mode)
        self.mode=mode;d=32
        self.v_projection=nn.Linear(64,d);self.a_projection=nn.Linear(64,d);self.q_projection=nn.Linear(48,d)
        self.v_retrieval=Retrieval(d);self.a_retrieval=Retrieval(d)
        self.gate=nn.Sequential(nn.Linear(3*d,32),nn.GELU(),nn.Linear(32,1))
        self.joint_head=nn.Sequential(nn.Linear(5*d,64),nn.GELU(),nn.Linear(64,1))
    def encode_observations(self,rgb,spectrum):
        vc=self.visual.conv(rgb);ac=self.acoustic.conv(spectrum)
        v=self.visual.head(F.adaptive_avg_pool2d(vc,self.visual.pooled).flatten(1));a=self.acoustic.head(F.adaptive_avg_pool2d(ac,self.acoustic.pooled).flatten(1))
        vt=self.v_projection(F.adaptive_avg_pool2d(vc,(1,16)).flatten(2).transpose(1,2))
        at=self.a_projection(F.adaptive_avg_pool2d(ac,(4,16)).flatten(2).transpose(1,2))
        # Fixed token coordinates: visual columns and acoustic frequency/time.
        frequency=torch.arange(1,33,device=rgb.device)[None,:]
        vp=torch.linspace(-1,1,16,device=rgb.device)[:,None]
        ap=torch.stack(torch.meshgrid(torch.linspace(-1,1,4,device=rgb.device),torch.linspace(-1,1,16,device=rgb.device),indexing='ij'),-1).reshape(-1,2)
        vt=vt+.1*torch.sin(vp*frequency)
        at=at+.1*(torch.sin(ap[:,:1]*frequency)+torch.cos(ap[:,1:]*frequency))
        return v,a,vt,at
    def encode(self,rgb,spectrum,floorplan):
        return (*self.encode_observations(rgb,spectrum),self.floorplan(floorplan))
    def score_chunk(self,v,a,vt,at,feature,candidates,cal):
        fp=self.fp_descriptors(feature,candidates,cal);q=self.q_projection(torch.cat([fp,candidates[:,3:]],-1))
        weight=.25+.75*candidates[:,19:35]
        return self.score_context(v,a,vt,at,fp,q,weight,cal.get('hfov_deg',106.2602))
    def score_context(self,v,a,vt,at,fp,q,weight,hfov,acoustic_fp=None,acoustic_scale=None):
        vp=self.v_retrieval(q,vt,self.mode=='attention');ap=self.a_retrieval(q,at,self.mode=='attention');qq=q[None].expand(len(v),-1,-1)
        sv=self.match(v,fp,weight*visible_rays(hfov,device=v.device));sa=self.match(a,fp if acoustic_fp is None else acoustic_fp,weight,log_scale=acoustic_scale)
        if self.mode=='separable':
            # Sum of visual-only and acoustic-only functions conditional on the map.
            jv=self.joint_head(torch.cat([qq,vp,vp,vp.square(),vp.abs()],-1)).squeeze(-1)
            ja=self.joint_head(torch.cat([qq,ap,ap,ap.square(),ap.abs()],-1)).squeeze(-1)
            gate=2*torch.sigmoid(self.gate(torch.cat([qq,ap,ap],-1)).squeeze(-1));joint=.5*(jv+ja)
        else:
            gate=2*torch.sigmoid(self.gate(torch.cat([qq,vp,ap],-1)).squeeze(-1))
            joint=self.joint_head(torch.cat([qq,vp,ap,vp*ap,(vp-ap).abs()],-1)).squeeze(-1)
        return .85*sv+.15*gate*sa+.1*joint
    def forward(self,rgb,spectrum,floorplan,cal,candidates,chunk_size=2048,checkpoint_chunks=True,return_aux=False):
        v,a,vt,at,feature=self.encode(rgb,spectrum,floorplan);parts=[]
        def compute(v,a,vt,at,feature,p):return self.score_chunk(v,a,vt,at,feature,p,cal)
        for p in candidates.split(chunk_size):
            if self.training and torch.is_grad_enabled() and checkpoint_chunks:parts.append(checkpoint(compute,v,a,vt,at,feature,p,use_reentrant=False))
            else:parts.append(compute(v,a,vt,at,feature,p))
        field=torch.cat(parts,-1)
        return (field,v,a) if return_aux else field

    @torch.no_grad()
    def prepare_map(self,floorplan,cal,candidates,chunk_size=2048):
        """Frozen-model inference cache: learned FP descriptors/queries, never RIRs.

        Invalidate after changing model parameters, map, candidate bank or calibration.
        """
        if self.training:raise RuntimeError('Map cache requires eval mode')
        feature=self.floorplan(floorplan);fp=torch.cat([self.fp_descriptors(feature,p,cal) for p in candidates.split(chunk_size)])
        q=self.q_projection(torch.cat([fp,candidates[:,3:]],-1));weight=.25+.75*candidates[:,19:35]
        return dict(fp=fp,q=q,weight=weight,hfov=cal.get('hfov_deg',106.2602))
    @torch.no_grad()
    def forward_cached(self,rgb,spectrum,cache,chunk_size=2048):
        if self.training:raise RuntimeError('Cached inference requires eval mode')
        v,a,vt,at=self.encode_observations(rgb,spectrum);fields=[]
        for start in range(0,len(cache['fp']),chunk_size):
            part=slice(start,start+chunk_size)
            fields.append(self.score_context(v,a,vt,at,cache['fp'][part],cache['q'][part],cache['weight'][part],cache['hfov']))
        return torch.cat(fields,-1)

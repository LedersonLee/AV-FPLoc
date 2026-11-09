"""Foundation token interaction preserving independently learned modality maps."""
import copy
import torch
from torch import nn
from track_b.models.interaction import InteractionMatcher
from track_b.models.foundation import FoundationVisualAdapter

class FoundationInteractionMatcher(InteractionMatcher):
    def __init__(self,mode='attention'):
        super().__init__(mode);self.visual=FoundationVisualAdapter()
        self.acoustic_floorplan=copy.deepcopy(self.floorplan)
        self.acoustic_fp_head=copy.deepcopy(self.fp_head)
        self.acoustic_log_scale=nn.Parameter(self.log_scale.detach().clone())

    def initialize_independent(self,visual,acoustic):
        self.visual.load_state_dict(visual.visual.state_dict());self.acoustic.load_state_dict(acoustic.acoustic.state_dict())
        self.floorplan.load_state_dict(visual.floorplan.state_dict());self.fp_head.load_state_dict(visual.fp_head.state_dict())
        self.acoustic_floorplan.load_state_dict(acoustic.floorplan.state_dict());self.acoustic_fp_head.load_state_dict(acoustic.fp_head.state_dict())
        with torch.no_grad():
            self.log_scale.copy_(visual.log_scale);self.acoustic_log_scale.copy_(acoustic.log_scale)
            # Exact F1 starting field; new interaction paths become active after output updates.
            self.gate[-1].weight.zero_();self.gate[-1].bias.zero_()
            self.joint_head[-1].weight.zero_();self.joint_head[-1].bias.zero_()

    def encode(self,rgb,spectrum,floorplan):
        feature=torch.cat([self.floorplan(floorplan),self.acoustic_floorplan(floorplan)],1)
        return (*self.encode_observations(rgb,spectrum),feature)

    def score_chunk(self,v,a,vt,at,feature,candidates,cal):
        fp=self.fp_descriptors(feature[:,:16],candidates,cal)
        afp=self.fp_descriptors(feature[:,16:],candidates,cal,head=self.acoustic_fp_head)
        q=self.q_projection(torch.cat([fp,candidates[:,3:]],-1));weight=.25+.75*candidates[:,19:35]
        return self.score_context(v,a,vt,at,fp,q,weight,cal.get('hfov_deg',106.2602),afp,self.acoustic_log_scale)

    def map_descriptors(self,floorplan,cal,candidates,chunk_size=8192):
        vf=self.floorplan(floorplan);af=self.acoustic_floorplan(floorplan)
        v=torch.cat([self.fp_descriptors(vf,p,cal) for p in candidates.split(chunk_size)])
        a=torch.cat([self.fp_descriptors(af,p,cal,head=self.acoustic_fp_head) for p in candidates.split(chunk_size)])
        return v,a

    @torch.no_grad()
    def prepare_map(self,floorplan,cal,candidates,chunk_size=2048):
        if self.training:raise RuntimeError('Map cache requires eval mode')
        fp,afp=self.map_descriptors(floorplan,cal,candidates,chunk_size)
        return dict(fp=fp,acoustic_fp=afp,q=self.q_projection(torch.cat([fp,candidates[:,3:]],-1)),weight=.25+.75*candidates[:,19:35],hfov=cal.get('hfov_deg',106.2602))

    @torch.no_grad()
    def forward_cached(self,rgb,spectrum,cache,chunk_size=2048):
        if self.training:raise RuntimeError('Cached inference requires eval mode')
        v,a,vt,at=self.encode_observations(rgb,spectrum);fields=[]
        for start in range(0,len(cache['fp']),chunk_size):
            part=slice(start,start+chunk_size)
            fields.append(self.score_context(v,a,vt,at,cache['fp'][part],cache['q'][part],cache['weight'][part],cache['hfov'],cache['acoustic_fp'][part],self.acoustic_log_scale))
        return torch.cat(fields,-1)

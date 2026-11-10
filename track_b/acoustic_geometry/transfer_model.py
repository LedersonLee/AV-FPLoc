"""Replace only the observation branch; retain Track B's16-ray full-bank scorer."""
import torch
from torch import nn
from .models import GeometryNet


class TransferAcoustic(nn.Module):
    def __init__(self,architecture='spatial'):
        super().__init__()
        self.geometry=GeometryNet(architecture,'regression')
        latent=512 if architecture=='echo' else 32*64
        self.readout=nn.Sequential(nn.Flatten(),nn.Linear(latent,128),nn.GELU(),nn.Linear(128,16),nn.Sigmoid())

    def forward(self,x,auxiliary=False):
        z=self.geometry.encoder(x)
        score_features=self.readout(z)
        return (score_features,self.geometry.decode(z)[0]) if auxiliary else score_features

    def configure(self,arm,pretrained=None):
        if pretrained is not None:self.geometry.load_state_dict(pretrained,strict=True)
        for p in self.geometry.parameters():p.requires_grad=False
        for p in self.geometry.encoder.parameters():p.requires_grad=arm!='frozen'
        if arm=='joint':
            for p in self.geometry.parameters():p.requires_grad=True
        for p in self.readout.parameters():p.requires_grad=True
        self.frozen_encoder=arm=='frozen'

    def train(self,mode=True):
        super().train(mode)
        if getattr(self,'frozen_encoder',False):self.geometry.encoder.eval()
        return self

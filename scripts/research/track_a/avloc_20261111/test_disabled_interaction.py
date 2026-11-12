"""Dependency checks of the disabled forward path, independent of trained accuracy."""
import unittest
import torch
from torch import nn
from model import AVLoc,Expert,moment_match

class TokensFixture(nn.Module):
 def forward(self,x):
  pattern=torch.arange(128,dtype=x.dtype)[None,None,:]/128
  value=torch.sin(x.reshape(len(x),1,1)*pattern)
  return value.expand(-1,4,-1),value.transpose(1,2).unsqueeze(-1).expand(-1,-1,15,20)

class DisabledTest(unittest.TestCase):
 def test_separate_pass_dependency_and_equal_fusion(self):
  torch.set_num_threads(2);torch.manual_seed(42)
  # Exercise AVLoc.forward exactly, replacing only expensive input encoders.
  m=AVLoc.__new__(AVLoc);nn.Module.__init__(m);m.arm='A2_disabled'
  m.visual=TokensFixture();m.audio=TokensFixture();m.joint_expert=Expert('joint');m.eval()
  calls=[]
  hook=m.joint_expert.register_forward_hook(lambda mod,args,out:calls.append(tuple(x.detach().clone() for x in out)))
  def forward(v,a):
   calls.clear()
   with torch.no_grad():out=m(torch.tensor([v]),torch.tensor([a]))['AV']
   self.assertEqual(len(calls),2)
   return out,list(calls)
  fused,original=forward(1.,2.)
  _,changed_audio=forward(1.,9.)
  _,changed_visual=forward(9.,2.)
  for component in range(2):
   torch.testing.assert_close(original[0][component],changed_audio[0][component],rtol=0,atol=0)
   torch.testing.assert_close(original[1][component],changed_visual[1][component],rtol=0,atol=0)
  means=torch.stack([original[0][0],original[1][0]],-1);scales=torch.stack([original[0][1],original[1][1]],-1)
  expected=moment_match(means,scales,torch.full_like(means,.5))
  for x,y in zip(fused,expected):torch.testing.assert_close(x,y,rtol=0,atol=0)
  self.assertTrue(any(float((x-y).abs().max())>1e-5 for x,y in zip(original[1],changed_audio[1])))
  self.assertTrue(any(float((x-y).abs().max())>1e-5 for x,y in zip(original[0],changed_visual[0])))
  hook.remove()

if __name__=='__main__':unittest.main()

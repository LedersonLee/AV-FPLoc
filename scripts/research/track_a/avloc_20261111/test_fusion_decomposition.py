import unittest
import numpy as np
from decompose_fusion_predictions import decompose
from matching import FOV
class DecompositionTest(unittest.TestCase):
 def test_exact_inside_outside_partition(self):
  v=np.arange(22).reshape(2,11).astype(float);a=np.arange(72).reshape(2,36).astype(float)+100;f=a+100
  vb=v+1;ab=a+1;fb=f+1;out=decompose(v,vb,a,ab,f,fb);outside=np.setdiff1d(np.arange(36),FOV)
  for suffix,vx,ax,fx in [('mu',v,a,f),('scale',vb,ab,fb)]:
   np.testing.assert_array_equal(out['visual_inside_audio_outside']['F_'+suffix][:,FOV],vx)
   np.testing.assert_array_equal(out['visual_inside_audio_outside']['F_'+suffix][:,outside],ax[:,outside])
   np.testing.assert_array_equal(out['fused_fov']['V_'+suffix],fx[:,FOV])
 def test_invalid_scale(self):
  with self.assertRaises(AssertionError):decompose(np.ones((1,11)),np.zeros((1,11)),*[np.ones((1,36))]*4)
if __name__=='__main__':unittest.main()

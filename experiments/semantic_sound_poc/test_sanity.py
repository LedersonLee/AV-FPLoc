import unittest,inspect,json
from pathlib import Path
import numpy as np
from model import wrap,bearing,posterior,softmax

class Tests(unittest.TestCase):
 def setUp(self):
  self.c=json.loads(Path(__file__).with_name('config.json').read_text());self.p=np.array([[0,0,0],[0,0,90.],[3,0,180.]])
  self.support=[dict(label='kitchen',integration=[[1,1.2,0],[2,1.2,0]]),dict(label='bathroom',integration=[[0,1.2,1],[0,1.2,2]])]
 def test_wrap_and_body_world(self):
  np.testing.assert_allclose(wrap(np.deg2rad([0,360,720,-360])),0,atol=1e-12)
  np.testing.assert_allclose(bearing(self.p[:2],np.array([[1,1,0],[0,1,1]])),[[0,np.pi/2],[-np.pi/2,0]],atol=1e-12)
 def test_normalization_silence(self):
  for e in ['silence','running_water']:
   for m in ['B0','B1','B2','B3']:
    p=posterior(self.p,[2,1,0],e,.3,self.support,self.c,m);self.assertAlmostEqual(sum(p),1);self.assertTrue(np.isfinite(p).all())
    if e=='silence' or m=='B1':np.testing.assert_array_equal(p,softmax(np.array([2,1,0])))
 def test_truth_absent(self):
  self.assertEqual(list(inspect.signature(posterior).parameters),['poses','visual_scores','event','angle','support','c','method','labels'])
  self.assertNotIn('generation',inspect.getsource(posterior));p=posterior(self.p,[0,0,0],'running_water',0,self.support,self.c)
  for r in self.support:r['generation']=[[1e6,1e6,1e6]];r['truth']=[-1e9]*3
  np.testing.assert_array_equal(p,posterior(self.p,[0,0,0],'running_water',0,self.support,self.c))
 def test_tdoa_rotation(self):
  angles=np.deg2rad(np.array(self.c['mic_clockwise_angles_deg']));src=np.array([10,1.5,0]);center=np.array([0,1.5,0]);dif=[]
  for yaw in [0,np.pi]:
   mic=center+.05*np.stack([np.cos(angles+yaw),np.zeros(6),np.sin(angles+yaw)],1);dist=np.linalg.norm(mic-src,axis=1);dif.append((dist-dist[0])/343)
  np.testing.assert_allclose(dif[0],-dif[1],atol=1e-6)
 def test_polygon_transform(self):
  from matplotlib.path import Path as Poly
  points=np.array([[1,1],[3,3]]);self.assertEqual(Poly([[0,0],[2,0],[2,2],[0,2]]).contains_points(points).tolist(),[True,False]);origin=np.array([-15.4,-17.55]);p=points*.01+origin;np.testing.assert_allclose((p-origin)/.01,points)
if __name__=='__main__':unittest.main()

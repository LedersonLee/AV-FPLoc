import json
import numpy as np
from fusion_predictions import equal_fusion
from matching import FOV
v=np.full((2,11),2.);vb=np.ones((2,11));a=np.full((2,36),6.);ab=np.full((2,36),2.)
mu,b,w=equal_fusion((v,vb),(a,ab));outside=[k for k in range(36) if k not in FOV]
np.testing.assert_array_equal(mu[:,FOV],4);np.testing.assert_allclose(b[:,FOV],np.sqrt(4.5));np.testing.assert_array_equal(mu[:,outside],6);np.testing.assert_array_equal(b[:,outside],2);assert (w[:,outside,0]==0).all()
m4,b4,w4=equal_fusion((v,vb),(a,ab),(np.full_like(a,10),np.full_like(ab,3)))
np.testing.assert_array_equal(m4[:,FOV],6);np.testing.assert_allclose(b4[:,FOV],np.sqrt(10));np.testing.assert_array_equal(m4[:,outside],8);np.testing.assert_allclose(b4[:,outside],np.sqrt(8.5))
# Identical distributions retain their mean and scale; disagreement increases scale.
u=np.full((2,11),6.);ub=np.full((2,11),2.);m2,b2,w2=equal_fusion((u,ub),(a,ab),(a,ab));np.testing.assert_array_equal(m2,6);np.testing.assert_array_equal(b2,2);np.testing.assert_allclose(w2.sum(-1),1)
# Directional indexing: FoV uses wrap-around indices31..35,0..5.
v3=np.tile(np.arange(11),(2,1));m3,_,_=equal_fusion((v3,vb),(a,ab));np.testing.assert_array_equal(m3[:,FOV],(v3+6)/2)
print(json.dumps(dict(status='pass',checks=['analytic mixture mean/variance','audio-only outside FoV','equal experts invariant','weight normalization','FoV wrap-around direction ordering']),indent=2))

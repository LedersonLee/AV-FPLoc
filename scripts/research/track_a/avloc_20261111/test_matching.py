"""Independent numeric checks of geometry scoring and diagnostic definitions."""
import json
import numpy as np
from scipy.stats import laplace
from matching import laplace_field,pose_values,proposals,diagnostic,calibration,FOV
rng=np.random.default_rng(19);world=rng.uniform(.1,20,(7,36));mu=rng.uniform(.1,20,36);b=rng.uniform(.1,2,36)
s=laplace_field(world,mu,b,xy_chunk=2)
ref=np.array([sum(laplace.logpdf(world[i,(j+k)%36],loc=mu[k],scale=b[k]) for k in range(36)) for i in range(7) for j in range(36)])
np.testing.assert_allclose(s,ref,rtol=1e-14,atol=1e-11)
v=laplace_field(world,mu[FOV],b[FOV],FOV,xy_chunk=3)
refv=np.array([sum(laplace.logpdf(world[i,(j+k)%36],loc=mu[k],scale=b[k]) for k in FOV) for i in range(7) for j in range(36)])
np.testing.assert_allclose(v,refv,rtol=1e-14,atol=1e-11)
xy=np.column_stack([np.arange(7)*3,np.zeros(7)]);target_id=3*36+35;truth=pose_values(xy,[target_id])[0]
exact=laplace_field(world,world[3,(np.arange(36)+35)%36],np.ones(36)*.01)
assert exact.argmax()==target_id
ids=proposals(exact,xy);d=diagnostic(exact,xy,truth,candidate_ids=ids);assert d['joint'] and d['fine'] and d['top5_nms']
# Circular yaw correctness and GT-only diagnostics cannot change the decision.
d2=diagnostic(exact,xy,[*truth[:2],-10],candidate_ids=ids);assert d2['yaw_error_deg']==0
bad=diagnostic(exact,xy,[100,100,90],candidate_ids=ids);assert bad['top1_id']==d['top1_id'] and not bad['gt_region_exists'] and np.isinf(bad['gt_region_nll'])
shift=diagnostic(exact+10000,xy,truth,candidate_ids=ids);assert abs(shift['gt_region_mass']-d['gt_region_mass'])<1e-10
hot=diagnostic(exact,xy,truth,temperature=1000,candidate_ids=ids);assert hot['entropy_normalized']>d['entropy_normalized'] and hot['top1_id']==d['top1_id']
assert np.array_equal(proposals(np.zeros(252),xy),proposals(np.zeros(252),xy))
c=calibration([.2,.2,.8,.8],[0,1,1,1],bins=5);assert abs(c['ece']-.25)<1e-12
print(json.dumps(dict(status='pass',checks=['Eq2 versus scipy Laplace 36 rays','11 FoV ray index/heading convention','known exact pose recovery','circular yaw','GT diagnostic separation','empty GT region explicit infinity','score offset invariance','temperature changes entropy not top1','stable tie NMS','manual ECE'],max_logscore_difference=float(abs(s-ref).max())),indent=2))

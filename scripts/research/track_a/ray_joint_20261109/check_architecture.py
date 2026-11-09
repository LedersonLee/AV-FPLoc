import json,torch
from pathlib import Path
from joint_models import JointRay
from train_acoustic import RayModel
P=Path('/output');torch.set_num_threads(2);torch.manual_seed(9)
x=torch.randn(2,32,16,16);torch.testing.assert_close(torch.nn.AvgPool2d(4,4)(x),torch.nn.AdaptiveAvgPool2d((4,4))(x),rtol=0,atol=0)
a=torch.randn(2,18,32,32);v=torch.randn(2,40,128);out={}
for kind in ['vision','acoustic','concat','attention','disabled']:
 m=JointRay(kind).eval();y=m(a,v);assert y.shape==(2,36)
 if kind=='vision':torch.testing.assert_close(y,m(a+torch.randn_like(a),v),rtol=0,atol=0)
 if kind=='acoustic':torch.testing.assert_close(y,m(a,v+torch.randn_like(v)),rtol=0,atol=0)
 out[kind]=sum(p.numel() for p in m.parameters())
assert out['attention']==out['disabled'];assert abs(out['concat']/out['attention']-1)<.1
for k in [4,8,12,36]:
 for kind in ['ecn','embedding','raw','chirp']:
  m=RayModel(kind,k);z=torch.randn(2,552 if kind=='ecn' else 128) if kind in ['ecn','embedding'] else a;assert m(z).shape==(2,k)
(P/'architecture_verification.json').write_text(json.dumps(dict(status='PASS',joint_parameters=out,pool_equivalence='exact CPU floating values on fixed16x16',unimodal_isolation='other modality perturbation leaves output exactly unchanged',raw_chirp='same RayModel CNN architecture',scope='Architecture invariants only, not trained result validation'),indent=2));print(out)

import json
import numpy as np
from temperature_calibration import curve,crossfit
from matching import diagnostic
xy=np.array([[0,0],[3,0]]);scores=np.linspace(-10,0,72);truth=[3,0,350];temps=np.array([.1,1,10.]);values,meta=curve(scores,xy,truth,temps)
for i,t in enumerate(temps):
 d=diagnostic(scores,xy,truth,temperature=t);np.testing.assert_allclose(values[i],[d['nearest_grid_nll'],d['gt_region_nll'],d['joint_confidence'],d['entropy_normalized'],d['gt_region_mass']],atol=1e-12)
shift,_=curve(scores+1000,xy,truth,temps);np.testing.assert_allclose(values,shift,atol=1e-11)
scenes=np.array([f'{i}_a' for i in range(10) for _ in range(2)]);v=np.tile(values,(20,1,1));correct=np.ones(20,bool);a=crossfit(scenes,v,correct,temps)
# Changing held-fold truth-dependent objective cannot alter that fold's selected temperature.
held=a['folds'][0]['held_scenes'];changed=v.copy();changed[np.isin(scenes,held),:,0]=np.array([100,20,0]);b=crossfit(scenes,changed,correct,temps)
assert a['folds'][0]['index']==b['folds'][0]['index']
for f in a['folds']:assert not set(f['fit_scenes'])&set(f['held_scenes'])
# Both scenes in the same source family must remain together.
s2=np.array([f'{i//2}_{i%2}' for i in range(20)]);a2=crossfit(s2,v,correct,temps)
for f in a2['folds']:assert not {s.split('_')[0] for s in f['fit_scenes']}&{s.split('_')[0] for s in f['held_scenes']}
print(json.dumps(dict(status='pass',checks=['temperature statistics match independent diagnostic path','offset invariance','held fold cannot choose its own temperature','fit/held scene separation','source-family grouping']),indent=2))

import json
import numpy as np
from geometry_observability import diagnose
xy=np.array([[0,0],[3,0]]);profile=np.arange(36)*.1+.5;world=np.stack([profile,profile]);truth=np.array([3,0,0])
a=diagnose(world,xy,profile,truth);assert abs(a['joint_geometry_margin_m'])<1e-12 and not a['perfect_profile_joint'];assert a['incorrect_profiles_within_5cm_of_best_correct']>=1
world[0]+=4;b=diagnose(world,xy,profile,truth);assert b['joint_geometry_margin_m']>0 and b['perfect_profile_joint'] and b['perfect_profile_fine']
# A GT pose outside the bank's tolerance region must not receive a fabricated margin.
c=diagnose(world,xy,profile,[100,100,0]);assert c['joint_geometry_margin_m'] is None and not c['correct_region_exists']
print(json.dumps(dict(status='pass',checks=['identical distant profiles are ambiguous','known distinct profile has positive margin','tie can yield wrong pose with zero ray error','missing correct region explicit']),indent=2))

import json
import numpy as np
from paired_summary import paired,align,matched_keys
# Deliberately unequal scene sizes distinguish macro from micro averaging.
sc=['a']*2+['b']*10;m=np.array([[1,1]+[0]*10,[1,1]+[0]*10]);b=np.array([0,0]+[1]*10);d=paired(m,b,sc)
assert d['macro_delta']==0 and abs(d['micro_delta']+2/3)<1e-12
assert d['macro_ci95']==[-1.,1.] and d['scenes']==2 and d['queries']==12
assert d['repair_count_seed_mean']==2 and d['harm_count_seed_mean']==10
# Replicating queries within each scene does not create extra independent units.
r=paired(np.repeat(m,3,axis=1),np.repeat(b,3),np.repeat(sc,3));assert r['macro_ci95']==d['macro_ci95'] and r['scenes']==2
assert paired(m[::-1],b,sc)==d
rows=[dict(scene='a',query_id=1),dict(scene='b',query_id=2)];assert align(rows[::-1],matched_keys(rows))==rows
for bad in [rows[:1],rows+rows[:1]]:
 try:align(bad,matched_keys(rows))
 except ValueError:pass
 else:raise AssertionError('Invalid cohort accepted')
print(json.dumps(dict(status='pass',checks=['unequal scene-size macro/micro distinction','paired clustered uncertainty','seed averaging no pseudo-replication','query replication does not create scenes','strict ID alignment','duplicate/missing rows rejected']),indent=2))

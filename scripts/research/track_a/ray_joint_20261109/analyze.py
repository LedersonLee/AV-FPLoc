import json,collections
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;pr=json.loads((P/'PROTOCOL.json').read_text());sc=pr['training_scenes'];boot=np.random.default_rng(20261109).integers(0,12,(20000,12));j=json.loads((P/'candidate_evaluation.json').read_text());rows=j['records'];rays=j['ray_records']
def vec(rr,k):return np.array([np.mean([r[k] for r in rr if r['scene']==s]) for s in sc])
def ci(v):return dict(mean=float(v.mean()),ci95=np.quantile(v[boot].mean(1),[.025,.975]).tolist(),scene_values=v.tolist())
def pick(group,kind,k=36,control='matched',method='visual_geometry',eval_k=None):return [r for r in rows if r['group']==group and r['kind']==kind and r['trained_k']==k and r['eval_k']==(eval_k or k) and r['control']==control and r['method']==method]
contrasts={}
for tag,a,b in [('chirp-raw',pick('acoustic','chirp'),pick('acoustic','raw')),('attention-concat',pick('joint','attention'),pick('joint','concat')),('attention-disabled',pick('joint','attention'),pick('joint','disabled')),('attention-acoustic',pick('joint','attention'),pick('joint','acoustic')),('attention-late',pick('joint','attention'),pick('joint','late')),('raw-sorted',pick('acoustic','raw'),pick('acoustic','sorted')),('raw-isotropic',pick('acoustic','raw'),pick('acoustic','isotropic'))]:contrasts[tag]={k:ci(vec(a,k)-vec(b,k)) for k in ['joint','fine','pair_accuracy']}
summary={}
for group,kind,k in sorted(set((r['group'],r['kind'],r['trained_k']) for r in rows)):
 rr=pick(group,kind,k);summary[f'{group}/{kind}/K{k}']={m:ci(vec(rr,m)) for m in ['joint','fine','R1','pair_accuracy','ranking_auroc','high_ambiguity_joint','repair','harm']}
ray_summary={}
for group,kind,k,ek in sorted(set((r['group'],r['kind'],r['trained_k'],r['eval_k']) for r in rays)):
 rr=[r for r in rays if (r['group'],r['kind'],r['trained_k'],r['eval_k'],r['control'])==(group,kind,k,ek,'matched')];ray_summary[f'{group}/{kind}/K{k}/subset{ek}']={m:ci(vec(rr,m)) for m in ['mae','median','p90','rank']}
out=dict(scope='Exploratory reused New12; seeds averaged within scene;20k paired scene bootstrap; no multiplicity correction.',contrasts=contrasts,summary=summary,ray_summary=ray_summary);(P/'analysis.json').write_text(json.dumps(out,indent=2));print(json.dumps({n:{k:v for k,v in rr['joint'].items() if k!='scene_values'} for n,rr in contrasts.items()},indent=2))

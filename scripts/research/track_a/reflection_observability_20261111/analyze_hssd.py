"""Six frozen controls, all candidate windows, scene-level paired uncertainty."""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
from metrics import measure,average,bootstrap

def eligible(h):return [p for p in h['paths'] if min(p['times_ms'])>=2 and max(p['times_ms'])<=79]
def full(raw):
 x=raw[:,48:3841].astype(float);e=x*x;n=np.linalg.norm(x,axis=-1);c=x@x.T/np.maximum(n[:,None]*n[None,:],1e-30)
 return dict(power_ratio=1.,peak_ratio=float(e.max(-1).mean()/e.mean()),sharpness=float(np.mean(e.max(-1)/e.mean(-1))),coherence=float(np.mean(abs(c[np.triu_indices(6,1)]))),timing_residual_ms=None,peak_error_ms=None)
def main():
 a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);args=a.parse_args();root=args.root;output=root/'hssd_metrics';output.mkdir(exist_ok=True);start=time.perf_counter()
 for source in sorted((root/'hssd').glob('*.json')):
  dst=output/source.name
  if dst.exists():continue
  data=json.loads(source.read_text());queries=data['queries'];raws=[]
  for q in queries:
   path=Path(q['raw_path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==q['raw_sha256'];x=np.roll(np.load(path),q['roll'],axis=0);assert x.shape[1]>=3841;raws.append(x)
  donors={seed:np.roll(np.arange(len(queries)),int(np.random.default_rng(seed).integers(1,len(queries)))) for seed in [42,43,44]}
  results=[];pathrows=[];counts=dict(total=0,supported=0,unknown=0,outside_analysis=0)
  for qi,q in enumerate(queries):
   raw=raws[qi];rngs={seed:np.random.default_rng(seed+qi*1000) for seed in donors};hyp=q['hypotheses']
   for h in hyp:
    k=h['rank'];pp=eligible(h);counts['total']+=len(h['paths']);counts['outside_analysis']+=len(h['paths'])-len(pp)
    for support in [True,False]:
     selected=[p for p in pp if p['supported']==support];counts['supported' if support else 'unknown']+=len(selected)
     if not selected:continue
     for half in [.25,.5,1.]:
      aligned=[measure(raw,p['times_ms'],half) for p in selected]
      shifted=[measure(raw,1+half+(np.array(p['times_ms'])+sign*5-(1+half))%(79-2*half),half) for p in selected for sign in [-1,1]]
      random=[];donor=[]
      for seed,rng in rngs.items():
       for p in selected:
        center=np.mean(p['times_ms']);delta=np.asarray(p['times_ms'])-center
        random.append(measure(raw,np.full(6,rng.uniform(2+np.max(abs(delta)),79-np.max(abs(delta))))+delta,half))
        donor.append(measure(raws[donors[seed][qi]],p['times_ms'],half))
      # Reassign query waveform to each fixed window: exact scene time/distance
      # marginal preserved, including candidate/path counts and support strata.
      other=eligible(hyp[(k+1)%20]);other=[p for p in other if p['supported']==support]
      arms=dict(aligned=average(aligned),random=average(random),shifted=average(shifted),other_candidate=average([measure(raw,p['times_ms'],half) for p in other]),matched_marginal=average(donor),full_early=full(raw))
      results.append(dict(query=qi,rank=k,supported=support,half_ms=half,path_count=len(selected),arms=arms))
      if half==.5:
       tt=np.array([np.mean(p['times_ms']) for p in pp])
       for p,m in zip(selected,aligned):
        pathrows.append(dict(query=qi,rank=k,supported=support,distance_m=p['distance_m'],bearing_deg=p['bearing_deg'],overlap=int(np.sum(abs(tt-np.mean(p['times_ms']))<1))>1,**m))
  result=dict(scene=data['scene'],counts=counts,records=results,path_strata=pathrows,raw_lengths=[x.shape[1] for x in raws],exact_marginal_permutation=True,donor_offsets={str(s):int(d[0]) for s,d in donors.items()})
  dst.write_text(json.dumps(result));print(data['scene'],counts,flush=True)
 allresults=[json.loads(p.read_text()) for p in sorted(output.glob('*.json'))];summary=[]
 for support in [True,False]:
  for half in [.25,.5,1.]:
   for ranks in ['all20','top1']:
    for control in ['random','shifted','other_candidate','matched_marginal','full_early']:
     for metric in ['power_ratio','peak_ratio','sharpness','coherence','timing_residual_ms']:
      paired=[];scene_values=[]
      for r in allresults:
       records=[x for x in r['records'] if x['supported']==support and x['half_ms']==half and (ranks=='all20' or x['rank']==0) and x['arms'][control] is not None and x['arms'][control][metric] is not None];qs=sorted({x['query'] for x in records})
       if not qs:continue
       av=np.mean([np.mean([x['arms']['aligned'][metric] for x in records if x['query']==q]) for q in qs]);cv=np.mean([np.mean([x['arms'][control][metric] for x in records if x['query']==q]) for q in qs]);paired.append(av-cv);scene_values.append(dict(scene=r['scene'],aligned=float(av),control=float(cv),queries=len(qs)))
      if paired:summary.append(dict(supported=support,half_ms=half,ranks=ranks,control=control,metric=metric,difference=bootstrap(paired),aligned_mean=float(np.mean([x['aligned'] for x in scene_values])),control_mean=float(np.mean([x['control'] for x in scene_values])),per_scene=scene_values))
 strata=[]
 for field in ['supported','overlap','distance','bearing']:
  groups={}
  for r in allresults:
   local={}
   for p in r['path_strata']:
    key=str(p[field]) if field in ['supported','overlap'] else ('<1m' if p['distance_m']<1 else '1-3m' if p['distance_m']<3 else '>=3m') if field=='distance' else str(int((p['bearing_deg']%360)//90)*90)
    local.setdefault(key,[]).append(p)
   for key,ps in local.items():groups.setdefault(key,[]).append(dict(scene=r['scene'],paths=len(ps),power_ratio=float(np.mean([x['power_ratio'] for x in ps])),coherence=float(np.mean([x['coherence'] for x in ps])),timing_residual_ms=float(np.mean([x['timing_residual_ms'] for x in ps]))))
  strata.extend(dict(field=field,group=k,scenes=v) for k,v in groups.items())
 result=dict(scenes=len(allresults),queries=sum(len(r['raw_lengths']) for r in allresults),counts={k:sum(r['counts'][k] for r in allresults) for k in allresults[0]['counts']},raw_length_samples=[min(min(r['raw_lengths']) for r in allresults),max(max(r['raw_lengths']) for r in allresults)],contrasts=summary,strata=strata,seconds=time.perf_counter()-start,test_accessed=False)
 (root/'hssd_summary.json').write_text(json.dumps(result,indent=2));print('COMPLETE',result['counts'],result['seconds'])
if __name__=='__main__':main()

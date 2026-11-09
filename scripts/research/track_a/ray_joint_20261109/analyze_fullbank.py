"""Require complete full-bank outputs before reporting scene-level paired contrasts."""
from pathlib import Path
import json,contextlib,io
import numpy as np
with contextlib.redirect_stdout(io.StringIO()):
 from analyze_extensions import summarize,difference
P=Path(__file__).resolve().parent;W=P.parent.parent

def main():
 primary=json.loads((P/'fullbank_evaluation.json').read_text())['records'];cond=json.loads((P/'conditioned_fullbank_results.json').read_text())['records'];assert len(primary)==2880,len(primary);assert len(cond)==1728,len(cond);assert len(list((P/'fullbank').rglob('*.json')))==576;assert len(list((P/'conditioned_fullbank').rglob('*.json')))==96
 metrics=['joint','fine','R1','top5','top20','repair','harm','preservation'];s=summarize(primary,['group','kind','trained_k','eval_k','method'],metrics);c=summarize(cond,['group','kind','control','method','fusion'],metrics);contrasts={};conditioned={}
 for a,b in [('acoustic/chirp','acoustic/raw'),('joint/attention','joint/concat'),('joint/attention','joint/disabled'),('joint/attention','joint/acoustic'),('acoustic/raw','acoustic/isotropic'),('acoustic/raw','acoustic/sorted')]:
  for method in ['geometry','visual_geometry']:contrasts[a+' minus '+b+'/'+method]=difference(s[a+'/36/36/'+method],s[b+'/36/36/'+method])
 for group,kind in [('acoustic','raw'),('acoustic','chirp'),('acoustic','embedding'),('joint','attention')]:
  for method in ['geometry','visual_geometry']:
   base=f'{group}/{kind}/matched/uniform/{method}';p=s[f'{group}/{kind}/36/36/{method}'];u=c[base]
   for metric in ['joint','fine','R1','top5','top20']:assert p[metric]['scene_values']==u[metric]['scene_values'],(base,metric)
   for matcher in ['uncertainty','anchor_selection']:
    key=f'{group}/{kind}/matched/{matcher}/{method}';conditioned[key+' minus uniform']=difference(c[key],u)
   for matcher in ['uniform','uncertainty','anchor_selection']:
    for control in ['donor17','shuffled']:
     key=f'{group}/{kind}/matched/{matcher}/{method}';other=f'{group}/{kind}/{control}/{matcher}/{method}';conditioned[key+' minus '+control]=difference(c[key],c[other])
 prior=json.loads((W/'experiments/directional_structure_20261109/fullbank_results.json').read_text())['records'];baseline=summarize([r for r in prior if r['arm'] in ['visual','cosine','visual_cosine']],['arm'],metrics)
 for key in s:
  if key.endswith('/36/36/visual_geometry'):
   for arm in baseline:contrasts[key+' minus baseline '+arm]=difference(s[key],baseline[arm])
 out=dict(primary_records=len(primary),conditioned_records=len(cond),primary=s,conditioned=c,contrasts=contrasts,conditioned_contrasts=conditioned,baselines=baseline,uniform_cross_implementation_exact=True,scope='New12, seeds averaged within scene, 20k paired scene bootstrap; exploratory unadjusted CIs. Fullbank anchor extension differs from fixed20 pair Borda. Strong B2+GEOQ framework baseline uses historical support/ranking candidates and is reported separately.')
 (P/'fullbank_analysis.json').write_text(json.dumps(out,indent=2));print('FULLBANK COMPLETE AND UNIFORM CROSSCHECK PASSED',len(primary),len(cond))
if __name__=='__main__':main()
